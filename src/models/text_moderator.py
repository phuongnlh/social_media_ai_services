import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException
from googletrans import Translator
import re
import logging
import time
import os
from typing import Dict, List, Union, Tuple, Optional
from functools import lru_cache
import json
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LRUCache:
    """Limited size LRU cache with persistence capability"""
    def __init__(self, capacity: int = 1000, cache_file: Optional[str] = None):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.cache_file = cache_file
        
        # Load cache from file if exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert to OrderedDict
                    for k, v in data.items():
                        self.cache[k] = v
                logger.info(f"Loaded {len(self.cache)} entries from cache file {cache_file}")
            except Exception as e:
                logger.warning(f"Error loading cache from {cache_file}: {e}")
    
    def get(self, key: str) -> Optional[float]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: float) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
    
    def save(self) -> None:
        """Save cache to file if cache_file is specified"""
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(self.cache), f, ensure_ascii=False)
                logger.info(f"Saved {len(self.cache)} entries to cache file {self.cache_file}")
            except Exception as e:
                logger.warning(f"Error saving cache to {self.cache_file}: {e}")
        else:
            logger.debug("No cache file specified, skipping save")

class TextModerator:
    def __init__(
        self, 
        english_model: str = "martin-ha/toxic-comment-model", 
        threshold: float = 0.5,
        cache_dir: Optional[str] = None,
        batch_size: int = 16,
        use_quantization: bool = False
    ):
        """
        Initialize text moderation model with improved caching and performance
        
        Args:
            english_model: Model for toxicity classification
            threshold: Toxicity threshold
            cache_dir: Directory to store cache files
            batch_size: Batch size for inference
            use_quantization: Whether to use quantized model for faster inference
        """
        # Setup device and performance parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.toxic_threshold = threshold
        self.auto_save_counter = 0
        self.auto_save_interval = 5
        
        # Setup cache paths
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.translation_cache_file = os.path.join(cache_dir, "translation_cache.json")
            self.prediction_cache_file = os.path.join(cache_dir, "prediction_cache.json")
        else:
            self.translation_cache_file = None
            self.prediction_cache_file = None
            
        # Load model with performance optimizations
        logger.info(f"Loading toxicity model: {english_model}")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(english_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(english_model).to(self.device)
        
        # Optimize model for inference
        self.model.eval()
        if use_quantization and self.device.type == "cuda":
            logger.info("Using quantized model for faster inference")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Initialize translator and caches
        self.translator = Translator()
        self.translation_cache = LRUCache(capacity=5000, cache_file=self.translation_cache_file)
        self.prediction_cache = LRUCache(capacity=10000, cache_file=self.prediction_cache_file)
        
        # Setup predefined toxic words (language-specific)
        self._initialize_predefined_toxic_words()
        
    def _initialize_predefined_toxic_words(self):
        """Initialize predefined toxic words in multiple languages"""
        # Vietnamese toxic words
        vietnamese_toxic = [
            "vl", "vcl", "dm", "đm", "cc", "cl", "dmm", "đéo",
            "đĩ", "địt", "lồn", "buồi", "cặc", "lon", "loz", 
            "đụ", "đít", "đốn mạt", "ngu", "súc vật", "chó đẻ", "dcm"
        ]
        
        # English toxic words (helps avoid unnecessary translations)
        english_toxic = [
            "fuck", "shit", "asshole", "bitch", "cunt", "dick",
            "pussy", "whore", "slut", "bastard", "motherfucker",
            "f*ck", "s**t", "a**hole", "b***h", "c**t", "d**k",
            "p****", "w****", "s***", "b*****", "m*****"
        ]
        
        # Cache predefined words with high toxicity scores
        for word in vietnamese_toxic:
            normalized = self._normalize_word(word)
            self.translation_cache.put(normalized, word)
            self.prediction_cache.put(normalized, 1.0)
            
        for word in english_toxic:
            normalized = self._normalize_word(word)
            self.translation_cache.put(normalized, word)
            self.prediction_cache.put(normalized, 1.0)
    
    def moderate(self, text: str) -> Dict[str, Union[str, float, List[Dict[str, Union[str, float]]]]]:
        """
        Moderate text with improved processing and context awareness
        
        Args:
            text: Text to moderate
            
        Returns:
            Dictionary with moderation results including censored text
        """
        start_time = time.time()
        
        # Handle empty text
        if not text or not text.strip():
            result = {"label": "normal", "censored_text": "", "processing_time": 0}
            logger.debug(f"Empty text received")
            return result

        # Detect language
        try:
            lang = detect(text)
            logger.debug(f"Detected language: {lang}")
        except LangDetectException as e:
            logger.warning(f"Language detection error: {e}")
            lang = "en"  # Default to English
        
        # Process text based on language
        if lang != "en":
            text_to_analyze = self._translate_text(text, lang)
        else:
            text_to_analyze = text
            
        # Split into sentences for better context preservation
        sentences = self._split_into_sentences(text_to_analyze)
        
        # Analyze toxicity with batch processing
        toxic_segments = self._analyze_toxicity(sentences)
        
        # Map toxic segments back to original text for censoring
        if lang != "en":
            censored_text = self._censor_non_english_text(text, toxic_segments, lang)
        else:
            censored_text = self._censor_english_text(text, toxic_segments)
        
        # Determine if text is toxic
        is_toxic = any(segment["toxic"] for segment in toxic_segments)
        
        result = {
            "label": "toxic" if is_toxic else "normal",
            "censored_text": censored_text,
            "processing_time": round(time.time() - start_time, 3),
            "details": toxic_segments if is_toxic else []
        }
        
        logger.info(f"Moderation completed in {result['processing_time']}s: {result['label']}")

        self.auto_save_counter += 1
        if self.auto_save_counter >= self.auto_save_interval:
            logger.info(f"Auto-saving cache after {self.auto_save_counter} operations")
            self.save_caches()
            self.auto_save_counter = 0
        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better context preservation"""
        # Simple sentence splitting - could be improved with language-specific models
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Handle very short or very long sentences
        processed_sentences = []
        current = ""
        
        for sentence in sentences:
            if len(sentence.split()) < 3:  # Very short, likely not enough context
                current += " " + sentence
            elif len(current.split()) > 0:
                processed_sentences.append(current.strip())
                current = sentence
            else:
                processed_sentences.append(sentence)
                
        if current:
            processed_sentences.append(current.strip())
            
        # Handle very long sentences by breaking them into chunks
        max_length = 128  # Maximum tokens for model
        final_sentences = []
        
        for sentence in processed_sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if len(tokens) > max_length:
                # Break into chunks preserving words
                words = sentence.split()
                chunks = []
                current_chunk = []
                
                for word in words:
                    current_chunk.append(word)
                    # Check if adding this word exceeds max length
                    if len(self.tokenizer.tokenize(" ".join(current_chunk))) > max_length:
                        current_chunk.pop()  # Remove the last word
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]  # Start new chunk with the removed word
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                final_sentences.extend(chunks)
            else:
                final_sentences.append(sentence)
        
        return final_sentences

    def _translate_text(self, text: str, source_lang: str) -> str:
        """Translate non-English text to English with caching"""
        # Check if we already have this text in cache
        cache_key = f"{source_lang}:{text}"
        cached = self.translation_cache.get(cache_key)
        if cached:
            logger.debug(f"Using cached translation for: {text[:30]}...")
            return cached
        
        try:
            logger.debug(f"Translating text from {source_lang} to English")
            translated = self.translator.translate(text, src=source_lang, dest="en").text
            self.translation_cache.put(cache_key, translated)
            return translated
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Fall back to original text
            return text

    def _analyze_toxicity(self, sentences: List[str]) -> List[Dict[str, Union[str, float, bool]]]:
        """Analyze toxicity of sentences with batch processing"""
        results = []
        
        # Process in batches for efficiency
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
        return results

    def _process_batch(self, sentences: List[str]) -> List[Dict[str, Union[str, float, bool]]]:
        """Process a batch of sentences for toxicity detection"""
        # Check cache first
        results = []
        uncached_indices = []
        uncached_sentences = []
        
        for i, sentence in enumerate(sentences):
            normalized = self._normalize_word(sentence)
            cached_score = self.prediction_cache.get(normalized)
            if cached_score is not None:
                results.append({
                    "text": sentence,
                    "score": cached_score,
                    "toxic": cached_score > self.toxic_threshold
                })
            else:
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_sentences.append(sentence)
        
        # Process uncached sentences
        if uncached_sentences:
            encoded = self.tokenizer(
                uncached_sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                toxicity_scores = scores[:, 1].tolist()  # Get toxicity scores
            
            # Update results and cache
            for i, (idx, sentence, score) in enumerate(zip(uncached_indices, uncached_sentences, toxicity_scores)):
                results[idx] = {
                    "text": sentence,
                    "score": score,
                    "toxic": score > self.toxic_threshold
                }
                self.prediction_cache.put(sentence, score)
        
        return results

    def _censor_english_text(self, text: str, toxic_segments: List[Dict[str, Union[str, float, bool]]]) -> str:
        """Censor toxic parts of English text"""
        if not any(segment["toxic"] for segment in toxic_segments):
            return text
            
        # Build a censoring dictionary
        censoring_dict = {}
        for segment in toxic_segments:
            if segment["toxic"]:
                toxic_text = segment["text"]
                # Extract words that might be toxic
                words = re.findall(r'\b\w+\b', toxic_text)
                for word in words:
                    if len(word) >= 3:  # Only consider words with at least 3 characters
                        word_score = self._get_word_toxicity(word)
                        if word_score > self.toxic_threshold:
                            censoring_dict[word] = self._censor_word(word)
        
        # Apply censoring
        censored = text
        for word, censored_word in censoring_dict.items():
            # Use regex to match whole words only
            censored = re.sub(r'\b' + re.escape(word) + r'\b', censored_word, censored, flags=re.IGNORECASE)
            
        return censored

    def _censor_non_english_text(self, original_text: str, toxic_segments: List[Dict[str, Union[str, float, bool]]], lang: str) -> str:       
        """Censor toxic parts of non-English text using translation mapping"""
        # ✅ Nếu model không detect toxic, vẫn phải check predefined list
        if not any(segment["toxic"] for segment in toxic_segments):
            words = re.findall(r'\b\w+\b', original_text)
            censoring_dict = {}
            for word in words:
                normalized = self._normalize_word(word)
                cached_score = self.prediction_cache.get(normalized)
                if cached_score is not None and cached_score > self.toxic_threshold:
                    censoring_dict[word] = self._censor_word(word)

            if censoring_dict:
                censored = original_text
                for word, censored_word in censoring_dict.items():
                    censored = re.sub(r'\b' + re.escape(word) + r'\b', censored_word, censored, flags=re.IGNORECASE)
                return censored

            return original_text

            
        words = re.findall(r'\b\w+\b', original_text)
        censoring_dict = {}
        
        for word in words:
            if len(word) >= 2:
                normalized = self._normalize_word(word)
                
                cached_score = self.prediction_cache.get(normalized)
                if cached_score is not None and cached_score > self.toxic_threshold:
                    censoring_dict[word] = self._censor_word(word)
                    continue

                try:
                    translated = self._translate_word(normalized, lang)
                    word_score = self._get_word_toxicity(translated)
                    if word_score > self.toxic_threshold:
                        censoring_dict[word] = self._censor_word(word)
                except Exception:
                    pass

        censored = original_text
        for word, censored_word in censoring_dict.items():
            censored = re.sub(r'\b' + re.escape(word) + r'\b', censored_word, censored, flags=re.IGNORECASE)
            
        return censored


    @lru_cache(maxsize=1000)
    def _translate_word(self, word: str, source_lang: str) -> str:
        """Translate a single word with caching"""
        cache_key = f"{source_lang}:{word}"
        cached = self.translation_cache.get(cache_key)
        if cached:
            return cached
            
        try:
            translated = self.translator.translate(word, src=source_lang, dest="en").text
            self.translation_cache.put(cache_key, translated)
            return translated
        except Exception as e:
            logger.debug(f"Translation error for word '{word}': {e}")
            return word

    @lru_cache(maxsize=1000)
    def _get_word_toxicity(self, word: str) -> float:
        """Get toxicity score for a single word with caching"""
        normalized = self._normalize_word(word)
        cached = self.prediction_cache.get(normalized)
        if cached:
            return cached
            
        try:
            encoded = self.tokenizer([normalized], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score = scores[0, 1].item()  # Get toxicity score
                
            self.prediction_cache.put(normalized, score)
            return score
        except Exception as e:
            logger.debug(f"Error getting toxicity for word '{word}': {e}")
            return 0.0

    def _normalize_word(self, word: str) -> str:
        """Normalize word by removing non-alphanumeric characters"""
        word = re.sub(r'\W+', '', word.lower())
        # giảm lặp ký tự (vd: ccc -> cc, dmmm -> dmm)
        word = re.sub(r'(.)\1{2,}', r'\1\1', word)
        return word

    def _censor_word(self, word: str) -> str:
        """Censor a word by replacing characters with asterisks"""
        if len(word) < 5:
            return '*' * len(word)
        else:
            return '*' * 5

    def save_caches(self):
        """Save caches to disk"""
        logger.info("=== SAVING CACHES ===")
        logger.info(f"Translation cache has {len(self.translation_cache.cache)} entries")
        logger.info(f"Prediction cache has {len(self.prediction_cache.cache)} entries")
        
        self.translation_cache.save()
        self.prediction_cache.save()
        
        logger.info("=== CACHE SAVING COMPLETED ===")
        
    def __del__(self):
        """Save caches on object destruction"""
        try:
            logger.info("TextModerator destructor called - saving caches...")
            self.save_caches()
        except Exception as e:
            logger.error(f"Error in destructor: {e}")