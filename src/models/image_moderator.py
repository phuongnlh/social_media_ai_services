import requests
from transformers import AutoProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import uuid
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip
from statistics import mean, median
from typing import List, Dict, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageModerator:
    def __init__(self, cache_dir: Optional[str] = None, use_cuda: bool = True):
        """
        Initialize the image moderation models with improved loading and caching
        
        Args:
            cache_dir: Directory to cache models
            use_cuda: Whether to use CUDA if available
        """
        # Setup device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Configure model paths and caching
        self.cache_dir = cache_dir

        self.use_amp = False  # Will be set later if applicable
        self._load_models()
        
        # Configure thresholds
        self.violence_threshold = 0.7
        self.political_threshold = 0.7
        self.abuse_threshold = 0.6
        self.nsfw_threshold = 0.5
        
        # Setup temporary directory
        self.temp_dir = os.path.join("src", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def _load_models(self) -> None:
        """Load all required models with error handling"""
        try:
            start_time = time.time()
            logger.info("Loading NSFW detection model...")
            self.nsfw_model = AutoModelForImageClassification.from_pretrained(
                "Falconsai/nsfw_image_detection", 
                cache_dir=self.cache_dir
            ).to(self.device)
            self.nsfw_processor = AutoProcessor.from_pretrained(
                "Falconsai/nsfw_image_detection", 
                cache_dir=self.cache_dir
            )
            self.nsfw_labels = ["normal", "nsfw"]
            logger.info(f"NSFW model loaded in {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            logger.info("Loading Anime vs Real detection model...")
            self.anime_model = AutoModelForImageClassification.from_pretrained(
                "prithivMLmods/AI-vs-Deepfake-vs-Real", 
                cache_dir=self.cache_dir
            ).to(self.device)
            self.anime_processor = AutoProcessor.from_pretrained(
                "prithivMLmods/AI-vs-Deepfake-vs-Real", 
                cache_dir=self.cache_dir
            )
            logger.info(f"Anime vs Real model loaded in {time.time() - start_time:.2f} seconds")

            start_time = time.time()
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", 
                cache_dir=self.cache_dir
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", 
                cache_dir=self.cache_dir
            )
            logger.info(f"CLIP model loaded in {time.time() - start_time:.2f} seconds")
            
            # Define labels for different content categories
            self.violence_labels = ["normal", "violence", "fighting", "weapons", "blood", "injury"]
            self.political_labels = ["normal", "political", "propaganda", "political symbol", "protest"]
            self.abuse_labels = [
                "normal", 
                "nude child", 
                "child abuse", 
                "child in swimsuit", 
                "child in bikini", 
                "child without clothes", 
                "child shirtless", 
                "child naked"
            ]
            self.meme_labels = ["normal", "meme", "funny image", "cartoon", "joke", "pet with object", "animal meme"]
            # If using CUDA, optimize models for inference
            if self.use_cuda:
                self._optimize_for_inference()
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to CPU if CUDA loading fails
            if self.use_cuda:
                logger.info("Falling back to CPU")
                self.device = torch.device("cpu")
                self.use_cuda = False
                self._load_models()
            else:
                raise
    
    def _optimize_for_inference(self) -> None:
        """Optimize models for inference speed"""
        logger.info("Optimizing models for inference...")
        
        # Use torch.inference_mode for better performance
        self.nsfw_model.eval()
        self.anime_model.eval()
        self.clip_model.eval()
        
        # Use mixed precision where available
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
            logger.info("Enabling mixed precision inference")
            self.use_amp = True
        else:
            self.use_amp = False
            
    def _fetch_media(self, base_url: str, media_filename: str) -> str:
        """
        Fetch media from URL with improved error handling and progress tracking
        
        Args:
            base_url: The base URL to fetch the media from
            media_filename: The filename of the media
            
        Returns:
            Path to the downloaded temporary file
        """
        try:
            media_url = f"{base_url}"
            logger.info(f"Fetching media from: {media_url}")

            # Configure session with proper timeouts and headers
            session = requests.Session()
            session.headers.update({'User-Agent': 'DailyVibeAI/1.0'})
            
            # Stream the download with timeout
            response = session.get(
                media_url, 
                stream=True, 
                timeout=(5, 30),  # (connect timeout, read timeout)
                verify=True
            )
            response.raise_for_status()

            # Check for valid content
            content_length = int(response.headers.get("Content-Length", 0))
            content_type = response.headers.get("Content-Type", "")
            logger.info(f"Content-Length: {content_length}, Content-Type: {content_type}")
            
            if content_length == 0:
                raise ValueError("Content length is zero")
                
            # Create temp file
            suffix = os.path.splitext(media_filename)[1]
            temp_filename = f"temp_{uuid.uuid4().hex}{suffix}"
            tmp_path = os.path.join(self.temp_dir, temp_filename)
            
            # Download with progress tracking
            chunk_size = 8192
            total_size = 0
            start_time = time.time()
            
            with open(tmp_path, "wb") as tmp_file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp_file.write(chunk)
                        total_size += len(chunk)
                        
                        # Log progress for large files
                        if content_length > 1024*1024 and total_size % (1024*1024) < chunk_size:
                            elapsed = time.time() - start_time
                            percent = (total_size / content_length) * 100 if content_length else 0
                            logger.info(f"Downloaded: {total_size/1024/1024:.1f}MB ({percent:.1f}%) in {elapsed:.1f}s")
            
            downloaded_size = os.path.getsize(tmp_path)
            logger.info(f"Download complete. File size: {downloaded_size/1024/1024:.2f}MB")

            if downloaded_size == 0:
                raise ValueError("Downloaded file is empty")
                
            # Verify the file is not corrupted
            if content_length > 0 and abs(downloaded_size - content_length) > 1024:
                logger.warning(f"File size mismatch: expected {content_length}, got {downloaded_size}")

            return tmp_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching media: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching or saving media: {e}")
            raise

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Preprocess an image with error handling and resizing if needed
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Handle extremely large images by resizing
            max_dimension = 1920
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                logger.info(f"Resizing large image from {image.size} to {new_size}")
                image = image.resize(new_size, Image.LANCZOS)
                
            return image
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise

    def moderate_image(self, image_path: str) -> dict:
        """
        Moderate a single image with improved processing pipeline
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with moderation results
        """
        try:
            start_time = time.time()
            image = self._preprocess_image(image_path)

            # Check if image is anime/cartoon vs real photo
            anime_result = self._check_anime(image)
            is_anime = anime_result["is_anime"]
            
            # Get meme detection result
            meme_result = self._check_meme(image)
            is_meme = meme_result["is_meme"]

            # Content is non-photorealistic if it's anime or meme
            is_non_photorealistic = is_anime or is_meme

            # Adjust thresholds for non-photorealistic content
            nsfw_threshold = self.nsfw_threshold * 1.3 if is_non_photorealistic else self.nsfw_threshold
            violence_threshold = self.violence_threshold * 1.5 if is_non_photorealistic else self.violence_threshold
            political_threshold = self.political_threshold * 1.3 if is_non_photorealistic else self.political_threshold
            abuse_threshold = self.abuse_threshold * 1.5 if is_non_photorealistic else self.abuse_threshold

            # Check NSFW content first (most important filter)
            nsfw_result = self._check_nsfw(image)
            if nsfw_result["label"] == "nsfw" and nsfw_result["score"] > nsfw_threshold:
                if is_non_photorealistic:
                    logger.info(f"NSFW anime/meme content detected: {nsfw_result}, flagging for review")
                    return {
                        "label": "warning",
                        "score": round(nsfw_result["score"], 4),
                        "detail": "anime_nsfw" if is_anime else "meme_nsfw",
                        "anime_info": anime_result if is_anime else None,
                        "meme_info": meme_result if is_meme else None,
                        "needs_review": True
                    }
                else:
                    logger.info(f"NSFW content detected: {nsfw_result}")
                    return nsfw_result
            
            # Check child abuse content (high priority)
            abuse_result = self._check_abuse(image)
            if abuse_result["label"] == "abuse" and abuse_result["score"] > abuse_threshold:
                if is_non_photorealistic:
                    logger.info(f"Abuse in anime/meme content detected: {abuse_result}, flagging for review")
                    return {
                        "label": "warning",
                        "score": round(abuse_result["score"], 4),
                        "detail": "anime_abuse" if is_anime else "meme_abuse",
                        "anime_info": anime_result if is_anime else None,
                        "meme_info": meme_result if is_meme else None,
                        "needs_review": True
                    }
                else:
                    logger.info(f"Abuse content detected: {abuse_result}")
                    return abuse_result
                
            # Check violence content
            violence_result = self._check_violence(image)
            if violence_result["label"] == "violence" and violence_result["score"] > violence_threshold:
                if is_non_photorealistic:
                    logger.info(f"Violence in anime/meme content detected: {violence_result}, flagging for review")
                    return {
                        "label": "warning",
                        "score": round(violence_result["score"], 4),
                        "detail": "anime_violence" if is_anime else "meme_violence",
                        "anime_info": anime_result if is_anime else None,
                        "meme_info": meme_result if is_meme else None,
                        "needs_review": True
                    }
                else:
                    logger.info(f"Violence content detected: {violence_result}")
                    return violence_result
                
            # Check political content
            political_result = self._check_political(image)
            if political_result["label"] == "political" and political_result["score"] > political_threshold:
                if is_non_photorealistic:
                    logger.info(f"Political anime/meme content detected: {political_result}, flagging for review")
                    return {
                        "label": "warning",
                        "score": round(political_result["score"], 4),
                        "detail": "anime_political" if is_anime else "meme_political",
                        "anime_info": anime_result if is_anime else None,
                        "meme_info": meme_result if is_meme else None,
                        "needs_review": True
                    }
                else:
                    logger.info(f"Political content detected: {political_result}")
                    return political_result
            
            # If all checks pass, content is normal
            if is_non_photorealistic:
                result = {
                    "label": "normal", 
                    "score": round(0.8, 4), 
                    "is_anime": is_anime,
                    "is_meme": is_meme,
                    "anime_info": anime_result if is_anime else None,
                    "meme_info": meme_result if is_meme else None
                }
            else:
                result = {"label": "normal", "score": round(1.0 - nsfw_result["score"], 4)}
            
            logger.info(f"Image moderation completed in {time.time() - start_time:.2f}s: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error moderating image {image_path}: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}

    def _check_nsfw(self, image: Image.Image) -> dict:
        """Check if image contains NSFW content"""
        with torch.no_grad():
            inputs_nsfw = self.nsfw_processor(images=image, return_tensors="pt").to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_nsfw = self.nsfw_model(**inputs_nsfw)
            else:
                outputs_nsfw = self.nsfw_model(**inputs_nsfw)
                
            probs_nsfw = outputs_nsfw.logits.softmax(dim=1).tolist()[0]
            nsfw_score = probs_nsfw[1]  # Index 1 is "nsfw" class
            nsfw_label = "nsfw" if nsfw_score > self.nsfw_threshold else "normal"
            
            logger.info(f"NSFW detection: {nsfw_label} ({nsfw_score:.4f})")
            return {"label": nsfw_label, "score": round(nsfw_score, 4)}

    def _check_violence(self, image: Image.Image) -> dict:
        """Check if image contains violence"""
        with torch.no_grad():
            inputs_clip = self.clip_processor(text=self.violence_labels, images=image, return_tensors="pt", padding=True).to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_clip = self.clip_model(**inputs_clip)
            else:
                outputs_clip = self.clip_model(**inputs_clip)
                
            probs = outputs_clip.logits_per_image.softmax(dim=1).tolist()[0]
            # Get the highest non-normal score
            non_normal_scores = {label: score for label, score in zip(self.violence_labels[1:], probs[1:])}
            max_label = max(non_normal_scores, key=non_normal_scores.get) if non_normal_scores else "normal"
            max_score = non_normal_scores.get(max_label, 0)
            
            result_label = max_label if max_score > self.violence_threshold else "normal"
            is_violence = result_label != "normal"
            
            logger.info(f"Violence detection: {result_label} ({max_score:.4f})")
            return {
                "label": "violence" if is_violence else "normal",
                "score": round(max_score, 4),
                "detail": result_label if is_violence else ""
            }

    def _check_political(self, image: Image.Image) -> dict:
        """Check if image contains political content"""
        with torch.no_grad():
            inputs_clip = self.clip_processor(text=self.political_labels, images=image, return_tensors="pt", padding=True).to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_clip = self.clip_model(**inputs_clip)
            else:
                outputs_clip = self.clip_model(**inputs_clip)
                
            probs = outputs_clip.logits_per_image.softmax(dim=1).tolist()[0]
            # Get the highest non-normal score
            non_normal_scores = {label: score for label, score in zip(self.political_labels[1:], probs[1:])}
            max_label = max(non_normal_scores, key=non_normal_scores.get) if non_normal_scores else "normal"
            max_score = non_normal_scores.get(max_label, 0)
            
            result_label = max_label if max_score > self.political_threshold else "normal"
            is_political = result_label != "normal"
            
            logger.info(f"Political detection: {result_label} ({max_score:.4f})")
            return {
                "label": "political" if is_political else "normal",
                "score": round(max_score, 4),
                "detail": result_label if is_political else ""
            }

    def _check_abuse(self, image: Image.Image) -> dict:
        """Check if image contains child abuse or similar content"""
        with torch.no_grad():
            inputs_clip = self.clip_processor(text=self.abuse_labels, images=image, return_tensors="pt", padding=True).to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_clip = self.clip_model(**inputs_clip)
            else:
                outputs_clip = self.clip_model(**inputs_clip)
                
            probs = outputs_clip.logits_per_image.softmax(dim=1).tolist()[0]
            
            # Log detailed scores for all abuse labels
            for label, score in zip(self.abuse_labels, probs):
                if score > 0.1:  # Only log significant scores
                    logger.info(f"Abuse label '{label}': {score:.4f}")
            
            # Get the highest non-normal score
            non_normal_scores = {label: score for label, score in zip(self.abuse_labels[1:], probs[1:])}
            max_label = max(non_normal_scores, key=non_normal_scores.get) if non_normal_scores else "normal"
            max_score = non_normal_scores.get(max_label, 0)
            
            is_abuse = max_label != "normal" and max_score > self.abuse_threshold
            
            logger.info(f"Abuse detection: {max_label if is_abuse else 'normal'} ({max_score:.4f})")
            return {
                "label": "abuse" if is_abuse else "normal",
                "score": round(max_score, 4),
                "detail": max_label if is_abuse else ""
            }

    def moderate(self, base_url: str, media_filename: str) -> dict:
        """
        Moderate media from a URL
        
        Args:
            base_url: URL to fetch the media from
            media_filename: Filename of the media
            
        Returns:
            Dictionary with moderation results
        """
        start_time = time.time()
        temp_path = None
        try:
            temp_path = self._fetch_media(base_url, media_filename)
            
            # Determine if the file is a video or image
            is_video = media_filename.lower().endswith((".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"))
            
            result = self._moderate_video(temp_path) if is_video else self.moderate_image(temp_path)
            logger.info(f"Moderation completed in {time.time() - start_time:.2f}s: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Moderation error: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file {temp_path}: {e}")

    def _moderate_video(self, video_path: str, num_frames: int = 5) -> dict:
        """
        Moderate a video by sampling frames and analyzing them
        
        Args:
            video_path: Path to the video file
            num_frames: Number of frames to sample
            
        Returns:
            Dictionary with moderation results
        """
        try:
            # Open the video file
            logger.info(f"Processing video: {video_path}")
            clip = VideoFileClip(video_path)
            
            # Calculate frames to extract (beginning, middle, and end)
            duration = clip.duration
            logger.info(f"Video duration: {duration:.2f}s")
            
            if duration < 1.0:
                # For very short videos, just take the middle frame
                timestamps = [duration / 2]
            else:
                # Distribute frames across the video
                timestamps = [
                    duration * i / (num_frames + 1) 
                    for i in range(1, num_frames + 1)
                ]
            
            # Extract frames
            temp_files = []
            for i, t in enumerate(timestamps):
                frame = clip.get_frame(t)
                frame_path = os.path.join(self.temp_dir, f"frame_{uuid.uuid4().hex}.jpg")
                Image.fromarray(frame).save(frame_path)
                temp_files.append(frame_path)
                logger.info(f"Extracted frame {i+1}/{len(timestamps)} at {t:.2f}s")
            
            # Process frames (optionally in parallel)
            if len(temp_files) <= 2:
                # Sequential processing for small number of frames
                results = [self.moderate_image(path) for path in temp_files]
            else:
                # Parallel processing for multiple frames
                with ThreadPoolExecutor(max_workers=min(len(temp_files), 3)) as executor:
                    results = list(executor.map(self.moderate_image, temp_files))
            
            # Clean up temporary files
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)
            
            # Close video resources
            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()
            
            # Analyze results with weighted scoring system
            return self._analyze_video_results(results)
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise

    def _analyze_video_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze video frame results with improved scoring
        
        Args:
            results: List of frame analysis results
            
        Returns:
            Aggregated result
        """
        if not results:
            return {"label": "error", "score": 0.0, "error": "No frames analyzed"}
            
        # Count occurrences of each label
        labels = [r["label"] for r in results]
        label_counts = {label: labels.count(label) for label in set(labels)}
        
        # Calculate percentage of problematic frames
        total_frames = len(results)
        
        # Process by priority order
        for priority_label in ["nsfw", "abuse", "violence", "political"]:
            if priority_label in label_counts:
                # If more than 40% of frames have this label or at least 2 frames
                count = label_counts[priority_label]
                if count / total_frames >= 0.4 or count >= 2:
                    # Get all scores for this label
                    relevant_results = [r for r in results if r["label"] == priority_label]
                    scores = [r["score"] for r in relevant_results]
                    
                    # Get the highest scoring detail if available
                    details = [r.get("detail", "") for r in relevant_results if "detail" in r]
                    detail = max(details, key=details.count) if details else ""
                    
                    # Use 90th percentile score rather than mean to capture problematic frames
                    scores.sort()
                    percentile_idx = min(int(len(scores) * 0.9), len(scores) - 1)
                    final_score = scores[percentile_idx]
                    
                    return {
                        "label": priority_label,
                        "score": round(final_score, 4),
                        "detail": detail,
                        "frame_ratio": f"{count}/{total_frames}"
                    }
        
        # If no problematic frames exceed the threshold, the video is normal
        normal_scores = [r["score"] for r in results if r["label"] == "normal"]
        final_score = median(normal_scores) if normal_scores else 0.0
        
        return {"label": "normal", "score": round(final_score, 4)}
    def _check_meme(self, image: Image.Image) -> dict:
        """Check if image is a meme or humorous content"""
        with torch.no_grad():
            inputs_clip = self.clip_processor(text=self.meme_labels, images=image, return_tensors="pt", padding=True).to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs_clip = self.clip_model(**inputs_clip)
            else:
                outputs_clip = self.clip_model(**inputs_clip)
                
            probs = outputs_clip.logits_per_image.softmax(dim=1).tolist()[0]
            
            # Lấy nhãn và điểm cao nhất cho meme
            non_normal_scores = {label: score for label, score in zip(self.meme_labels[1:], probs[1:])}
            max_label = max(non_normal_scores, key=non_normal_scores.get) if non_normal_scores else "normal"
            max_score = non_normal_scores.get(max_label, 0)
            
            is_meme = max_score > 0.6  # Ngưỡng cho meme
            
            logger.info(f"Meme detection: {max_label if is_meme else 'normal'} ({max_score:.4f})")
            return {
                "is_meme": is_meme,
                "meme_score": round(max_score, 4),
                "meme_type": max_label if is_meme else ""
            }
    def _check_anime(self, image: Image.Image) -> dict:
        """Check if image is anime/cartoon or real photo"""
        with torch.no_grad():
            inputs = self.anime_processor(images=image, return_tensors="pt").to(self.device)
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.anime_model(**inputs)
            else:
                outputs = self.anime_model(**inputs)
                
            probs = outputs.logits.softmax(dim=1).tolist()[0]
            # The model outputs [anime_score, real_score]
            anime_score = probs[0]  # Index 0 for anime class
            is_anime = anime_score > 0.7  # Threshold for anime detection
            
            logger.info(f"Anime detection: {'anime' if is_anime else 'real'} ({anime_score:.4f})")
            return {
                "is_anime": is_anime,
                "anime_score": round(anime_score, 4)
            }