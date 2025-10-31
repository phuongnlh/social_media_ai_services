import random
import requests
from transformers import AutoProcessor, AutoModelForImageClassification, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import uuid
import threading, glob
import math
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
        self.abuse_threshold = 0.7
        self.nsfw_threshold = 0.5
        
        # Setup temporary directory
        self.temp_dir = os.path.join("src", "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        threading.Thread(target=self._cleanup_temp, daemon=True).start()

    def _cleanup_temp(self):
        """
        Periodic cleanup for stale temp files (older than 10 minutes and no lock file)
        """
        while True:
            try:
                now = time.time()
                for f in glob.glob(os.path.join(self.temp_dir, "temp_*")):
                    if f.endswith(".lock"):
                        continue  # bỏ qua file lock
                    lock_path = f + ".lock"

                    # ✅ Bỏ qua file đang được xử lý hoặc mới tải về
                    if os.path.exists(lock_path):
                        continue

                    # ✅ Xóa file cũ hơn 10 phút
                    if now - os.path.getmtime(f) > 600:
                        os.remove(f)
                        logger.info(f"[CLEANUP] Removed stale temp file: {f}")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

            time.sleep(600)  # chạy mỗi 10 phút
    
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
            self.nsfw_model_2 = AutoModelForImageClassification.from_pretrained("giacomoarienti/nsfw-classifier",cache_dir=self.cache_dir).to(self.device)
            self.nsfw_processor_2 = AutoProcessor.from_pretrained("giacomoarienti/nsfw-classifier",cache_dir=self.cache_dir)
            self.nsfw_labels = ["normal", "nsfw", "porn", "sexy", "hentai"]
            logger.info(f"NSFW model loaded in {time.time() - start_time:.2f} seconds")

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
            logger.info("All models loaded successfully")
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

    def _fetch_media(self, base_url: str, media_type: str) -> tuple[str, str]:
        """
        Fetch media from URL and save as temp file with correct extension.
        """
        try:
            media_url = f"{base_url}"
            logger.info(f"Fetching media from: {media_url}")

            session = requests.Session()
            session.headers.update({'User-Agent': 'DailyVibeAI/1.0'})
            response = session.get(media_url, stream=True, timeout=(5, 30))
            response.raise_for_status()

            content_length = int(response.headers.get("Content-Length", 0))
            content_type = response.headers.get("Content-Type", "")
            logger.info(f"Content-Length: {content_length}, Content-Type: {content_type}")

            # ✅ Xác định phần mở rộng dựa theo media_type hoặc Content-Type
            ext_map = {"image": ".jpg", "video": ".mp4"}
            if not media_type or media_type == "auto":
                if "image" in content_type:
                    media_type = "image"
                elif "video" in content_type:
                    media_type = "video"
                else:
                    media_type = "unknown"

            ext = ext_map.get(media_type, "")
            tmp_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4().hex}{ext}")
            lock_path = tmp_path + ".lock"

            # ✅ Tạo file lock ngay trước khi bắt đầu ghi file
            open(lock_path, "w").close()

            with open(tmp_path, "wb") as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)

            logger.info(f"Download complete: {tmp_path}")
            return (tmp_path, media_type)

        except Exception as e:
            logger.error(f"Error fetching media: {e}")
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

            # Check NSFW content first (most important filter)
            nsfw_result = self._check_nsfw(image)
            if nsfw_result["label"] == "nsfw" and nsfw_result["score"] > self.nsfw_threshold:
                logger.info(f"NSFW content detected: {nsfw_result}")
                return nsfw_result
            
            # Check child abuse content (high priority)
            abuse_result = self._check_abuse(image)
            if abuse_result["label"] == "abuse" and abuse_result["score"] > self.abuse_threshold:
                logger.info(f"Abuse content detected: {abuse_result}")
                return abuse_result
                
            # Check violence content
            violence_result = self._check_violence(image)
            if violence_result["label"] == "violence" and violence_result["score"] > self.violence_threshold:
                logger.info(f"Violence content detected: {violence_result}")
                return violence_result
            
            # If all checks pass, content is normal
            result = {"label": "normal", "score": round(1.0 - nsfw_result["score"], 4)}
            
            logger.info(f"Image moderation completed in {time.time() - start_time:.2f}s: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error moderating image {image_path}: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}

    def _check_nsfw(self, image: Image.Image) -> dict:
        """Check if image contains NSFW content using 2 models"""
        with torch.no_grad():
            # Model 1
            inputs1 = self.nsfw_processor(images=image, return_tensors="pt").to(self.device)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs1 = self.nsfw_model(**inputs1)
            else:
                outputs1 = self.nsfw_model(**inputs1)
            probs1 = outputs1.logits.softmax(dim=1).tolist()[0]
            nsfw_score_1 = probs1[1]  # index 1 = nsfw

            # Model 2
            inputs2 = self.nsfw_processor_2(images=image, return_tensors="pt").to(self.device)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs2 = self.nsfw_model_2(**inputs2)
            else:
                outputs2 = self.nsfw_model_2(**inputs2)
            probs2 = outputs2.logits.softmax(dim=1).tolist()[0]
            probs2_filtered = probs2[:2] + probs2[3:]

            nsfw_score_2 = max(probs2_filtered)
            nsfw_label_2 = "nsfw" if nsfw_score_2 > self.nsfw_threshold else "normal"

            # Take the max score between 2 models
            nsfw_score = max(nsfw_score_1, nsfw_score_2)
            nsfw_label = "nsfw" if nsfw_score > self.nsfw_threshold else "normal"
            detail = "model1" if nsfw_score_1 >= nsfw_score_2 else "model2"

            logger.info(f"NSFW detection: {nsfw_label} ({nsfw_score:.4f}), detail: {detail}, model1_score={nsfw_score_1:.4f}, model2_score={nsfw_score_2:.4f}")
            return {"label": nsfw_label, "score": round(nsfw_score, 4), "detail": detail}

    def _check_violence(self, image: Image.Image) -> dict:
        """Check if image contains violence"""
        with torch.no_grad():
            # Chuẩn bị dữ liệu cho CLIP
            inputs_clip = self.clip_processor(
                text=self.violence_labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Inference (hỗ trợ mixed precision nếu có)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs_clip = self.clip_model(**inputs_clip)

            probs = outputs_clip.logits_per_image.softmax(dim=1).tolist()[0]

            # Kiểm tra NaN để tránh lỗi
            if any(math.isnan(p) for p in probs):
                logger.warning("NaN detected in violence scores, returning normal")
                return {"label": "normal", "score": 0.0, "detail": ""}

            # Bỏ nhãn "normal" ra khỏi so sánh
            normal_index = self.violence_labels.index("normal") if "normal" in self.violence_labels else None
            non_normal_scores = {
                label: score
                for i, (label, score) in enumerate(zip(self.violence_labels, probs))
                if i != normal_index
            }

            # Lấy nhãn có xác suất cao nhất
            max_label = max(non_normal_scores, key=non_normal_scores.get)
            max_score = non_normal_scores[max_label]

            # Xác định kết quả
            is_violence = max_score > self.violence_threshold
            result_label = max_label if is_violence else "normal"

            # Log gọn, rõ
            if is_violence:
                logger.warning(f"[VIOLENCE DETECTED] {result_label} ({max_score:.4f})")
            else:
                logger.info(f"[SAFE] Violence score max={max_score:.4f}")

            return {
                "label": "violence" if is_violence else "normal",
                "score": round(max_score, 4),
                "detail": result_label if is_violence else ""
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

    def moderate(self, base_url: str, media_type: str) -> dict:
        """
        Moderate media by its type ("image" or "video")
        """
        start_time = time.time()
        temp_path = None
        try:
            if media_type not in ("image", "video"):
                logger.warning(f"Unsupported media type: {media_type}")
                return {"label": "skipped", "score": 0.0, "detail": "unsupported_type"}

            temp_path, media_type = self._fetch_media(base_url, media_type)

            # ✅ Model xử lý theo loại
            if media_type == "video":
                result = self._moderate_video(temp_path)
            else:
                result = self.moderate_image(temp_path)

            elapsed = time.time() - start_time
            logger.info(f"✅ Final moderation result for {media_type}: {result['label']} ({result.get('score', 0):.2f}) in {elapsed:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Moderation error: {e}")
            return {"label": "error", "score": 0.0, "error": str(e)}

        finally:
            # ✅ Cleanup an toàn sau khi xử lý xong
            if temp_path:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    lock_path = temp_path + ".lock"
                    if os.path.exists(lock_path):
                        os.remove(lock_path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_path}: {e}")

    def _moderate_video(self, video_path: str, num_frames: int = 5) -> dict:
        """Moderate a video by sampling frames and analyzing them with early stop"""
        try:
            logger.info(f"Processing video: {video_path}")
            clip = VideoFileClip(video_path)
            duration = clip.duration
            logger.info(f"Video duration: {duration:.2f}s")

            # Determine number of frames based on video length
            if duration < 5:
                num_frames = 3
            elif duration < 30:
                num_frames = 6
            elif duration < 120:
                num_frames = 10
            elif duration < 300:
                num_frames = 14
            elif duration < 600:
                num_frames = 18
            else:
                num_frames = 20

            # Calculate timestamps for frames
            if duration > 600:
                timestamps = sorted(random.uniform(0, duration) for _ in range(num_frames))
            else:
                timestamps = [duration * i / (num_frames + 1) for i in range(1, num_frames + 1)]

            # Extract frames
            temp_files = []
            for i, t in enumerate(timestamps):
                frame = clip.get_frame(t)
                frame_path = os.path.join(self.temp_dir, f"frame_{uuid.uuid4().hex}.jpg")
                Image.fromarray(frame).save(frame_path)
                temp_files.append(frame_path)
                logger.info(f"Extracted frame {i+1}/{len(timestamps)} at {t:.2f}s")

            # Moderate frames
            results = []
            early_stop = False
            for i, path in enumerate(temp_files):
                result = self.moderate_image(path)
                results.append(result)

                if result["label"] in ("nsfw", "violence", "abuse") and result.get("score", 0) > 0.8:
                    logger.info(f"⚠️ Early stop: Detected {result['label']} with score {result['score']:.2f} at frame {i+1}")
                    early_stop = True
                    break

            # Process remaining frames in parallel if not early stop
            if not early_stop and len(results) < len(temp_files):
                remaining = temp_files[len(results):]
                with ThreadPoolExecutor(max_workers=min(len(remaining), 3)) as executor:
                    results.extend(executor.map(self.moderate_image, remaining))

            # Clean up temp files
            for path in temp_files:
                if os.path.exists(path):
                    os.remove(path)

            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()

            # Aggregate frame results
            return self._analyze_video_results(results)

        except Exception as e:
            logger.error(f"Error processing video: {e}")
            raise

    def _analyze_video_results(self, results: list) -> dict:
        """
        Analyze video frame results with improved scoring for NSFW and other labels
        """
        if not results:
            return {"label": "error", "score": 0.0, "error": "No frames analyzed"}

        total_frames = len(results)

        # Nhóm frame theo label
        labels = [r["label"] for r in results]
        label_counts = {label: labels.count(label) for label in set(labels)}

        # Kiểm tra các nhãn ưu tiên
        for priority_label in ["nsfw", "abuse", "violence"]:
            if priority_label in label_counts:
                count = label_counts[priority_label]
                frames = [r for r in results if r["label"] == priority_label]
                scores = [r["score"] for r in frames]
                details = [r.get("detail", "") for r in frames if "detail" in r]
                detail = max(details, key=details.count) if details else ""

                # Điều kiện gán nhãn: >=40% frame hoặc >=2 frame hoặc score frame cao cực (>0.90)
                if count / total_frames >= 0.4 or count >= 2 or max(scores) > 0.90:
                    final_score = max(scores)
                    return {
                        "label": priority_label,
                        "score": round(final_score, 4),
                        "detail": detail,
                        "frame_ratio": f"{count}/{total_frames}"
                    }

        # Nếu không có nhãn ưu tiên nào
        normal_scores = [r["score"] for r in results if r["label"] == "normal"]
        final_score = median(normal_scores) if normal_scores else 0.0
        return {"label": "normal", "score": round(final_score, 4)}