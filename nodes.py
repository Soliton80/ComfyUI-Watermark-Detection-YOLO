import os
import torch
import numpy as np
from PIL import Image
import cv2

# Import ComfyUI modules
import comfy.model_management as model_management

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print(
        "WARNING: ultralytics not available. "
        "Please install with: pip install ultralytics"
    )


class WatermarkDetectorLoader:
    """Load watermark detection model"""

    CATEGORY = "Watermark Detection"
    # Updated: 2025-07-22 - Only watermarks_detect_best model

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    ["watermarks_detect_best"],
                    {"default": "watermarks_detect_best"},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "iou_threshold": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("WATERMARK_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Watermark Detection"

    def load_model(self, model_type, confidence, iou_threshold):
        """Load and return the watermark detection model"""

        if not ULTRALYTICS_AVAILABLE:
            raise Exception(
                "ultralytics not available. "
                "Please install with: pip install ultralytics"
            )

        # Get device
        device = model_management.get_torch_device()

        # Set model path
        if model_type == "watermarks_detect_best":
            # Use trained watermark detection model
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(models_dir, "watermarks_detect_best.pt")

            if not os.path.exists(model_path):
                raise Exception(f"Watermark detection model not found at {model_path}")
            print(f"Using trained watermark detection model: {model_path}")
        else:
            raise Exception(f"Unknown model type: {model_type}")

        try:
            model = YOLO(model_path)
            model.to(device)
            print(f"Successfully loaded model: {model_path} on device: {device}")

            # Print model info for debugging
            if hasattr(model, "model") and hasattr(model.model, "names"):
                print(f"Model classes: {model.model.names}")
        except Exception as e:
            raise Exception(f"Failed to load model {model_path}: {e}")

        return (
            {
                "model": model,
                "confidence": confidence,
                "iou_threshold": iou_threshold,
                "model_path": model_path,
                "device": device,
            },
        )


class WatermarkDetector:
    """Detect watermarks in images using YOLO11"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "watermark_model": ("WATERMARK_MODEL",),
                "images": ("IMAGE",),
                "resolution": (["640", "1280", "1920"], {"default": "1280"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("detection_image", "watermark_mask", "detection_info")
    FUNCTION = "detect_watermarks"
    CATEGORY = "Watermark Detection"

    def detect_watermarks(self, watermark_model, images, resolution):
        """Perform watermark detection on input images"""

        if not ULTRALYTICS_AVAILABLE:
            raise Exception("ultralytics not available")

        model = watermark_model["model"]
        confidence = watermark_model["confidence"]
        iou_threshold = watermark_model["iou_threshold"]

        results_images = []
        results_masks = []
        detection_info = []

        for i, image in enumerate(images):
            # Convert tensor to numpy array
            img_np = (image.cpu().numpy() * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(img_np, "RGB")

            # Run detection
            detections = self._run_detection(
                model, pil_image, confidence, iou_threshold, int(resolution)
            )

            # Create detection visualization and mask
            detection_image = self._create_detection_image(pil_image, detections)
            watermark_mask = self._create_watermark_mask(pil_image.size, detections)
            info = self._format_detection_info(detections, i)

            # Convert back to tensors
            detection_array = np.array(detection_image).astype(np.float32)
            detection_tensor = torch.from_numpy(detection_array / 255.0).unsqueeze(0)

            mask_tensor = torch.from_numpy(watermark_mask).unsqueeze(0)

            results_images.append(detection_tensor)
            results_masks.append(mask_tensor)
            detection_info.append(info)

        # Stack results
        output_images = torch.cat(results_images, dim=0)
        output_masks = torch.cat(results_masks, dim=0)
        combined_info = "\n".join(detection_info)

        return (output_images, output_masks, combined_info)

    def _run_detection(self, model, image, confidence, iou_threshold, resolution):
        """Perform detection at specified resolution"""
        img_array = np.array(image)

        results = model(
            img_array,
            conf=confidence,
            iou=iou_threshold,
            imgsz=resolution,
            verbose=False,
        )

        return results[0] if results else None

    def _create_detection_image(self, original_image, detections):
        """Create image with detection boxes"""
        if detections is None or detections.boxes is None:
            return original_image

        img = np.array(original_image)

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label with confidence
            label = f"Watermark {conf:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Draw label background
            cv2.rectangle(
                img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return Image.fromarray(img)

    def _create_watermark_mask(self, image_size, detections):
        """Create binary mask for detected watermarks"""
        width, height = image_size
        mask = np.zeros((height, width), dtype=np.float32)

        if detections is None or detections.boxes is None:
            return mask

        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Clamp coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            mask[y1:y2, x1:x2] = 1.0

        return mask

    def _format_detection_info(self, detections, image_index):
        """Format detection information"""
        if detections is None or detections.boxes is None:
            return f"Image {image_index + 1}: No watermarks detected"

        num_detections = len(detections.boxes)
        confidences = [box.conf[0].cpu().numpy() for box in detections.boxes]
        avg_confidence = np.mean(confidences) if confidences else 0

        info = f"Image {image_index + 1}: {num_detections} watermarks detected\n"
        info += f"Average confidence: {avg_confidence:.3f}\n"

        for i, (box, conf) in enumerate(zip(detections.boxes, confidences)):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            info += f"  Watermark {i+1}: {conf:.3f} at [{x1}, {y1}, {x2}, {y2}]\n"

        return info


NODE_CLASS_MAPPINGS = {
    "WatermarkDetectorLoader": WatermarkDetectorLoader,
    "WatermarkDetector": WatermarkDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkDetectorLoader": "Load Watermark Detector (Custom)",
    "WatermarkDetector": "Detect Watermarks (Custom)",
}
