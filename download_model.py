"""
Download and setup YOLOv8 model weights.

This script downloads a pre-trained YOLOv8 model and saves it to models/best.pt.
The model will be cached by ultralytics for future use.
"""

import os
from pathlib import Path
from ultralytics import YOLO

def download_yolo_model():
    """Download YOLOv8 model weights."""
    
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "best.pt"
    
    print("=" * 60)
    print("Downloading YOLOv8 model...")
    print("=" * 60)
    
    try:
        # Download the model - ultralytics will handle caching
        # Using yolov8m (medium) as a good balance of speed and accuracy
        model = YOLO("yolov8m.pt")
        
        # Save to the expected location
        model.export(format="pt", imgsz=640)
        
        # Also explicitly save a copy to best.pt
        model_file = models_dir / "yolov8m.pt"
        if model_file.exists():
            import shutil
            shutil.copy(str(model_file), str(model_path))
            print(f"✓ Model saved to: {model_path}")
            print(f"  File size: {model_path.stat().st_size / (1024**2):.2f} MB")
        else:
            # Alternative: save directly from ultralytics cache
            ultralytics_cache = Path.home() / ".ultralytics" / "weights"
            src = ultralytics_cache / "yolov8m.pt"
            if src.exists():
                import shutil
                shutil.copy(str(src), str(model_path))
                print(f"✓ Model saved to: {model_path}")
                print(f"  File size: {model_path.stat().st_size / (1024**2):.2f} MB")
        
        print("=" * 60)
        print("✓ YOLOv8 model ready!")
        print("✓ Run: streamlit run dashboard.py")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nAlternative: Download manually from:")
        print("  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt")
        print(f"  and save to: {model_path}")
        return False

if __name__ == "__main__":
    download_yolo_model()
