#!/usr/bin/env python
"""
Download YOLOv8 model and save to models/best.pt
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

print("=" * 70)
print("YOLOv8 Model Setup")
print("=" * 70)

print("\nLoading YOLOv8 medium model...")
model = YOLO('yolov8m.pt')
print("✓ Model loaded successfully")

# Find where ultralytics cached it
cache_dirs = [
    Path.home() / '.ultralytics' / 'weights',
    Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov8',
]

# Also check current directory and typical locations
potential_paths = cache_dirs + [
    Path('yolov8m.pt'),
    Path(r'C:\Users\Admin\.ultralytics\weights\yolov8m.pt'),
]

found = False
for path in potential_paths:
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024**2)
        print(f"\nFound model at: {path}")
        print(f"Size: {size_mb:.2f} MB")
        
        if size_mb > 40:  # YOLOv8m should be ~49 MB
            try:
                target = Path('models/best.pt')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(path), str(target))
                print(f"✓ Copied to {target}")
                found = True
                break
            except Exception as e:
                print(f"✗ Failed to copy: {e}")

if not found:
    print("\n⚠ Model not found in expected cache locations")
    print("Attempting to export model...")
    try:
        output = model.export(format='pt', imgsz=640)
        print(f"✓ Model exported to: {output}")
    except Exception as e:
        print(f"✗ Export failed: {e}")

# Final check
model_path = Path('models/best.pt')
if model_path.exists():
    size_mb = model_path.stat().st_size / (1024**2)
    print(f"\n" + "=" * 70)
    print(f"✓ SUCCESS! Model ready at: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print("=" * 70)
    print("\nNext: Run 'streamlit run dashboard.py'")
else:
    print(f"\n✗ Model file not found at {model_path}")
    print("\nAlternative: Download manually from:")
    print("  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt")
