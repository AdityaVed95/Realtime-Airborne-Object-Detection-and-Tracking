import os
from PIL import Image

# Validating that all the images in the dataset are 1024x1024 pixels dimensions

base_dir = "aod_dataset/images"
expected = (1024, 1024)
splits = ["train", "valid", "test"]

for split in splits:
    split_dir = os.path.join(base_dir, split)
    images = [f for f in os.listdir(split_dir) if f.endswith(".jpg")]
    mismatched = []

    for i, fname in enumerate(images):
        img = Image.open(os.path.join(split_dir, fname))
        if img.size != expected:
            mismatched.append((fname, img.size))
        if (i + 1) % 1000 == 0:
            print(f"  [{split}] Checked {i + 1}/{len(images)}...")

    print(f"\n{split}: {len(images)} images, {len(mismatched)} mismatched")
    for fname, size in mismatched:
        print(f"  {fname}: {size[0]}x{size[1]}")
