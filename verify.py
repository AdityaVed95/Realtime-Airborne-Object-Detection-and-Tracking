from pathlib import Path

# Pillow is used to verify that images actually open correctly.
# Install with: python3 -m pip install pillow
from PIL import Image

ROOT = Path("aod_dataset")
IMAGE_ROOT = ROOT / "images"
LABEL_ROOT = ROOT / "labels"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {
    0: "aeroplane",
    1: "bird",
    2: "drone",
    3: "helicopter",
}


def get_files_by_stem(folder: Path, allowed_suffixes=None):
    """
    Returns a dictionary:
        {file_stem: full_path}
    """
    files = {}

    if not folder.exists():
        return files

    for path in folder.iterdir():
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        if allowed_suffixes is None or suffix in allowed_suffixes:
            files[path.stem] = path

    return files


def verify_image(image_path: Path):
    """
    Returns None if image is valid, else returns error string.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return None
    except Exception as e:
        return str(e)


def validate_yolo_label_file(label_path: Path):
    """
    Validate a YOLO-format label file.

    Expected per non-empty line:
        class_id x_center y_center width height

    Returns a list of issues found in this file.
    Empty files are allowed.
    """
    issues = []

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        issues.append(f"Could not read file: {e}")
        return issues

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Empty label file or empty line is allowed
        if not stripped:
            continue

        parts = stripped.split()

        if len(parts) != 5:
            issues.append(
                f"Line {line_number}: expected 5 values, got {len(parts)} -> {stripped}"
            )
            continue

        # Parse values
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            issues.append(
                f"Line {line_number}: contains non-numeric values -> {stripped}"
            )
            continue

        # Validate class_id
        if class_id < 0:
            issues.append(f"Line {line_number}: class_id must be >= 0 -> {class_id}")

        # Validate normalized coordinates
        if not (0.0 <= x_center <= 1.0):
            issues.append(
                f"Line {line_number}: x_center out of range [0,1] -> {x_center}"
            )

        if not (0.0 <= y_center <= 1.0):
            issues.append(
                f"Line {line_number}: y_center out of range [0,1] -> {y_center}"
            )

        if not (0.0 < width <= 1.0):
            issues.append(
                f"Line {line_number}: width must be in (0,1] -> {width}"
            )

        if not (0.0 < height <= 1.0):
            issues.append(
                f"Line {line_number}: height must be in (0,1] -> {height}"
            )

    return issues

def get_class_distribution(label_folder: Path):
    """
    Count annotations per class and empty label files across all label files.
    Returns a tuple: ({class_id: count}, empty_file_count)
    """
    counts = {}
    empty_count = 0
    for label_path in label_folder.iterdir():
        if label_path.suffix != ".txt":
            continue
        has_annotations = False
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                has_annotations = True
                parts = stripped.split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        counts[class_id] = counts.get(class_id, 0) + 1
                    except ValueError:
                        pass
        if not has_annotations:
            empty_count += 1
    return counts, empty_count


def validate_split(split_name: str):
    image_folder = IMAGE_ROOT / split_name
    label_folder = LABEL_ROOT / split_name

    print(f"\n{'=' * 80}")
    print(f"Checking split: {split_name}")
    print(f"Images folder: {image_folder}")
    print(f"Labels folder: {label_folder}")

    if not image_folder.exists():
        print(f"ERROR: Missing image folder: {image_folder}")
        return False

    if not label_folder.exists():
        print(f"ERROR: Missing label folder: {label_folder}")
        return False

    image_files = get_files_by_stem(image_folder, IMAGE_EXTENSIONS)
    label_files = get_files_by_stem(label_folder, {".txt"})

    image_stems = set(image_files.keys())
    label_stems = set(label_files.keys())

    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)
    matched_stems = sorted(image_stems & label_stems)

    print(f"Number of images: {len(image_files)}")
    print(f"Number of labels: {len(label_files)}")
    print(f"Matched pairs: {len(matched_stems)}")
    print(f"Images without labels: {len(images_without_labels)}")
    print(f"Labels without images: {len(labels_without_images)}")

    if images_without_labels:
        print("\nSample images without labels:")
        for stem in images_without_labels[:10]:
            print(f"  {image_files[stem].name}")

    if labels_without_images:
        print("\nSample labels without images:")
        for stem in labels_without_images[:10]:
            print(f"  {label_files[stem].name}")

    # Verify images
    bad_images = []
    for stem in matched_stems:
        error = verify_image(image_files[stem])
        if error is not None:
            bad_images.append((image_files[stem], error))

    print(f"\nUnreadable/corrupt images: {len(bad_images)}")
    if bad_images:
        print("Sample bad images:")
        for path, error in bad_images[:10]:
            print(f"  {path.name} -> {error}")

    # Validate labels
    bad_labels = []
    for stem in matched_stems:
        issues = validate_yolo_label_file(label_files[stem])
        if issues:
            bad_labels.append((label_files[stem], issues))

    print(f"Label files with format issues: {len(bad_labels)}")
    if bad_labels:
        print("Sample bad label files:")
        for path, issues in bad_labels[:5]:
            print(f"\n  {path.name}")
            for issue in issues[:5]:
                print(f"    - {issue}")

    # Class distribution
    class_counts, empty_count = get_class_distribution(label_folder)
    total_annotations = sum(class_counts.values())
    print(f"\nClass distribution ({total_annotations} total annotations):")
    for class_id in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(class_id, f"unknown-{class_id}")
        count = class_counts[class_id]
        pct = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"  {class_id}: {name:40s} -> {count:6d}  ({pct:.1f}%)")
    print(f"\nEmpty label files (no object in sky): {empty_count}")


    # Final status for this split
    split_ok = (
        len(images_without_labels) == 0
        and len(labels_without_images) == 0
        and len(bad_images) == 0
        and len(bad_labels) == 0
    )

    if split_ok:
        print(f"\nResult for '{split_name}': PASS")
    else:
        print(f"\nResult for '{split_name}': FAIL")

    return split_ok


def main():
    print("AOD YOLOv8 dataset validation starting...\n")
    print(f"Working directory: {ROOT.resolve()}")

    if not IMAGE_ROOT.exists():
        print(f"ERROR: Missing folder: {IMAGE_ROOT}")
        return

    if not LABEL_ROOT.exists():
        print(f"ERROR: Missing folder: {LABEL_ROOT}")
        return

    all_ok = True
    for split in SPLITS:
        split_ok = validate_split(split)
        all_ok = all_ok and split_ok

    print(f"\n{'=' * 80}")
    if all_ok:
        print("FINAL RESULT: PASS")
        print("Dataset structure, image readability, and YOLO label format all look good.")
    else:
        print("FINAL RESULT: FAIL")
        print("Some issues were found. Review the messages above carefully.")


if __name__ == "__main__":
    main()