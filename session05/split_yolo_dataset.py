import argparse
import json
import random
import shutil
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def split_dataset(
    annotations_dir: Path,
    output_dir: Path,
    classes_file: Path,
    seed: int,
) -> None:
    if not annotations_dir.is_dir():
        raise ValueError(f"Annotations directory does not exist: {annotations_dir}")
    if not classes_file.is_file():
        raise ValueError(f"Classes file does not exist: {classes_file}")

    classes = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"No class names found in {classes_file}")

    images = sorted(
        path
        for path in annotations_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(
            f"No .jpg, .jpeg, or .png images found in {annotations_dir}"
        )

    stems = [image.stem for image in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Image files must have unique names without extensions")

    missing_labels = [
        image.name
        for image in images
        if not (annotations_dir / f"{image.stem}.txt").is_file()
    ]
    if missing_labels:
        raise ValueError(
            "Missing label files for: " + ", ".join(missing_labels)
        )

    pairs = list(images)
    random.Random(seed).shuffle(pairs)
    validation_count = max(1, round(len(pairs) * 0.2)) if len(pairs) > 1 else 0
    validation = pairs[:validation_count]
    training = pairs[validation_count:]

    for split_name, split_images in (("train", training), ("val", validation)):
        split_images_dir = output_dir / split_name / "images"
        split_labels_dir = output_dir / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, split_images_dir / image.name)
            shutil.copy2(
                annotations_dir / f"{image.stem}.txt",
                split_labels_dir / f"{image.stem}.txt",
            )

    yaml_contents = (
        f"path: {json.dumps(str(output_dir.resolve()))}\n"
        "train: train/images\n"
        "val: val/images\n\n"
        f"nc: {len(classes)}\n"
        f"names: {json.dumps(classes, ensure_ascii=False)}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_contents, encoding="utf-8")
    print(
        f"Split {len(pairs)} image/label pairs: "
        f"{len(training)} train, {len(validation)} val."
    )
    print(f"Dataset YAML: {output_dir / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split YOLO image/label pairs into train and validation sets."
    )
    parser.add_argument("--annotations", type=Path, default=Path("annotations"))
    parser.add_argument("--classes", type=Path, default=Path("images/classes.txt"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        split_dataset(args.annotations, args.output, args.classes, args.seed)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
