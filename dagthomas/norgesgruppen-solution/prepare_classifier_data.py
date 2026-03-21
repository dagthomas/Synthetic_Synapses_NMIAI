"""
Step 2: Prepare classifier training data.

Creates:
  classifier_data/crops/{category_id}/   - Shelf crops from annotations
  classifier_data/refs/{category_id}/    - Product reference images mapped to categories
  classifier_data/category_map.json      - product_code -> category_id mapping
"""

import json
from pathlib import Path
from collections import defaultdict

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("X:/norgesgruppen")
COCO_DIR = DATA_ROOT / "NM_NGD_coco_dataset" / "train"
PROD_DIR = DATA_ROOT / "NM_NGD_product_images"
OUT_DIR = ROOT / "classifier_data"

CROP_SIZE = 224
PADDING_RATIO = 0.10  # 10% padding around crops


def load_coco():
    with open(COCO_DIR / "annotations.json", encoding="utf-8") as f:
        return json.load(f)


def load_metadata():
    with open(PROD_DIR / "metadata.json", encoding="utf-8") as f:
        return json.load(f)


def build_name_to_catid(coco):
    """Build mapping from category name -> category_id."""
    return {cat["name"]: cat["id"] for cat in coco["categories"]}


def build_product_mapping(metadata, name_to_catid):
    """Map product_code -> category_id using name matching."""
    mapping = {}
    unmatched = []

    for product in metadata["products"]:
        name = product["product_name"]
        code = product["product_code"]

        if name in name_to_catid:
            mapping[code] = name_to_catid[name]
        else:
            # Try mojibake fix: encode cp1252, decode utf-8
            try:
                fixed = name.encode("cp1252").decode("utf-8")
                if fixed in name_to_catid:
                    mapping[code] = name_to_catid[fixed]
                    continue
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            unmatched.append((code, name))

    return mapping, unmatched


def extract_shelf_crops(coco):
    """Extract crops from shelf images for each annotation."""
    images = {img["id"]: img for img in coco["images"]}
    crops_dir = OUT_DIR / "crops"

    # Group annotations by image to avoid reopening images
    annos_by_image = defaultdict(list)
    for anno in coco["annotations"]:
        annos_by_image[anno["image_id"]].append(anno)

    total = 0
    for img_id, annos in annos_by_image.items():
        img_info = images[img_id]
        img_path = COCO_DIR / "images" / img_info["file_name"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Cannot open {img_path}: {e}")
            continue

        w, h = img.size

        for anno in annos:
            cat_id = anno["category_id"]
            cat_dir = crops_dir / str(cat_id)
            cat_dir.mkdir(parents=True, exist_ok=True)

            x, y, bw, bh = anno["bbox"]

            # Add padding
            pad_x = bw * PADDING_RATIO
            pad_y = bh * PADDING_RATIO
            x1 = max(0, int(x - pad_x))
            y1 = max(0, int(y - pad_y))
            x2 = min(w, int(x + bw + pad_x))
            y2 = min(h, int(y + bh + pad_y))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop = crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)

            crop_path = cat_dir / f"shelf_{img_id}_{anno['id']}.jpg"
            crop.save(crop_path, quality=90)
            total += 1

        if img_id % 50 == 0:
            print(f"  Processed image {img_id}, total crops: {total}")

    return total


def organize_reference_images(product_mapping):
    """Copy and organize product reference images by category_id."""
    refs_dir = OUT_DIR / "refs"
    total = 0

    for code, cat_id in product_mapping.items():
        prod_folder = PROD_DIR / code
        if not prod_folder.exists():
            continue

        cat_dir = refs_dir / str(cat_id)
        cat_dir.mkdir(parents=True, exist_ok=True)

        for img_file in prod_folder.iterdir():
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img = Image.open(img_file).convert("RGB")
                    img = img.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
                    dst = cat_dir / f"ref_{code}_{img_file.stem}.jpg"
                    img.save(dst, quality=90)
                    total += 1
                except Exception as e:
                    print(f"Warning: Cannot process {img_file}: {e}")

    return total


def main():
    print("Loading data...")
    coco = load_coco()
    metadata = load_metadata()

    name_to_catid = build_name_to_catid(coco)
    print(f"COCO categories: {len(name_to_catid)}")

    # Build product code -> category_id mapping
    product_mapping, unmatched = build_product_mapping(metadata, name_to_catid)
    print(f"Matched products: {len(product_mapping)}")
    if unmatched:
        print(f"Unmatched products: {len(unmatched)}")
        for code, name in unmatched:
            print(f"  {code}: {name}")

    # Save mapping
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "category_map.json", "w") as f:
        json.dump(product_mapping, f, indent=2)

    # Extract shelf crops
    print("\nExtracting shelf crops...")
    n_crops = extract_shelf_crops(coco)
    print(f"Total shelf crops: {n_crops}")

    # Organize reference images
    print("\nOrganizing reference images...")
    n_refs = organize_reference_images(product_mapping)
    print(f"Total reference images: {n_refs}")

    # Print stats
    crops_dir = OUT_DIR / "crops"
    refs_dir = OUT_DIR / "refs"
    n_crop_cats = len(list(crops_dir.iterdir())) if crops_dir.exists() else 0
    n_ref_cats = len(list(refs_dir.iterdir())) if refs_dir.exists() else 0
    print(f"\nCrop categories: {n_crop_cats}")
    print(f"Reference categories: {n_ref_cats}")


if __name__ == "__main__":
    main()
