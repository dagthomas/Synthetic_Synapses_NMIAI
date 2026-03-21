#!/bin/bash
# Download and prepare the FULL NorgesGruppen COCO dataset on the VM.
# Downloads fresh from GCS, converts COCO→YOLO format, merges all into one dir.
set -e

cd ~

COCO_URL="$1"
if [ -z "$COCO_URL" ]; then
    echo "Usage: bash cloud_setup_data.sh <COCO_ZIP_URL>"
    exit 1
fi

echo "=== Downloading COCO dataset ==="
curl -L -o coco_dataset.zip "$COCO_URL"
ls -lh coco_dataset.zip

echo "=== Extracting ==="
rm -rf ~/coco_raw
mkdir -p ~/coco_raw
cd ~/coco_raw
unzip -q ~/coco_dataset.zip
ls -la

# Find the actual directory
COCO_DIR=$(find . -name "annotations" -type d | head -1 | xargs dirname)
echo "COCO dir: $COCO_DIR"
ls "$COCO_DIR"

echo "=== Converting COCO to YOLO format ==="
cd ~

python3 -c "
import json
from pathlib import Path

# Find annotations file
coco_root = Path('coco_raw')
ann_dirs = list(coco_root.rglob('annotations'))
if not ann_dirs:
    print('ERROR: No annotations dir found')
    exit(1)

ann_dir = ann_dirs[0]
coco_dir = ann_dir.parent
print(f'COCO dir: {coco_dir}')

# Find train annotations
ann_file = None
for f in ann_dir.glob('*.json'):
    print(f'  Found: {f}')
    ann_file = f

if not ann_file:
    print('ERROR: No annotation JSON found')
    exit(1)

print(f'Using: {ann_file}')

with open(ann_file) as f:
    coco = json.load(f)

print(f'Images: {len(coco[\"images\"])}')
print(f'Annotations: {len(coco[\"annotations\"])}')
print(f'Categories: {len(coco[\"categories\"])}')

# Find image directory
img_dir = None
for d in ['train', 'images', 'train2017']:
    candidate = coco_dir / d
    if candidate.exists():
        img_dir = candidate
        break
if not img_dir:
    # Try to find jpg files
    for f in coco_dir.rglob('*.jpg'):
        img_dir = f.parent
        break

if not img_dir:
    print('ERROR: No image directory found')
    exit(1)

print(f'Image dir: {img_dir} ({len(list(img_dir.glob(\"*.jpg\")))} jpgs)')

# Build category mapping: coco_id -> sequential index
cat_map = {}
for i, cat in enumerate(sorted(coco['categories'], key=lambda c: c['id'])):
    cat_map[cat['id']] = i

print(f'Category mapping: {len(cat_map)} classes (0 to {max(cat_map.values())})')

# Build image lookup
img_lookup = {img['id']: img for img in coco['images']}

# Group annotations by image
from collections import defaultdict
img_anns = defaultdict(list)
for ann in coco['annotations']:
    img_anns[ann['image_id']].append(ann)

# Output directories
out_dir = Path('datasets/merged')
out_imgs = out_dir / 'images'
out_lbls = out_dir / 'labels'
out_imgs.mkdir(parents=True, exist_ok=True)
out_lbls.mkdir(parents=True, exist_ok=True)

import shutil

copied = 0
for img_info in coco['images']:
    img_id = img_info['id']
    fname = img_info['file_name']
    w, h = img_info['width'], img_info['height']

    # Find source image
    src = img_dir / fname
    if not src.exists():
        # Try subdirectories
        matches = list(img_dir.rglob(fname))
        if matches:
            src = matches[0]
        else:
            continue

    # Copy image
    dst_img = out_imgs / fname
    if not dst_img.exists():
        shutil.copy2(src, dst_img)

    # Write YOLO label
    anns = img_anns.get(img_id, [])
    label_path = out_lbls / Path(fname).with_suffix('.txt').name
    with open(label_path, 'w') as lf:
        for ann in anns:
            cat_id = cat_map.get(ann['category_id'])
            if cat_id is None:
                continue
            bx, by, bw, bh = ann['bbox']  # COCO format: x,y,w,h
            # Convert to YOLO format: cx, cy, nw, nh (normalized)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lf.write(f'{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')

    copied += 1

print(f'Prepared {copied} images with labels in {out_dir}')

# Create data_all.yaml
import yaml
names = {i: cat['name'] for cat in coco['categories'] for i, c_id in cat_map.items() if c_id == cat_map[cat['id']]}
# Simpler: just map index -> name
idx_to_name = {}
for cat in coco['categories']:
    idx_to_name[cat_map[cat['id']]] = cat['name']

# Add unknown_product as last class if not present
nc = max(idx_to_name.keys()) + 1
if 'unknown_product' not in idx_to_name.values():
    idx_to_name[nc] = 'unknown_product'
    nc += 1

data_yaml = {
    'path': str(Path('~/datasets').expanduser()),
    'train': 'merged/images',
    'val': 'merged/images',
    'nc': nc,
    'names': idx_to_name,
}

yaml_path = Path('datasets/data_all.yaml')
with open(yaml_path, 'w') as f:
    # Write manually to avoid yaml formatting issues with special chars
    f.write(f'path: {data_yaml[\"path\"]}\n')
    f.write(f'train: {data_yaml[\"train\"]}\n')
    f.write(f'val: {data_yaml[\"val\"]}\n')
    f.write(f'nc: {data_yaml[\"nc\"]}\n')
    f.write('names:\n')
    for idx in sorted(idx_to_name.keys()):
        name = idx_to_name[idx].replace(\"'\", \"''\")
        f.write(f\"  {idx}: '{name}'\n\")

print(f'Created {yaml_path} with {nc} classes')
"

# Verify
echo "=== Verification ==="
echo "Images:" && ls ~/datasets/merged/images/*.jpg | wc -l
echo "Labels:" && ls ~/datasets/merged/labels/*.txt | wc -l
echo "data_all.yaml:" && head -5 ~/datasets/data_all.yaml

# Cleanup
rm -f ~/coco_dataset.zip
rm -rf ~/coco_raw

echo "=== Setup complete ==="
