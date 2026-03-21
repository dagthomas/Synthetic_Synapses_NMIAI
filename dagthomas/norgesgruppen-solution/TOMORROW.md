# Submission Plan — March 22, 2026

Current best: **0.9002** (rank #23). Leader: **0.9255**. Gap: 0.025.

## Quick Reference
```bash
export PATH="/c/Users/dagth/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin:$PATH"
```

## What's Training Overnight

| VM | Zone | Seeds | Script | imgsz | Epochs | Key Changes |
|----|------|-------|--------|-------|--------|-------------|
| nmiai-train | us-central1-f | 42, 123, 256 | cloud_train_overnight.py | 1280 | 120 | Exact v1 copy (safe) |
| nmiai-train2 | us-east1-b | 7, 99 | cloud_train_improved.py | 1536 | 150 | Higher res, more augmentation |

**Previous failures fixed:**
- `degrees=10.0` → `0.0` (no rotation — products are always upright)
- Missing 38 `.jpeg` images → all 248 images now included
- `lr0=0.0005` → `0.001` (proper learning rate)
- `scale=0.6` → `0.5`, `translate=0.15` → `0.1`

## Step 1: Check overnight training results

```bash
# VM1 — safe models (seeds 42, 123, 256)
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  tail -5 ~/overnight_train.log
  echo "=== Stripped models ==="
  ls -lh ~/best_yolov8x_*.pt 2>/dev/null'

# VM2 — improved models (seeds 7, 99)
gcloud compute ssh nmiai-train2 --zone=us-east1-b --command='
  tail -5 ~/overnight_train.log
  echo "=== Stripped models ==="
  ls -lh ~/best_yolov8x_*.pt 2>/dev/null'
```

## Step 2: Run eval on VMs

```bash
# VM1
gcloud compute ssh nmiai-train --zone=us-central1-f --command='
  python3 cloud_test.py --use-train 2>&1 | tail -20'

# VM2
gcloud compute ssh nmiai-train2 --zone=us-east1-b --command='
  python3 cloud_test.py --use-train 2>&1 | tail -20'
```

Look at the **SUMMARY** table — pick the best 2 models (one from each VM ideally for diversity).

## Step 3: Download best models

Models are already stripped (optimizer removed, ~131MB each).

```bash
# Download from VM1 (replace seed with the best one)
gcloud compute scp nmiai-train:/home/dagth/best_yolov8x_seed42.pt \
  "X:/KODE/AINMNO/norgesgruppen-solution/best_cloud_a.pt" --zone=us-central1-f

# Download from VM2 (replace seed with the best one)
gcloud compute scp nmiai-train2:/home/dagth/best_yolov8x_v3_seed7.pt \
  "X:/KODE/AINMNO/norgesgruppen-solution/best_cloud_b.pt" --zone=us-east1-b
```

## Step 4: Build submissions

### Size budget (420 MB max zip)
| Config | Uncompressed | Compressed | Fits? |
|--------|-------------|------------|-------|
| 2 models + cls nano (30MB) | 294 MB | ~270 MB | YES |
| 2 models + cls tiny (54MB) | 318 MB | ~295 MB | YES |
| 3 models + cls nano | 426 MB | ~390 MB | YES |
| 3 models + cls tiny | 450 MB | ~410 MB | TIGHT |

### Sub A: Best cloud model + local v1 + cls nano (SAFE)
```bash
cd X:/KODE/AINMNO/submissions
mkdir -p sub15_cloud_ensemble
cp sub12_ensemble/run.py sub15_cloud_ensemble/
cp sub12_ensemble/classifier.py sub15_cloud_ensemble/
cp sub12_ensemble/classifier_config.json sub15_cloud_ensemble/
cp ../norgesgruppen-solution/best.pt sub15_cloud_ensemble/           # local v1
cp ../norgesgruppen-solution/best_cloud_a.pt sub15_cloud_ensemble/best_v2.pt  # best cloud
cp ../norgesgruppen-solution/classifier.safetensors sub15_cloud_ensemble/
cd sub15_cloud_ensemble
powershell -c "Compress-Archive -Path '.\*' -DestinationPath '..\submission_sub15.zip' -Force"
ls -lh ../submission_sub15.zip
```

### Sub B: Two best cloud models + local v1 (3-model if fits)
```bash
# Only if sub15 scores well — add the second cloud model
cp ../norgesgruppen-solution/best_cloud_b.pt sub15_cloud_ensemble/best_v3.pt
cd sub15_cloud_ensemble
powershell -c "Compress-Archive -Path '.\*' -DestinationPath '..\submission_sub16_3model.zip' -Force"
ls -lh ../submission_sub16_3model.zip
# WARNING: check size < 420MB! If too big, remove best_v3.pt
```

### Sub C: With classifier_tiny (if eval shows improvement)
```bash
mkdir -p sub17_tiny_cls
cp sub15_cloud_ensemble/* sub17_tiny_cls/
cp ../norgesgruppen-solution/classifier_tiny.safetensors sub17_tiny_cls/classifier.safetensors
echo '{"model_name":"convnextv2_tiny","input_size":256,"num_classes":357}' > sub17_tiny_cls/classifier_config.json
cd sub17_tiny_cls
powershell -c "Compress-Archive -Path '.\*' -DestinationPath '..\submission_sub17_tiny.zip' -Force"
```

## Step 5: Local synth test (optional, quick sanity check)

```bash
cd X:/KODE/AINMNO/norgesgruppen-solution
python synth_test.py
```
Note: local synth test scores are NOT representative of competition scores (tiny val set, different eval). Use it only to compare configs relative to each other.

## Step 6: Submit

Upload at https://app.ainm.no — NorgesGruppen Data challenge.
- Max 420 MB zip, 300s timeout on L4 GPU
- 6 submissions/day

### Submission priority:
1. **Sub A** (2-model: local v1 + best cloud + cls nano) — should beat 0.9002
2. **Sub B** (3-model if fits) — more diversity, but timing tight (~287s)
3. **Sub C** (with cls_tiny) — if Sub A works, try bigger classifier
4. **Sub12** (existing 0.9002 submission) — resubmit as safety if new ones fail

## Step 7: DELETE VMs (CRITICAL — $3.67/hr each!)

```bash
gcloud compute instances delete nmiai-train --zone=us-central1-f --quiet
gcloud compute instances delete nmiai-train2 --zone=us-east1-b --quiet
gcloud compute instances list  # verify empty
```

## Key Findings from Tonight

### Why cloud models failed before:
1. **`degrees=10.0`** — rotation augmentation on upright grocery products (should be 0.0)
2. **38 missing `.jpeg` images** — `*.jpg` glob missed `.jpeg` files (now fixed, all 248 included)
3. **Wrong LR** — `lr0=0.0005` instead of `0.001`
4. **Over-aggressive augmentation** — `scale=0.6`, `translate=0.15`
5. **Premature kills** — `run_extra.sh` was killing training processes mid-training

### What the improved models (VM2) change:
- `imgsz=1536` — trains at the same resolution used during inference
- `copy_paste=0.2` — more synthetic occlusion for dense shelves
- `close_mosaic=15` — keeps mosaic augmentation active 15 epochs longer
- `erasing=0.5` — more random erasing for robustness
- `patience=30` — auto-stops if overfitting

## Timing Budget on L4 (300s timeout)

| Config | Est. time | Fits? |
|--------|-----------|-------|
| 2 models: v1 @1536 TTA + v2 @1280 + cls | ~260s | YES |
| 2 models: v1 @1536 TTA + v2 @1536 TTA + cls | ~291s | TIGHT |
| 3 models: all @1280 no TTA + cls | ~287s | TIGHT |
| 3 models + TTA | ~330s+ | NO |

## File Inventory

| File | Size | What |
|------|------|------|
| `best.pt` | 132 MB | YOLO v1 — local training, in 0.9002 submission |
| `best_v2.pt` | 132 MB | YOLO v2 — cloud, old (bad hyperparams) |
| `best_cloud_a.pt` | ~131 MB | Best from VM1 safe training (download tomorrow) |
| `best_cloud_b.pt` | ~131 MB | Best from VM2 improved training (download tomorrow) |
| `classifier.safetensors` | 30 MB | ConvNeXt-V2 nano |
| `classifier_tiny.safetensors` | 54 MB | ConvNeXt-V2 tiny |
