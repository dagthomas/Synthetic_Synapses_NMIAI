"""Test TOP_K and MIN_SCORE variations on val split to find best generalization config."""

import json, time, copy, numpy as np
from pathlib import Path
import torch, torch.nn.functional as F

_orig = torch.load
def _patch(*a, **k):
    k.setdefault('weights_only', False)
    return _orig(*a, **k)
torch.load = _patch

from ultralytics import YOLO
from classifier import load_classifier
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import roi_align

ROOT = Path(__file__).resolve().parent
ANNO = "X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json"
device = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def compute_map(coco_gt, preds, cat_agnostic=False, img_ids=None):
    """Compute mAP, optionally restricted to specific image IDs."""
    if not preds:
        return 0.0
    if img_ids:
        preds = [p for p in preds if p["image_id"] in img_ids]
    if not preds:
        return 0.0

    if cat_agnostic:
        gt_data = {"images": list(coco_gt.imgs.values()), "annotations": [],
                    "categories": [{"id": 0, "name": "p"}]}
        for aid in coco_gt.getAnnIds():
            a = copy.deepcopy(coco_gt.anns[aid])
            a["category_id"] = 0
            gt_data["annotations"].append(a)
        coco_mod = COCO()
        coco_mod.dataset = gt_data
        coco_mod.createIndex()
        seen = {}
        for p in preds:
            pm = dict(p); pm["category_id"] = 0
            k = (pm["image_id"], tuple(pm["bbox"]))
            if k not in seen or pm["score"] > seen[k]["score"]:
                seen[k] = pm
        dt = coco_mod.loadRes(list(seen.values()))
        ev = COCOeval(coco_mod, dt, "bbox")
        if img_ids:
            ev.params.imgIds = list(img_ids)
    else:
        dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, dt, "bbox")
        if img_ids:
            ev.params.imgIds = list(img_ids)

    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [100, 500, 3000]
    ev.evaluate()
    ev.accumulate()
    s = ev.eval["precision"]
    return float(np.mean(s[:, :, :, 0, 2][s[:, :, :, 0, 2] > -1]))


def run_inference(yolo, classifier, image_files, top_k=2, min_score=0.001,
                  bg_reject_prob=0.5, disagree_weight=0.5):
    """Run full pipeline with configurable parameters."""
    BG_ID = 356
    has_bg = classifier.num_classes > BG_ID
    inp_sz = classifier.input_size
    predictions = []

    for img_path in image_files:
        r = yolo.predict(str(img_path), imgsz=1536, conf=0.001, iou=0.6,
                         max_det=3000, augment=True, verbose=False, device=device)
        r = r[0] if r else None
        if r is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)
        orig = r.orig_img
        ih, iw = orig.shape[:2]

        stem = Path(img_path).stem
        try:
            iid = int(stem.split("_")[-1])
        except ValueError:
            iid = 0

        valid = [i for i in range(len(boxes))
                 if (min(iw, boxes[i][2]) - max(0, boxes[i][0])) >= 5
                 and (min(ih, boxes[i][3]) - max(0, boxes[i][1])) >= 5]

        classified = set()
        if valid:
            ir = np.ascontiguousarray(orig[:, :, ::-1])
            it = torch.from_numpy(ir).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
            vb = boxes[valid]
            bg = torch.from_numpy(vb.copy()).float().to(device)
            bg[:, 0].clamp_(0, iw); bg[:, 1].clamp_(0, ih)
            bg[:, 2].clamp_(0, iw); bg[:, 3].clamp_(0, ih)

            for cs in range(0, len(valid), 500):
                ce = min(cs + 500, len(valid))
                cb = bg[cs:ce]
                rois = torch.cat([torch.zeros(len(cb), 1, device=device), cb], dim=1)
                crops = roi_align(it, rois, output_size=(inp_sz, inp_sz),
                                  spatial_scale=1.0, sampling_ratio=2, aligned=True)
                crops = ((crops - MEAN) / STD).half()
                pad = torch.zeros(64, 3, inp_sz, inp_sz, device=device, dtype=torch.float16)

                for bs in range(0, crops.shape[0], 64):
                    ac = min(64, crops.shape[0] - bs)
                    pad[:ac] = crops[bs:bs + ac]
                    with torch.no_grad():
                        lo = classifier(pad)
                        pr = F.softmax(lo[:ac], dim=1)
                        tp, tc = torch.topk(pr, k=min(top_k, pr.shape[1]), dim=1)
                    tpn, tcn = tp.cpu().numpy(), tc.cpu().numpy()
                    bgp = pr[:, BG_ID].cpu().numpy() if has_bg else None

                    for j in range(ac):
                        bi = valid[cs + bs + j]
                        yc = float(scores[bi])
                        yl = int(labels[bi])
                        classified.add(bi)

                        if has_bg and float(bgp[j]) > bg_reject_prob:
                            continue

                        for k in range(min(top_k, tpn.shape[1])):
                            ci = int(tcn[j, k])
                            cp = float(tpn[j, k])
                            if ci == BG_ID:
                                continue
                            if ci == yl:
                                bl = yc * (0.7 + 0.3 * cp)
                            else:
                                bl = yc * cp * disagree_weight
                            if bl >= min_score:
                                x1, y1, x2, y2 = boxes[bi]
                                predictions.append({
                                    "image_id": iid, "category_id": ci,
                                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                                    "score": round(bl, 4),
                                })
            del it

        for i in range(len(boxes)):
            if i not in classified:
                x1, y1, x2, y2 = boxes[i]
                predictions.append({
                    "image_id": iid, "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2-x1), 2), round(float(y2-y1), 2)],
                    "score": round(float(scores[i]), 4),
                })

    return predictions


def main():
    print("Loading GT...")
    coco_gt = COCO(ANNO)

    # Get val and train image IDs
    val_dir = ROOT / "datasets" / "val" / "images"
    train_dir = ROOT / "datasets" / "train" / "images"

    def get_img_ids(d):
        ids = set()
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for f in d.glob(ext):
                try:
                    ids.add(int(f.stem.split("_")[-1]))
                except ValueError:
                    pass
        return ids

    val_ids = get_img_ids(val_dir)
    train_ids = get_img_ids(train_dir)
    all_imgs = sorted(list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.jpeg")) +
                      list(val_dir.glob("*.png")) + list(train_dir.glob("*.jpg")) +
                      list(train_dir.glob("*.jpeg")) + list(train_dir.glob("*.png")))
    print(f"Val: {len(val_ids)} images, Train: {len(train_ids)} images, Total: {len(all_imgs)}")

    print("Loading models...")
    yolo = YOLO(str(ROOT / "best.pt"))
    classifier = load_classifier(str(ROOT / "classifier.safetensors"), device=device)

    # Test configurations
    configs = [
        {"top_k": 2, "min_score": 0.001, "bg_reject_prob": 0.5, "disagree_weight": 0.5, "name": "baseline_k2"},
        {"top_k": 1, "min_score": 0.001, "bg_reject_prob": 0.5, "disagree_weight": 0.5, "name": "k1"},
        {"top_k": 1, "min_score": 0.005, "bg_reject_prob": 0.5, "disagree_weight": 0.5, "name": "k1_ms005"},
        {"top_k": 1, "min_score": 0.01, "bg_reject_prob": 0.5, "disagree_weight": 0.5, "name": "k1_ms01"},
        {"top_k": 2, "min_score": 0.001, "bg_reject_prob": 0.3, "disagree_weight": 0.3, "name": "k2_aggr_bg03_dw03"},
        {"top_k": 2, "min_score": 0.005, "bg_reject_prob": 0.5, "disagree_weight": 0.3, "name": "k2_ms005_dw03"},
        {"top_k": 1, "min_score": 0.001, "bg_reject_prob": 0.3, "disagree_weight": 0.5, "name": "k1_bg03"},
        {"top_k": 2, "min_score": 0.001, "bg_reject_prob": 0.5, "disagree_weight": 0.0, "name": "k2_no_disagree"},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        print(f"\n=== {name} ===")
        t0 = time.time()
        preds = run_inference(yolo, classifier, all_imgs, **cfg)
        inf_time = time.time() - t0

        # Evaluate on val
        det_val = compute_map(coco_gt, preds, True, val_ids)
        cls_val = compute_map(coco_gt, preds, False, val_ids)
        comb_val = 0.7 * det_val + 0.3 * cls_val

        # Evaluate on train
        det_trn = compute_map(coco_gt, preds, True, train_ids)
        cls_trn = compute_map(coco_gt, preds, False, train_ids)
        comb_trn = 0.7 * det_trn + 0.3 * cls_trn

        gap = comb_trn - comb_val
        n_val = len([p for p in preds if p["image_id"] in val_ids])
        n_trn = len([p for p in preds if p["image_id"] in train_ids])

        print(f"  Val:   det={det_val:.4f} cls={cls_val:.4f} comb={comb_val:.4f} ({n_val} preds)")
        print(f"  Train: det={det_trn:.4f} cls={cls_trn:.4f} comb={comb_trn:.4f} ({n_trn} preds)")
        print(f"  Gap: {gap:+.4f}, Time: {inf_time:.1f}s")

        results.append({
            "name": name, "n_val": n_val, "n_trn": n_trn,
            "val_det": det_val, "val_cls": cls_val, "val_comb": comb_val,
            "trn_det": det_trn, "trn_cls": cls_trn, "trn_comb": comb_trn,
            "gap": gap, **cfg,
        })

    # Summary sorted by val combined (what matters for test!)
    print("\n" + "=" * 120)
    print(f"{'Config':<25} {'Val Det':>8} {'Val Cls':>8} {'Val Comb':>9} {'Trn Comb':>9} {'Gap':>7} {'VPreds':>7} {'TPreds':>7}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: -x["val_comb"]):
        print(f"{r['name']:<25} {r['val_det']:>8.4f} {r['val_cls']:>8.4f} {r['val_comb']:>9.4f} "
              f"{r['trn_comb']:>9.4f} {r['gap']:>+7.4f} {r['n_val']:>7} {r['n_trn']:>7}")
    print("=" * 120)


if __name__ == "__main__":
    main()
