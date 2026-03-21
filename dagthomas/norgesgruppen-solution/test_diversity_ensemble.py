"""Test old YOLO (mixup=0.15) + new YOLO (mixup=0.0) ensemble.
These models have genuinely different training augmentation, providing real diversity."""

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
from torchvision.ops import roi_align, nms as tv_nms

ROOT = Path(__file__).resolve().parent
ANNO = "X:/norgesgruppen/NM_NGD_coco_dataset/train/annotations.json"
device = "cuda"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def compute_map(coco_gt, preds, cat_agnostic=False):
    valid_ids = set(coco_gt.getImgIds())
    preds = [p for p in preds if p["image_id"] in valid_ids]
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
            pm = dict(p)
            pm["category_id"] = 0
            k = (pm["image_id"], tuple(pm["bbox"]))
            if k not in seen or pm["score"] > seen[k]["score"]:
                seen[k] = pm
        dt = coco_mod.loadRes(list(seen.values()))
        ev = COCOeval(coco_mod, dt, "bbox")
    else:
        dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, dt, "bbox")
    ev.params.iouThrs = np.array([0.5])
    ev.params.maxDets = [100, 500, 3000]
    ev.evaluate()
    ev.accumulate()
    s = ev.eval["precision"]
    return float(np.mean(s[:, :, :, 0, 2][s[:, :, :, 0, 2] > -1]))


def run_yolo(model, imgs, aug=True, imgsz=1536):
    dets = {}
    for p in imgs:
        r = model.predict(str(p), imgsz=imgsz, conf=0.001, iou=0.6,
                          max_det=3000, augment=aug, verbose=False, device=device)
        r = r[0] if r else None
        if r and len(r.boxes) > 0:
            dets[str(p)] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int), r.orig_img)
        elif r:
            dets[str(p)] = (np.empty((0, 4)), np.empty(0),
                            np.empty(0, dtype=int), r.orig_img)
    return dets


def merge(det_a, det_b, iou_thr=0.6):
    merged = {}
    for p in set(list(det_a.keys()) + list(det_b.keys())):
        dl = []
        orig = None
        for d in [det_a, det_b]:
            if p in d:
                b, s, l, oi = d[p]
                orig = oi
                if len(b) > 0:
                    dl.append((b, s, l))
        if dl:
            if len(dl) > 1:
                ab = np.concatenate([x[0] for x in dl])
                asc = np.concatenate([x[1] for x in dl])
                al = np.concatenate([x[2] for x in dl])
                keep_all = []
                for cid in np.unique(al):
                    mask = al == cid
                    idx = np.where(mask)[0]
                    k = tv_nms(torch.from_numpy(ab[mask]).float(),
                               torch.from_numpy(asc[mask]).float(), iou_thr).numpy()
                    keep_all.append(idx[k])
                keep = np.concatenate(keep_all) if keep_all else np.array([], dtype=int)
                keep.sort()
                merged[p] = (ab[keep], asc[keep], al[keep], orig)
            else:
                merged[p] = (dl[0][0], dl[0][1], dl[0][2], orig)
    return merged


def classify(dets, classifier, inp_sz, mean_gpu, std_gpu):
    BG_ID = 356
    BG_THR = 0.5
    has_bg = classifier.num_classes > BG_ID
    preds = []

    for p, (boxes, scores, labels, orig) in dets.items():
        if len(boxes) == 0:
            continue
        stem = Path(p).stem
        try:
            iid = int(stem.split("_")[-1])
        except ValueError:
            iid = 0
        ih, iw = orig.shape[:2]
        valid = [i for i in range(len(boxes))
                 if (min(iw, boxes[i][2]) - max(0, boxes[i][0])) >= 5
                 and (min(ih, boxes[i][3]) - max(0, boxes[i][1])) >= 5]
        classified = set()

        if valid:
            ir = np.ascontiguousarray(orig[:, :, ::-1])
            it = torch.from_numpy(ir).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
            vb = boxes[valid]
            bg = torch.from_numpy(vb.copy()).float().to(device)
            bg[:, 0].clamp_(0, iw)
            bg[:, 1].clamp_(0, ih)
            bg[:, 2].clamp_(0, iw)
            bg[:, 3].clamp_(0, ih)

            for cs in range(0, len(valid), 500):
                ce = min(cs + 500, len(valid))
                cb = bg[cs:ce]
                rois = torch.cat([torch.zeros(len(cb), 1, device=device), cb], dim=1)
                crops = roi_align(it, rois, output_size=(inp_sz, inp_sz),
                                  spatial_scale=1.0, sampling_ratio=2, aligned=True)
                crops = ((crops - mean_gpu) / std_gpu).half()
                pad = torch.zeros(64, 3, inp_sz, inp_sz, device=device, dtype=torch.float16)

                for bs in range(0, crops.shape[0], 64):
                    ac = min(64, crops.shape[0] - bs)
                    pad[:ac] = crops[bs:bs + ac]
                    with torch.no_grad():
                        lo = classifier(pad)
                        pr = F.softmax(lo[:ac], dim=1)
                        tp, tc = torch.topk(pr, k=2, dim=1)
                    tpn, tcn = tp.cpu().numpy(), tc.cpu().numpy()
                    bgp = pr[:, BG_ID].cpu().numpy() if has_bg else None

                    for j in range(ac):
                        bi = valid[cs + bs + j]
                        yc = float(scores[bi])
                        yl = int(labels[bi])
                        classified.add(bi)
                        if has_bg and float(bgp[j]) > BG_THR:
                            continue
                        for k in range(2):
                            ci = int(tcn[j, k])
                            cp = float(tpn[j, k])
                            if ci == BG_ID:
                                continue
                            bl = yc * (0.7 + 0.3 * cp) if ci == yl else yc * cp * 0.5
                            if bl >= 0.001:
                                x1, y1, x2, y2 = boxes[bi]
                                preds.append({
                                    "image_id": iid, "category_id": ci,
                                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                    "score": round(bl, 4),
                                })
            del it

        for i in range(len(boxes)):
            if i not in classified:
                x1, y1, x2, y2 = boxes[i]
                preds.append({
                    "image_id": iid, "category_id": int(labels[i]),
                    "bbox": [round(float(x1), 2), round(float(y1), 2),
                             round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                    "score": round(float(scores[i]), 4),
                })
    return preds


def main():
    print(f"Device: {device}")
    print("Loading GT...")
    coco_gt = COCO(ANNO)

    print("Loading classifier...")
    cls = load_classifier(str(ROOT / "classifier.safetensors"), device=device)
    inp_sz = cls.input_size
    mean_gpu = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_gpu = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    # Get all images
    imgs = []
    for d in [ROOT / "datasets" / "train" / "images", ROOT / "datasets" / "val" / "images"]:
        if d.exists():
            imgs.extend(d.glob("*.jpg"))
            imgs.extend(d.glob("*.jpeg"))
            imgs.extend(d.glob("*.png"))
    imgs = sorted(set(imgs))
    print(f"{len(imgs)} images")

    print("Loading YOLO new (mixup=0.0)...")
    yolo_new = YOLO(str(ROOT / "best.pt"))
    print("Loading YOLO old (mixup=0.15)...")
    yolo_old = YOLO(str(ROOT / "best_valrun_backup.pt"))

    print("\nRunning NEW YOLO (TTA)...")
    t0 = time.time()
    det_new = run_yolo(yolo_new, imgs, aug=True)
    print(f"  {time.time() - t0:.1f}s, {sum(len(d[0]) for d in det_new.values())} dets")

    print("Running OLD YOLO (TTA)...")
    t0 = time.time()
    det_old = run_yolo(yolo_old, imgs, aug=True)
    print(f"  {time.time() - t0:.1f}s, {sum(len(d[0]) for d in det_old.values())} dets")

    configs = {}

    # 1. New YOLO only (baseline)
    print("\n=== 1. Baseline: new YOLO TTA ===")
    t0 = time.time()
    p = classify(det_new, cls, inp_sz, mean_gpu, std_gpu)
    d = compute_map(coco_gt, p, True)
    c = compute_map(coco_gt, p, False)
    cb = 0.7 * d + 0.3 * c
    print(f"  det={d:.4f} cls={c:.4f} comb={cb:.4f} ({len(p)} preds, {time.time()-t0:.1f}s)")
    configs["1_new_only"] = (d, c, cb, len(p))

    # 2. Old YOLO only
    print("\n=== 2. Old YOLO TTA ===")
    t0 = time.time()
    p = classify(det_old, cls, inp_sz, mean_gpu, std_gpu)
    d = compute_map(coco_gt, p, True)
    c = compute_map(coco_gt, p, False)
    cb = 0.7 * d + 0.3 * c
    print(f"  det={d:.4f} cls={c:.4f} comb={cb:.4f} ({len(p)} preds, {time.time()-t0:.1f}s)")
    configs["2_old_only"] = (d, c, cb, len(p))

    # 3. Ensemble old+new, NMS=0.6
    print("\n=== 3. Ensemble old+new, NMS=0.6 ===")
    t0 = time.time()
    det_ens = merge(det_new, det_old, 0.6)
    p = classify(det_ens, cls, inp_sz, mean_gpu, std_gpu)
    d = compute_map(coco_gt, p, True)
    c = compute_map(coco_gt, p, False)
    cb = 0.7 * d + 0.3 * c
    print(f"  det={d:.4f} cls={c:.4f} comb={cb:.4f} ({len(p)} preds, {time.time()-t0:.1f}s)")
    configs["3_ens_nms06"] = (d, c, cb, len(p))

    # 4. Ensemble old+new, NMS=0.5
    print("\n=== 4. Ensemble old+new, NMS=0.5 ===")
    t0 = time.time()
    det_ens5 = merge(det_new, det_old, 0.5)
    p = classify(det_ens5, cls, inp_sz, mean_gpu, std_gpu)
    d = compute_map(coco_gt, p, True)
    c = compute_map(coco_gt, p, False)
    cb = 0.7 * d + 0.3 * c
    print(f"  det={d:.4f} cls={c:.4f} comb={cb:.4f} ({len(p)} preds, {time.time()-t0:.1f}s)")
    configs["4_ens_nms05"] = (d, c, cb, len(p))

    # 5. Ensemble old+new, NMS=0.7 (keep more)
    print("\n=== 5. Ensemble old+new, NMS=0.7 ===")
    t0 = time.time()
    det_ens7 = merge(det_new, det_old, 0.7)
    p = classify(det_ens7, cls, inp_sz, mean_gpu, std_gpu)
    d = compute_map(coco_gt, p, True)
    c = compute_map(coco_gt, p, False)
    cb = 0.7 * d + 0.3 * c
    print(f"  det={d:.4f} cls={c:.4f} comb={cb:.4f} ({len(p)} preds, {time.time()-t0:.1f}s)")
    configs["5_ens_nms07"] = (d, c, cb, len(p))

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Config':<30} {'Preds':>7} {'Det':>7} {'Cls':>7} {'Comb':>7} {'Delta':>7}")
    print("-" * 80)
    baseline = configs["1_new_only"][2]
    for n, (d, c, cb, np_) in sorted(configs.items(), key=lambda x: -x[1][2]):
        delta = cb - baseline
        ds = f"{delta:+.4f}" if n != "1_new_only" else "   ---"
        print(f"{n:<30} {np_:>7} {d:>7.4f} {c:>7.4f} {cb:>7.4f} {ds:>7}")
    print("=" * 80)


if __name__ == "__main__":
    main()
