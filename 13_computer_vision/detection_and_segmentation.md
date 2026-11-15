# Object Detection and Image Segmentation: Concepts, Metrics (IoU, mAP, Dice) and PyTorch Practice

> **Folder:** 13_computer_vision · **Prev:** [image_classification.md](image_classification.md) · **Next:** [recsys_workflow.md](../14_recommender_systems/recsys_workflow.md) · **Related:** [image_preprocessing_augmentation.md](image_preprocessing_augmentation.md) (joint transforms of boxes/masks)

---

## 1. Purpose
Go beyond image-level labels: locate objects with bounding boxes (detection) or classify every pixel (segmentation). Know how to measure these outputs (IoU, precision/recall at IoU thresholds, AP/mAP, Dice), and how to set up practical PyTorch baselines.

## 2. When to use it
- **Detection:** counting or locating discrete objects (defects on a sheet, components on a board, people/vehicles in a safety zone).
- **Segmentation:** measuring area/shape/length (corrosion area, crack length, phase fractions in micrographs, flooded areas in satellite images).

## 3. Conceptual foundation

### 3.1 Intersection over Union (IoU)
IoU(A, B) = |A ∩ B| / |A ∪ B| for boxes or masks. A detection counts as a **true positive** if its IoU with an unmatched ground-truth object is ≥ a threshold (0.5 is common; COCO averages 0.5 to 0.95).

### 3.2 Detection metrics
- Sort the detections by confidence; match greedily to the ground truth (per class, per image), giving TP/FP. Unmatched ground truths are FN.
- The **precision–recall curve** over the confidence thresholds gives the **AP** (area under it, interpolated). **mAP** is the mean over classes (and over IoU thresholds for COCO-style mAP@[.5:.95]).
- Operational metrics: recall at a fixed confidence, false positives per image, counting error.

### 3.3 Segmentation metrics
- **IoU (Jaccard)** per class = TP / (TP + FP + FN) over pixels. **mIoU** is the mean over classes.
- **Dice (F1)** = 2TP / (2TP + FP + FN) = 2·IoU / (1 + IoU). It is more forgiving for small objects.
- **Pixel accuracy** is misleading when the background dominates (98% background → 98% accuracy for an all-background prediction).
- Use **per-image** metrics for small objects (a dataset-level IoU is dominated by large objects), and **object-level** metrics (was each defect found at all?) for inspection.

### 3.4 Model families (torchvision and others)
| Task | Two-stage / accurate | One-stage / fast | Notes |
|---|---|---|---|
| Detection | Faster R-CNN (`fasterrcnn_resnet50_fpn_v2`) | RetinaNet, FCOS, SSD; YOLO (Ultralytics, external) | Fine-tune COCO-pretrained weights |
| Semantic segmentation | DeepLabV3, FCN (torchvision); U-Net (common in science/industry) | — | Pretrained encoders help |
| Instance segmentation | Mask R-CNN (`maskrcnn_resnet50_fpn_v2`) | — | Needs instance masks |

## 4. Assumptions and limitations
- Annotation quality bounds performance (box tightness, mask edges). Measure the inter-annotator IoU on a sample.
- Very small objects (a few pixels) are hard for standard detectors. Use tiling, higher resolution, or segmentation.
- Detection metrics depend on the matching conventions. Use a standard implementation (e.g. `torchmetrics` or `pycocotools`) for reported numbers. The manual code below is for understanding.

## 5. Step-by-step procedure
1. **Choose the task** from the decision (counting/locating → detection; area/shape → segmentation).
2. **Annotate consistently** (guidelines; QA on a sample). Convert to the model format (boxes as `[x1, y1, x2, y2]` in pixels; masks as integer label maps).
3. **Baseline:** a pretrained detector/segmenter, fine-tuned; or classical image processing (thresholding + morphology) when the contrast is clear. That is surprisingly strong in controlled inspection setups.
4. **Train** with joint augmentations; validate with the task metrics.
5. **Evaluate:** AP/mAP (detection) or IoU/Dice per class and per image (segmentation), plus object-level recall.
6. **Error analysis:** missed small objects, false positives on texture, boundary errors.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from cookbook_data import make_surface_defect_images

imgs, labels, masks = make_surface_defect_images(n=1200)

# --- IoU for boxes and masks --------------------------------------------------
def box_iou(a, b):
    """a, b: [x1, y1, x2, y2]."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0

def mask_iou_dice(pred, gt):
    tp = np.logical_and(pred, gt).sum(); fp = np.logical_and(pred, ~gt).sum(); fn = np.logical_and(~pred, gt).sum()
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return iou, (2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)

print("box IoU example:", round(box_iou([10, 10, 30, 30], [15, 15, 35, 35]), 3))

# Ground-truth boxes derived from masks (connected components)
def boxes_from_mask(m, min_area=3):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return [[x, y, x + w, y + h] for x, y, w, h, a in st[1:] if a >= min_area]
gt_boxes = [boxes_from_mask(m) for m in masks]
print("objects per defect image:", pd.Series([len(b) for b, l in zip(gt_boxes, labels) if l != "ok"]).value_counts().to_dict())
```

```python
# --- Classical baseline: local-contrast thresholding + morphology → masks and boxes ---
def classical_segment(img, k=3.0, open_px=0):
    bg = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 4)            # local background (removes illumination gradient)
    diff = img.astype(np.float32) - bg
    thr = k * 1.4826 * np.median(np.abs(diff - np.median(diff)))      # robust threshold on the local contrast
    m = (np.abs(diff) > thr).astype(np.uint8)
    if open_px:                                                         # morphological opening removes speckle...
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_px, open_px), np.uint8))   # ...but also 1-px lines!
    return m.astype(bool)

pred_masks = np.array([classical_segment(im) for im in imgs])
per_image = [mask_iou_dice(p, g) for p, g, l in zip(pred_masks, masks, labels) if l != "ok"]
print(f"classical baseline on defect images: mean IoU={np.mean([a for a, _ in per_image]):.3f}, "
      f"mean Dice={np.mean([b for _, b in per_image]):.3f}")
fp_ok = np.mean([p.sum() > 10 for p, l in zip(pred_masks, labels) if l == "ok"])
print(f"share of OK images with spurious detections (>10 px): {fp_ok:.2f}")
```

```python
# --- Average precision for detection, implemented by hand (for understanding) ---
def average_precision(dets, gts, iou_thr=0.5):
    """dets: list of (image_id, score, box); gts: dict image_id -> list of boxes. Single class."""
    n_gt = sum(len(v) for v in gts.values()); matched = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = []
    for img_id, score, box in sorted(dets, key=lambda d: -d[1]):
        ious = [box_iou(box, g) for g in gts.get(img_id, [])]
        j = int(np.argmax(ious)) if ious else -1
        if j >= 0 and ious[j] >= iou_thr and not matched[img_id][j]:
            matched[img_id][j] = True; tp.append(1)
        else:
            tp.append(0)
    tp = np.array(tp); fp = 1 - tp
    rec = np.cumsum(tp) / max(n_gt, 1); prec = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]                         # precision envelope
    r_grid = np.linspace(0, 1, 101)                                        # 101-point interpolation (COCO style)
    return float(np.mean([prec[rec >= r].max() if np.any(rec >= r) else 0 for r in r_grid]))

# Detections from the classical segmenter: boxes of connected components, score = mean |contrast| in the box
dets = []
for i, (im, pm) in enumerate(zip(imgs, pred_masks)):
    for b in boxes_from_mask(pm, min_area=4):
        x1, y1, x2, y2 = b
        dets.append((i, float(np.abs(im[y1:y2, x1:x2].astype(float) - im.mean()).mean()), b))
gts = {i: b for i, b in enumerate(gt_boxes)}
for thr in (0.3, 0.5):
    print(f"classical detector AP@IoU{thr}: {average_precision(dets, gts, thr):.3f}")
```

```python
# --- A small U-Net for binary defect segmentation (CPU, a few epochs) ---
torch.manual_seed(0)
class TinyUNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        def conv(cin, cout): return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                                  nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())
        self.e1, self.e2, self.b = conv(1, c), conv(c, 2 * c), conv(2 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), conv(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), conv(2 * c, c)
        self.out, self.pool = nn.Conv2d(c, 1, 1), nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)                                               # logits (N, 1, H, W)

X = (torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255 - 0.5) / 0.1
M = torch.tensor(masks, dtype=torch.float32).unsqueeze(1)
tr, te = np.arange(0, 900), np.arange(900, 1200)
net = TinyUNet(); opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
pos_weight = torch.tensor([(1 - M[tr].mean()) / M[tr].mean()]).clamp(max=20)       # defects are ~1% of pixels
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits); inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()
for epoch in range(6):
    net.train()
    for b in torch.split(torch.from_numpy(np.random.default_rng(epoch).permutation(tr)), 32):
        opt.zero_grad(); out = net(X[b]); loss = bce(out, M[b]) + dice_loss(out, M[b]); loss.backward(); opt.step()
net.eval()
with torch.no_grad():
    P = (torch.sigmoid(net(X[te])) > 0.5).numpy()[:, 0]
scores = [mask_iou_dice(p, masks[i]) for p, i in zip(P, te) if labels[i] != "ok"]
print(f"TinyUNet on test defect images: mean IoU={np.mean([a for a, _ in scores]):.3f}, mean Dice={np.mean([b for _, b in scores]):.3f}")
print(f"share of OK test images with >10 predicted defect pixels: {np.mean([p.sum() > 10 for p, i in zip(P, te) if labels[i] == 'ok']):.2f}")
```

Fine-tuning a pretrained detector (torchvision). Downloads COCO weights; GPU recommended:

```python
# not-run (downloads COCO-pretrained weights; GPU recommended)
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=1 + 3)    # background + scratch/spot/crack
# Training data: list of images (3, H, W) float [0, 1] and targets [{"boxes": FloatTensor[N, 4], "labels": Int64Tensor[N]}]
model.train()
# loss_dict = model(images, targets); loss = sum(loss_dict.values()); loss.backward(); optimizer.step()
model.eval()
# preds = model(images)  → [{"boxes", "labels", "scores"}]; evaluate with torchmetrics.detection.MeanAveragePrecision
```

## 7. Interpretation
- **Objects per defect image:** mostly one connected component, but thin scratches/cracks can break into several pieces. Box-level ground truth from masks inherits this (a split crack counts as several objects). Decide whether "one crack = one object" in the annotation guideline.
- **The classical baseline fails on this data** (mean IoU ≈ 0.17, with spurious detections in ~80% of the OK images). The surface texture has local contrast comparable to the defects, so a contrast threshold either misses the faint defects or fires on the texture. A morphological opening, the usual speckle remedy, also erases the 1-px scratches and cracks. While writing this recipe, k = 1.5–3 with and without opening were tried, and none gives a useful trade-off. In controlled setups with smooth backgrounds such baselines can work well, and they are always worth a quick try (interpretable, no training data). Here the result justifies a learned model.
- **AP of the classical detector ≈ 0:** most of its "detections" are texture blobs, and its boxes around real thin defects rarely reach IoU 0.3–0.5. In general, a higher AP at a looser IoU threshold means the detections are roughly located but the boxes are imprecise, which is typical for thin objects (a one-pixel offset destroys the IoU of a thin box). For thin, elongated defects, segmentation with object-level recall is often a better formulation than boxes.
- **TinyUNet:** after only 6 CPU epochs (≈ 20 s) it reaches a mean IoU ≈ 0.57 / Dice ≈ 0.69 on the test defect images, with **no** spurious detections on OK images. The learned model separates thin, low-contrast defects from texture, which the threshold could not. The combined BCE (with a positive weight) and Dice loss handles the extreme pixel imbalance (~1% defect pixels). Longer training, augmentation and a pretrained encoder would improve it further. IoU on thin structures is unforgiving: a mask one pixel too wide halves the IoU of a 1-px scratch. Complement it with object-level recall.

## 8. Decision framework
- **IF** the defects have clear contrast in a controlled setup → try classical image processing first; ML if the variability defeats it.
- **IF** the decision needs counts/locations → detection. **IF** it needs area/length/shape, or the objects are thin/amorphous → segmentation.
- **IF** the objects are tiny relative to the image → tile the images; increase the resolution; segmentation; anchor sizes adapted (detection).
- **IF** the pixel classes are extremely imbalanced → Dice/focal losses or positive weights; per-image and object-level metrics.
- **IF** reporting detection results → a standard mAP implementation (torchmetrics / pycocotools), and the stated IoU thresholds.
- **IF** labelled masks are scarce → weak supervision from boxes, or classification + class-activation maps for rough localisation (an extension).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Pixel accuracy for segmentation | Meaningless (background dominates) | IoU/Dice per class, per image |
| Dataset-level IoU only | Small objects ignored | Per-image and object-level metrics |
| Non-standard mAP code for reported results | Incomparable numbers | Standard libraries |
| Augmenting the images but not the boxes/masks | Corrupted targets | Joint transforms |
| Box formats mixed up (xywh vs xyxy) | Silent garbage training | Assert the formats; visualise the targets |
| Thin objects with box metrics only | Misleadingly low AP | Segmentation or relaxed IoU + object recall |

## 10. Checklist
- [ ] Task chosen from the decision; annotation guideline written.
- [ ] Targets visualised (boxes/masks on images).
- [ ] Classical baseline where applicable; pretrained model fine-tuned.
- [ ] Losses suited to the imbalance (segmentation).
- [ ] Metrics: AP/mAP with stated IoU, or IoU/Dice per image + object-level recall; false positives on clean images.
- [ ] Error analysis on small/thin objects and texture false positives.

## 11. References
- Everingham, M. et al. (2010). "The Pascal Visual Object Classes (VOC) Challenge." *IJCV* 88, 303–338 (AP definition).
- Lin, T.-Y. et al. (2014). "Microsoft COCO: Common Objects in Context." *ECCV* (mAP@[.5:.95]). https://arxiv.org/abs/1405.0312
- Ren, S., He, K., Girshick, R. & Sun, J. (2015). "Faster R-CNN." *NeurIPS*. https://arxiv.org/abs/1506.01497
- Ronneberger, O., Fischer, P. & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*. https://arxiv.org/abs/1505.04597
- torchvision, *TorchVision Object Detection Finetuning Tutorial*. https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- TorchMetrics, `MeanAveragePrecision`. https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
- Maier-Hein, L. et al. (2024). "Metrics reloaded: recommendations for image analysis validation." *Nature Methods* 21, 195–212. https://doi.org/10.1038/s41592-023-02151-z
