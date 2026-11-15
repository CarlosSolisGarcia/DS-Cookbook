# Computer Vision Workflow: Task Types, Image Data Exploration and Formats

> **Folder:** 13_computer_vision · **Start here for image data.** · **Next:** [image_preprocessing_augmentation.md](image_preprocessing_augmentation.md) · **Related:** [classification_workflow.md](../04_classification/classification_workflow.md), [efficient_development.md](../20_advanced_methodologies/efficient_development.md)

---

## 1. Purpose
Frame image problems correctly (classification, detection, segmentation), understand the annotation and data requirements of each, explore image datasets (formats, sizes, channels, class balance, duplicates, acquisition conditions), and choose a sensible first baseline. The progression goes from a simple classifier to detection and segmentation.

## 2. When to use it
Any project with images: visual inspection of surfaces and welds, microscopy, thermal images of equipment, satellite/aerial images ([remote_sensing_intro.md](../17_geospatial_data/remote_sensing_intro.md)), documents, camera monitoring.

## 3. Conceptual foundation

### 3.1 Task types and annotation requirements
| Task | Output | Annotation needed | Effort | Typical metric | First baseline |
|---|---|---|---|---|---|
| **Image classification** | One label (or several) per image | Image-level labels | Low | Accuracy / macro-F1, per-class recall, AUC | Pretrained CNN fine-tuned; or embeddings + logistic |
| **Object detection** | Boxes + class + score per object | Bounding boxes | Medium | mAP@[.5:.95], AP@0.5, recall at a confidence threshold | Pretrained Faster R-CNN / RetinaNet / YOLO fine-tuned |
| **Semantic segmentation** | A class per pixel | Pixel masks | High | IoU (Jaccard), Dice/F1 per class, pixel accuracy | U-Net / DeepLab with a pretrained encoder |
| **Instance segmentation** | A mask per object instance | Instance masks | Highest | Mask mAP | Mask R-CNN |
| **Anomaly detection (images)** | Normal vs abnormal (± a localisation map) | Mostly normal images | Low (normal only) | AUROC, PRO | Pretrained-feature kNN/PatchCore-type methods; autoencoders |

**Choose the simplest task that supports the decision.** If the decision is "reject the part", classification may suffice, even when detection would be "nicer". Detection or segmentation is needed when location, count or size matters (e.g. crack length → severity).

### 3.2 Image data basics
- **Format:** the array shape (H, W) for grayscale, (H, W, 3) for RGB in PIL/NumPy/OpenCV, and **(C, H, W)** in PyTorch tensors. OpenCV loads images as **BGR**, not RGB.
- **Bit depth:** uint8 (0–255) for most photos. 12/16-bit for scientific, thermal and medical images; do not blindly divide by 255.
- **Channels:** RGB, grayscale, multispectral (satellite: 4–13 bands), thermal (1 channel, temperature-calibrated).
- **Resolution vs object size:** the defect must remain visible after resizing. A 2-pixel crack in a 2000-px image disappears at 224 px. Use tiles/crops instead of downscaling.
- **Acquisition conditions:** lighting, camera, lens, angle and background. Models latch onto them (shortcut learning). Record them as metadata.

## 4. Assumptions and limitations
- Pretrained ImageNet models assume natural RGB photos. They transfer surprisingly well to industrial images, but grayscale/multispectral inputs need adaptation (repeat the channels, or replace the first layer).
- Labels from images are often subjective (borderline defects). Inter-annotator agreement limits the achievable accuracy.
- Near-duplicate images (consecutive video frames, the same part photographed twice) leak across splits unless they are grouped.

## 5. Step-by-step procedure
1. **Frame:** the decision, the task type, the unit (image, tile, object), the metric, the error costs (a missed defect vs a false reject).
2. **Inventory:** the number of images, classes, formats, sizes, bit depths, channels, acquisition metadata (camera, line, batch, date).
3. **Visual inspection:** grids of images per class; look at the borderline cases; check the label quality with an expert.
4. **Statistics:** the intensity distribution per class/camera (shortcut risk), image sizes, object sizes relative to the image.
5. **Duplicates:** perceptual hashes / embedding similarity; group duplicates for splitting.
6. **Split:** stratified, and grouped by part/batch/camera/patient/session when images are related; temporal if the production changes.
7. **Baseline:** (a) a simple feature baseline (intensity statistics/HOG + logistic) to detect shortcuts; (b) a pretrained CNN (fine-tuned) ([image_classification.md](image_classification.md)).

## 6. Python implementation: exploring an image dataset

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook_data import make_surface_defect_images

imgs, labels, masks = make_surface_defect_images()
print("array:", imgs.shape, imgs.dtype, "| value range:", imgs.min(), imgs.max())
print(pd.Series(labels).value_counts())

# Image grid per class (always LOOK at the data)
classes = ["ok", "scratch", "spot", "crack"]
fig, ax = plt.subplots(4, 6, figsize=(10, 7))
for r, c in enumerate(classes):
    idx = np.where(labels == c)[0][:6]
    for k, i in enumerate(idx):
        ax[r, k].imshow(imgs[i], cmap="gray", vmin=0, vmax=255); ax[r, k].axis("off")
    ax[r, 0].set_title(c, loc="left", fontsize=9)
plt.tight_layout(); plt.show()
```

```python
# Per-class intensity statistics: if these alone separate the classes, beware of shortcuts
stats = pd.DataFrame({"label": labels, "mean": imgs.mean((1, 2)), "std": imgs.std((1, 2)),
                      "p01": np.percentile(imgs, 1, axis=(1, 2)), "p99": np.percentile(imgs, 99, axis=(1, 2)),
                      "defect_area_%": 100 * masks.mean((1, 2))})
print(stats.groupby("label").median().round(2))

# How big are the defects relative to the image? (decides the resolution / tiling)
print(stats[stats["label"] != "ok"].groupby("label")["defect_area_%"].describe()[["min", "50%", "max"]].round(2))
```

```python
# Simple shortcut-detecting baseline: global statistics + logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X_stats = stats[["mean", "std", "p01", "p99"]].to_numpy()
s = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X_stats, labels,
                    cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1_macro")
print(f"global-statistics baseline macro-F1: {s.mean():.3f}")

# Near-duplicate check with a tiny perceptual hash (average hash on an 8×8 thumbnail)
import cv2
def ahash(img):
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()
H = np.array([ahash(im) for im in imgs])
d = (H[:200, None, :] != H[None, :, :]).sum(-1)                 # Hamming distances (first 200 vs all)
np.fill_diagonal(d[:, :200], 99)
print("pairs with Hamming distance ≤ 3 (near-duplicates):", int((d <= 3).sum()))
```

## 7. Interpretation
- **The grid:** scratches are bright thin lines, spots are dark blobs, and cracks are dark jagged lines. Illumination gradients and texture vary between images. A good classifier must ignore these.
- **Per-class statistics:** spots and cracks lower the 1st percentile, and scratches raise the 99th. The global-statistics baseline reaches macro-F1 ≈ 0.53, well above chance (0.25) and far from good. That is fine here (the defects *are* intensity anomalies). In real data, a strong statistics-only baseline can reveal **shortcuts** (e.g. defective parts photographed on a different line or under different lighting).
- **Defect area:** about 0.1–6% of the pixels (medians ≈ 0.8% for cracks, 1.2% for scratches, 2.2% for spots). Downscaling the 64-px images further would destroy thin scratches and cracks. Keep the resolution, or tile.
- **Near-duplicates:** none in this synthetic set. In real datasets (video frames, re-shoots) group them before splitting.

## 8. Decision framework
**IF the dataset contains images, which task?**
- The decision depends only on *whether* something is present → **classification** (image-level labels; cheapest annotation). Baseline: a pretrained ResNet/EfficientNet, fine-tuned, or frozen embeddings + logistic regression.
- Location/count matters, and the objects are separable → **detection** (box annotation). Baseline: a pretrained Faster R-CNN (torchvision) or YOLO, fine-tuned.
- Size/shape/area matters (crack length, corrosion area), or the objects are amorphous → **segmentation** (masks). Baseline: U-Net with a pretrained encoder.
- Defective examples are rare or unknown → **image anomaly detection**, trained on normal images only.

**Data and resolution:** IF the objects are small relative to the image → tiles/crops at native resolution. IF < ~100 labelled images per class → transfer learning with frozen features, heavy augmentation and simple heads. IF there are multispectral/thermal channels → adapt the first layer, or use per-band features.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Choosing segmentation when classification suffices | Expensive annotation without benefit | Start from the decision |
| Downscaling until the defects vanish | Impossible task | Check the object size vs the input size |
| BGR/RGB confusion (OpenCV vs PIL) | Degraded pretrained performance | Convert explicitly |
| Dividing 16-bit images by 255 | Saturated inputs | Scale by the actual bit depth / percentiles |
| Random splits with related images | Inflated metrics | Group by part/batch/session |
| Not looking at images | Shortcuts and label errors missed | Grids per class; expert review |

## 10. Checklist
- [ ] Decision and task type (classification/detection/segmentation/anomaly) chosen; annotation cost considered.
- [ ] Formats, bit depth, channels, sizes inventoried; acquisition metadata collected.
- [ ] Image grids reviewed with an expert; label quality checked.
- [ ] Object size vs input resolution checked.
- [ ] Duplicates/related images grouped; split designed.
- [ ] Statistics baseline (shortcut check) + pretrained baseline planned.

## 11. References
- Szeliski, R. (2022). *Computer Vision: Algorithms and Applications*, 2nd ed. Free online: https://szeliski.org/Book/
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition*, course notes. https://cs231n.github.io/
- Geirhos, R. et al. (2020). "Shortcut learning in deep neural networks." *Nature Machine Intelligence* 2, 665–673. https://doi.org/10.1038/s42256-020-00257-z
- torchvision documentation (models, transforms, detection). https://pytorch.org/vision/stable/
- Bergmann, P., Fauser, M., Sattlegger, D. & Steger, C. (2019). "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *CVPR*. https://doi.org/10.1109/CVPR.2019.00982
