# Image Preprocessing, Normalisation, Resizing and Data Augmentation

> **Folder:** 13_computer_vision · **Prev:** [cv_workflow.md](cv_workflow.md) · **Next:** [image_classification.md](image_classification.md) · **Related:** [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Convert images into model-ready tensors consistently, using the correct size, channels, scaling and normalisation. Apply data augmentation that teaches the model the invariances that really hold in deployment, without creating impossible images or label errors.

## 2. When to use it
Every image model. Augmentation is especially important with small datasets (hundreds to a few thousand images).

## 3. Conceptual foundation
- **Resizing:** models expect fixed sizes (or multiples of 32). Options: resize (it distorts the aspect ratio), resize + centre crop, pad to a square (letterbox, standard in detection), and tiling for small objects. Interpolation: area/bilinear for downscaling, nearest-neighbour for **masks** (never interpolate label values).
- **Scaling and normalisation:** convert uint8 → float in [0, 1], then normalise by channel mean/std. For pretrained ImageNet models use *their* statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). For models trained from scratch, use the training-set statistics.
- **Channels:** grayscale → 3 channels (repeat) for RGB-pretrained models, or adapt the first convolution. Multispectral → custom first layer.
- **Augmentation:** random transforms applied to **training** images only, each epoch differently. The model sees plausible variations and learns invariances.
| Augmentation | Teaches invariance to | Safe when | Dangerous when |
|---|---|---|---|
| Horizontal/vertical flips, 90° rotations | Orientation | Surfaces, microscopy, satellite | Text, digits, orientation-dependent defects |
| Small rotations/affine | Pose, camera angle | Most | Large angles on gauges/dials |
| Random resized crop | Scale, position | Objects remain inside the crop | Small defects cropped out (label becomes wrong!) |
| Brightness/contrast/gamma | Lighting | Lighting varies in deployment | Intensity *is* the signal (thermal temperature values) |
| Gaussian noise/blur | Sensor noise, focus | Mild amounts | Hides fine defects |
| Cutout/random erasing, mixup/cutmix | Robustness, regularisation | Classification with enough data | Small-object datasets |
- **Geometric augmentations must be applied identically to masks and boxes** (torchvision `transforms.v2` handles `tv_tensors.Mask` and `BoundingBoxes` jointly).
- **Test-time augmentation (TTA):** average the predictions over flips/crops at inference for a small, reliable gain.

## 4. Assumptions and limitations
- Augmentation encodes assumptions about deployment. Wrong invariances hurt (e.g. flipping asymmetric parts whose orientation matters).
- Normalisation statistics must come from training data (or the pretrained model). Computing them on all data is mild leakage, but stay consistent.
- Heavy augmentation slows convergence. Tune the augmentation strength like a hyperparameter.

## 5. Step-by-step procedure
1. **Decide the input size** from the object size and the model (keep the defects at ≥ ~5–10 px).
2. **Define the deterministic preprocessing** (shared by train/val/test): to tensor, resize/pad, convert channels, normalise.
3. **Define the training augmentation** from the deployment variations (ask: which orientations, lighting and positions occur?).
4. **Visualise augmented samples** (with masks/boxes overlaid) to verify that the labels remain valid.
5. **Wrap in a `Dataset`/`DataLoader`** with transforms applied on the fly.
6. **Ablation:** train with/without augmentation (or at different strengths), and compare the validation metrics.

## 6. Python implementation

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from cookbook_data import make_surface_defect_images

imgs, labels, masks = make_surface_defect_images(n=400)
classes = ["ok", "scratch", "spot", "crack"]; cls_idx = {c: i for i, c in enumerate(classes)}

# Deterministic preprocessing (all splits) + augmentation (train only). Images: uint8 (H, W) grayscale.
MEAN, STD = [imgs.mean() / 255] * 3, [imgs.std() / 255] * 3      # training-set statistics, repeated for 3 channels
base = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                   v2.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x)])   # gray → 3 channels
normalise = v2.Normalize(MEAN, STD)
train_aug = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
                        v2.RandomApply([v2.RandomRotation(degrees=(90, 90))], p=0.5),        # 90° rotations: surfaces have no orientation
                        v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                        v2.ColorJitter(brightness=0.2, contrast=0.2)])

class DefectDataset(Dataset):
    def __init__(self, imgs, labels, masks, augment=None):
        self.imgs, self.labels, self.masks, self.augment = imgs, labels, masks, augment
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        x = base(self.imgs[i][..., None])                         # (3, H, W) float in [0, 1]
        m = tv_tensors.Mask(torch.from_numpy(self.masks[i].astype(np.uint8))[None])
        if self.augment is not None:
            x, m = self.augment(x, m)                             # SAME geometric transform for image and mask
        return normalise(x), m[0].long(), cls_idx[self.labels[i]]

ds = DefectDataset(imgs, labels, masks, augment=train_aug)
x, m, yb = ds[1]
print("tensor:", x.shape, x.dtype, f"mean≈{x.mean():.2f}", "| mask:", m.shape, m.unique().tolist(), "| label:", yb)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
xb, mb, lb = next(iter(loader)); print("batch:", xb.shape, mb.shape, lb.shape)
```

```python
# ALWAYS visualise augmented samples with their masks to check that labels remain valid
i = int(np.where(labels == "scratch")[0][0])
fig, ax = plt.subplots(2, 6, figsize=(12, 4))
for k in range(6):
    xa, ma, _ = ds[i]
    ax[0, k].imshow(xa[0], cmap="gray"); ax[0, k].axis("off")
    ax[1, k].imshow(ma, cmap="Reds"); ax[1, k].axis("off")
ax[0, 0].set_title("augmented image"); ax[1, 0].set_title("augmented mask"); plt.tight_layout(); plt.show()

# A DANGEROUS augmentation for small defects: aggressive random crops can remove the defect → wrong label
risky = v2.RandomResizedCrop(size=64, scale=(0.2, 0.4))
lost = 0
for j in np.where(labels != "ok")[0][:100]:
    _, mc = risky(base(imgs[j][..., None]), tv_tensors.Mask(torch.from_numpy(masks[j].astype(np.uint8))[None]))
    lost += int(mc.sum() == 0)
print(f"defects completely cropped out by RandomResizedCrop(scale 0.2–0.4): {lost}/100 → those samples become mislabelled")
```

```python
# Resizing masks: nearest-neighbour only (bilinear would create fractional/invalid labels)
mask_t = tv_tensors.Mask(torch.from_numpy(masks[i].astype(np.uint8))[None])
print("values after nearest resize:", v2.Resize(32, interpolation=v2.InterpolationMode.NEAREST)(mask_t).unique().tolist())
```

## 7. Interpretation
- **Tensor checks:** (3, 64, 64) float, mean within about ±1 of 0 after normalisation (brightness jitter shifts individual images), mask values {0, 1}, batch shapes as expected. Checking the shapes and value ranges once prevents silent errors (e.g. images in [0, 255] fed to a model expecting normalised inputs).
- **Augmented grid:** the scratch moves, rotates and flips *together with its mask*. The labels remain valid. If a mask did not move with the image, the segmentation training would be corrupted.
- **RandomResizedCrop danger:** about 30% of the defects are cut out completely at crop scales of 0.2–0.4, and those images still carry the "defect" label. That is label noise created by preprocessing. Use milder crops, or defect-aware cropping.
- **Nearest-neighbour mask resizing** keeps the mask binary.

## 8. Decision framework
- **IF** the orientation is irrelevant (surfaces, microscopy, satellite) → flips + 90° rotations (cheap, very effective).
- **IF** the lighting varies in deployment → brightness/contrast jitter. **IF** the intensity is physically meaningful (thermal, X-ray density) → avoid photometric jitter.
- **IF** the objects are small → avoid aggressive crops/zoom-outs. Prefer tiling and mild affine transforms.
- **IF** the dataset is small (< 1000 images) → stronger augmentation + transfer learning; validate the augmentation strength.
- **IF** using pretrained models → their normalisation statistics and input sizes. For grayscale, repeat the channels.
- **IF** detection/segmentation → transforms that update boxes/masks jointly (`transforms.v2` + `tv_tensors`).

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Augmenting validation/test data | Noisy, unrepresentative evaluation | Augment the training data only (TTA is a separate, deliberate step) |
| Geometric transforms applied to images but not masks/boxes | Corrupted labels | Joint transforms (`v2` with `tv_tensors`) |
| Bilinear resizing of masks | Invalid label values | Nearest neighbour |
| Wrong normalisation statistics | Degraded pretrained performance | Use the model's statistics |
| Invariances that do not hold (flipping orientation-dependent parts) | Systematic errors | Derive the augmentations from the deployment conditions |
| Not visualising augmented samples | Silent label corruption | Always plot a grid |

## 10. Checklist
- [ ] Input size chosen from the object size; tiling if needed.
- [ ] Deterministic preprocessing shared by all splits.
- [ ] Augmentations derived from the deployment variations; applied to the training data only.
- [ ] Joint transforms for masks/boxes; nearest-neighbour for masks.
- [ ] Augmented samples visualised.
- [ ] Augmentation ablation on validation data.

## 11. References
- torchvision, *Transforms v2* (incl. `tv_tensors.Mask`, `BoundingBoxes`). https://pytorch.org/vision/stable/transforms.html
- Shorten, C. & Khoshgoftaar, T. M. (2019). "A survey on Image Data Augmentation for Deep Learning." *Journal of Big Data* 6, 60. https://doi.org/10.1186/s40537-019-0197-0
- Buslaev, A. et al. (2020). "Albumentations: Fast and Flexible Image Augmentations." *Information* 11(2), 125 (an alternative augmentation library). https://doi.org/10.3390/info11020125
- Zhang, H., Cisse, M., Dauphin, Y. N. & Lopez-Paz, D. (2018). "mixup: Beyond Empirical Risk Minimization." *ICLR*. https://arxiv.org/abs/1710.09412
