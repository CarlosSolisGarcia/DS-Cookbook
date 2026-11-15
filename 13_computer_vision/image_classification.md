# Image Classification: CNNs, Transfer Learning, Embeddings and Evaluation (PyTorch)

> **Folder:** 13_computer_vision · **Prev:** [image_preprocessing_augmentation.md](image_preprocessing_augmentation.md) · **Next:** [detection_and_segmentation.md](detection_and_segmentation.md) · **Related:** [classification evaluation](../04_classification/evaluation.md), [modeling_fundamentals.md](../00_methodology/modeling_fundamentals.md) (over/underfitting), [efficient_development.md](../20_advanced_methodologies/efficient_development.md)

---

## 1. Purpose
Train and evaluate image classifiers. Start with a small CNN from scratch (to understand the mechanics), then move to transfer learning with pretrained architectures (the practical default), and to image embeddings for few-label classification and similarity search. Evaluate per class and diagnose overfitting.

## 2. When to use it
Image-level decisions: defect/no defect, defect type, material class, quality grade, microscopy phenotype.

## 3. Conceptual foundation
- **Convolutional neural networks (CNNs):** stacks of convolution filters (local, weight-shared feature detectors), non-linearities, normalisation and pooling. Early layers detect edges and textures, deeper layers detect parts and objects. Translation equivariance and parameter sharing make them data-efficient compared with fully connected networks.
- **Pretrained architectures** (torchvision): ResNet-18/50, EfficientNet, ConvNeXt, ViT. They were trained on ImageNet (1.28M images, 1000 classes) and their features transfer to most visual domains, including industrial ones.
- **Transfer-learning strategies:**
  1. **Feature extraction:** freeze the backbone and train only a new head (logistic regression on embeddings, or a linear layer). Fast; good for small data.
  2. **Fine-tuning:** unfreeze some or all layers with a small learning rate (often after training the head first). Best accuracy with moderate data.
- **Training loop essentials:** cross-entropy loss (class weights for imbalance), the AdamW/SGD optimiser, a learning-rate schedule (cosine/one-cycle), mini-batches, `model.train()`/`model.eval()`, gradients disabled for evaluation, early stopping on validation, and seeds.
- **Over/underfitting in CNNs:** watch the train vs validation loss/metric curves. Remedies: augmentation, weight decay, dropout, smaller models, more data, pretrained features.

## 4. Assumptions and limitations
- Training from scratch needs much more data than fine-tuning. With < ~10⁴ images, pretrained models almost always win.
- CPU training is fine for tiny CNNs and small images (as here). For pretrained models at 224 px, use a GPU (an RTX 3060 6 GB handles ResNet-50 with batch ~32 and mixed precision) or Colab.
- Accuracy hides per-class failures. Always report per-class recall and the confusion matrix.

## 5. Step-by-step procedure
1. **Split** (stratified; grouped if the images are related); fixed seeds.
2. **Baselines:** majority class; a simple feature baseline ([cv_workflow.md](cv_workflow.md)).
3. **Small CNN from scratch** (optional for small/simple data, and for learning): train with augmentation; plot the curves.
4. **Transfer learning:** a pretrained ResNet-18/EfficientNet-B0: (a) frozen features + logistic; (b) fine-tuning.
5. **Evaluate on the test set:** per-class report, confusion matrix, misclassified examples.
6. **Diagnose:** the learning curves (over/underfitting); errors by acquisition condition (camera, batch).
7. **Embeddings for search and few-label tasks:** nearest neighbours in the embedding space.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from cookbook_data import make_surface_defect_images

torch.manual_seed(0); np.random.seed(0)
imgs, labels, _ = make_surface_defect_images(n=1600)
classes = ["ok", "scratch", "spot", "crack"]; y = np.array([classes.index(l) for l in labels])
X = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1) / 255.0           # (N, 1, 64, 64)
mean, std = X.mean().item(), X.std().item(); X = (X - mean) / std
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)
idx_tr, idx_va = train_test_split(idx_tr, test_size=0.2, stratify=y[idx_tr], random_state=0)
Xtr, Xva, Xte = X[idx_tr], X[idx_va], X[idx_te]
ytr, yva, yte = (torch.tensor(y[i]) for i in (idx_tr, idx_va, idx_te))

augment = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05))])

class SmallCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))   # 64 → 8 px
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, n_classes))
    def embed(self, x):
        return self.pool(self.features(x)).flatten(1)
    def forward(self, x):
        return self.head(self.embed(x))

def run_training(model, epochs=12, lr=3e-3, use_aug=True, batch=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(Xtr) / batch)))
    loss_fn, hist, best = nn.CrossEntropyLoss(), [], (np.inf, None)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train(); tl = []
        for xb, yb in loader:
            if use_aug:
                xb = torch.stack([augment(x) for x in xb])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); sched.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item(); vacc = (model(Xva).argmax(1) == yva).float().mean().item()
        hist.append({"epoch": ep + 1, "train_loss": np.mean(tl), "val_loss": vl, "val_acc": vacc})
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])                       # keep the best validation epoch (early stopping)
    return pd.DataFrame(hist)

cnn = SmallCNN()
hist = run_training(cnn)
print(hist.round(3).to_string(index=False))
```

```python
# Evaluation on the untouched test set: per-class report + confusion matrix + misclassified examples
cnn.eval()
with torch.no_grad():
    logits = cnn(Xte); pred = logits.argmax(1).numpy(); prob = logits.softmax(1).numpy()
print(classification_report(yte.numpy(), pred, target_names=classes, digits=3))
print(pd.DataFrame(confusion_matrix(yte.numpy(), pred), index=classes, columns=classes))

wrong = np.where(pred != yte.numpy())[0][:8]
fig, ax = plt.subplots(1, max(len(wrong), 1), figsize=(2 * max(len(wrong), 1), 2.4))
for a, i in zip(np.atleast_1d(ax), wrong):
    a.imshow(imgs[idx_te[i]], cmap="gray"); a.axis("off")
    a.set_title(f"true {classes[yte[i]]}\npred {classes[pred[i]]} ({prob[i].max():.2f})", fontsize=7)
plt.tight_layout(); plt.show()
```

```python
# Over/underfitting check: the same CNN WITHOUT augmentation on a small training subset
torch.manual_seed(0)
keep = np.random.default_rng(0).choice(len(Xtr), 200, replace=False)
Xtr_full, ytr_full = Xtr, ytr
Xtr, ytr = Xtr_full[keep], ytr_full[keep]
h_noaug = run_training(SmallCNN(), epochs=15, use_aug=False)
torch.manual_seed(0)
h_aug = run_training(SmallCNN(), epochs=15, use_aug=True)
Xtr, ytr = Xtr_full, ytr_full                                            # restore
print(pd.DataFrame({"no-aug train_loss": h_noaug["train_loss"], "no-aug val_loss": h_noaug["val_loss"],
                    "aug train_loss": h_aug["train_loss"], "aug val_loss": h_aug["val_loss"]}).iloc[[0, 4, 9, 14]].round(3))
```

```python
# Image embeddings: penultimate-layer features → nearest neighbours and a few-label classifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
with torch.no_grad():
    E_tr, E_te = cnn.embed(Xtr).numpy(), cnn.embed(Xte).numpy()
nn_index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(E_tr)
_, nbrs = nn_index.kneighbors(E_te[:1])
print("query class:", classes[yte[0]], "| neighbour classes:", [classes[ytr[j]] for j in nbrs[0]])
few = np.random.default_rng(0).choice(len(E_tr), 40, replace=False)
lr_few = LogisticRegression(max_iter=2000).fit(E_tr[few], ytr.numpy()[few])
print("logistic on 40 labelled embeddings, test accuracy:", round(lr_few.score(E_te, yte.numpy()), 3))
```

Transfer learning with a pretrained ResNet-18. This needs the ImageNet weights to be downloaded once, and a GPU or Colab is recommended:

```python
# not-run (downloads ImageNet weights ~45 MB; GPU recommended at 224 px)
from torchvision.models import ResNet18_Weights, resnet18
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(classes))           # new head
preprocess = weights.transforms()                                   # resize 256 → crop 224, ImageNet normalisation
to_rgb = lambda x: x.expand(3, -1, -1)                              # grayscale → 3 channels (before `preprocess`)

# Stage 1: freeze the backbone, train the head (fast, robust with little data)
for p in model.parameters(): p.requires_grad = False
for p in model.fc.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
# ... train a few epochs ...
# Stage 2: unfreeze everything with a lower learning rate (discriminative LRs are common)
for p in model.parameters(): p.requires_grad = True
opt = torch.optim.AdamW([{"params": model.fc.parameters(), "lr": 1e-3},
                         {"params": [p for n, p in model.named_parameters() if not n.startswith("fc")], "lr": 1e-4}],
                        weight_decay=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
scaler = torch.amp.GradScaler(enabled=device == "cuda")             # mixed precision on GPU
# with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"): loss = ...
```

## 7. Interpretation
- **Training history:** the validation loss is erratic in the first epochs (batch-norm statistics and the one-cycle warm-up), then decreases together with the training loss to a validation accuracy of ≈ 0.92. The best epoch is restored (early stopping). The validation loss is still falling at epoch 12, so a longer training would probably help.
- **Per-class report and confusion matrix:** every error in this run is a defect predicted as **"ok"**. The "ok" class has recall 1.0 but precision ≈ 0.81, and spots/scratches have recall ≈ 0.77–0.83. In inspection, these are the costly errors (missed defects). Remedies: more epochs or a pretrained backbone, class weights or a lower decision threshold for "ok" (accept more false rejects), and checking whether the missed defects are faint or at the image border.
- **Misclassified examples:** look at them. Typical causes are faint defects, defects at the border, strong illumination gradients and label errors.
- **Augmentation ablation on 200 images:** with only 15 epochs both models are still **underfitting** (train loss ≈ 1.0), so they look alike. That is itself a lesson: diagnose underfitting first. In a longer run (60 epochs, tried while writing this recipe) the model without augmentation reached train/val loss ≈ 0.32/0.49 (a generalisation gap), and the augmented model ≈ 0.51/0.52 (no gap), with similar validation accuracy (≈ 0.85). Augmentation reliably narrows the gap. Whether it also raises the validation accuracy depends on the data and the training length. Measure it.
- **Embeddings:** neighbours mostly share the query's class, so the embedding space is semantically organised. Logistic regression on only 40 labelled embeddings reaches decent accuracy, which illustrates why pretrained (or self-trained) embeddings are powerful for few-label problems. With ImageNet-pretrained embeddings this works even without training on your data.

## 8. Decision framework
| Situation | Approach |
|---|---|
| < ~500 labelled images | Pretrained backbone, frozen features + logistic/linear head; strong augmentation |
| 500–10 000 images | Fine-tune a pretrained ResNet/EfficientNet (head first, then all layers with a low LR) |
| > 10 000 images, a domain far from ImageNet | Fine-tune fully; consider domain-pretrained or self-supervised backbones |
| Small, simple images (like this synthetic set) | A small CNN from scratch may suffice; compare with a pretrained one |
| Need similarity search / dedup | Embeddings + cosine kNN |
| Grayscale/thermal/multispectral | Repeat channels, or adapt the first conv; check that the normalisation matches the modality |
| Imbalanced classes | Class weights / weighted sampling; per-class metrics; threshold tuning on probabilities |
| CPU only | Small images, small models, frozen features; Colab for fine-tuning at 224 px |

- **IF** train ≫ val accuracy → more augmentation, weight decay, a smaller model, or more data. **IF** both are low → a bigger/pretrained model, a higher resolution, a longer training.
- **IF** errors concentrate on a camera/batch → domain shift. Augment for that variation, or collect data from it.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Forgetting `model.eval()` / `torch.no_grad()` at evaluation | Dropout/batch-norm in training mode; wrong metrics; memory waste | Always switch modes |
| Augmenting the validation/test data | Noisy evaluation | Training data only |
| Early stopping on the test set | Optimistic metrics | A separate validation split |
| A high LR when fine-tuning a pretrained model | Destroys pretrained features | Low LR (1e-4) for the backbone |
| Only accuracy | Hidden per-class failures | Per-class report, confusion matrix |
| Not looking at the errors | Missing label errors/shortcuts | Show misclassified images |

## 10. Checklist
- [ ] Stratified/grouped split; seeds fixed.
- [ ] Baselines (majority, simple features).
- [ ] CNN or pretrained backbone with augmentation; early stopping on validation.
- [ ] Learning curves inspected (over/underfitting).
- [ ] Test evaluation: per-class report, confusion matrix, error gallery.
- [ ] Embeddings for kNN/few-label use where relevant.
- [ ] Hardware plan (CPU/GPU/Colab, mixed precision).

## 11. References
- He, K., Zhang, X., Ren, S. & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *CVPR*. https://arxiv.org/abs/1512.03385
- Yosinski, J., Clune, J., Bengio, Y. & Lipson, H. (2014). "How transferable are features in deep neural networks?" *NeurIPS*. https://arxiv.org/abs/1411.1792
- Kornblith, S., Shlens, J. & Le, Q. V. (2019). "Do Better ImageNet Models Transfer Better?" *CVPR*. https://arxiv.org/abs/1805.08974
- PyTorch tutorial, *Transfer Learning for Computer Vision*. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- torchvision, *Models and pre-trained weights*. https://pytorch.org/vision/stable/models.html
- Stanford CS231n notes (CNNs, training). https://cs231n.github.io/
