# Introduction to Raster Data and Remote Sensing (Satellite Imagery)

> **Folder:** 17_geospatial_data · **Prev:** [geospatial_workflow.md](geospatial_workflow.md) · **Next:** [scientific_ml_workflow.md](../18_scientific_machine_learning/scientific_ml_workflow.md) · **Related:** [13_computer_vision](../13_computer_vision/cv_workflow.md) (CNNs on imagery), [pca.md](../11_dimensionality_reduction/pca.md)

---

## 1. Purpose
Understand raster data (bands, resolution, georeferencing, nodata), compute spectral indices such as NDVI, extract raster values at points, and set up pixel- or patch-level classification with spatially honest validation. This is an introduction for environmental and energy applications (land cover, vegetation stress, solar-panel detection, flooding).

## 2. When to use it
Satellite or aerial imagery (Sentinel-2, Landsat, drone orthophotos), gridded climate/weather data, digital elevation models, any georeferenced raster.

## 3. Conceptual foundation
- **A raster** is a grid of cells, with one value per band per cell, plus an **affine transform** (row/col → x/y in the CRS), a **CRS**, a resolution (e.g. 10 m for Sentinel-2 visible/NIR bands), and a **nodata** value.
- **Multispectral bands:** reflectance in wavelength ranges (blue, green, red, near-infrared NIR, short-wave infrared SWIR). Materials have characteristic spectral signatures: vegetation reflects strongly in the NIR and absorbs red; water absorbs NIR.
- **Spectral indices:** normalised band ratios that reduce illumination effects:
  - NDVI = (NIR − Red) / (NIR + Red): vegetation vigour (≈ 0.6–0.9 dense vegetation; ≈ 0 bare soil; < 0 water).
  - NDWI = (Green − NIR) / (Green + NIR): water.
- **Clouds and shadows** must be masked (quality bands / cloud masks). **Temporal composites** (the median over a period) reduce gaps.
- **Spatial autocorrelation is extreme in rasters:** neighbouring pixels are nearly identical. Random pixel splits are severely optimistic, so validate with **spatial blocks/tiles**.

## 4. Assumptions and limitations
- Real satellite data need preprocessing levels (top-of-atmosphere vs surface reflectance), resampling between band resolutions, and co-registration. Use analysis-ready products where possible.
- Labels (ground truth polygons) are often few and spatially clustered. Validation must account for this.
- Large scenes need windowed/tiled processing (rasterio windows, xarray/dask).

## 5. Step-by-step procedure
1. **Load the raster:** inspect the CRS, transform, resolution, band descriptions, dtype and nodata (`rasterio.open(...).profile`).
2. **Mask nodata/clouds;** scale to reflectance if needed (scale factors from the metadata).
3. **Compute indices** (NDVI, NDWI) and visualise them.
4. **Sample the raster at labelled points/polygons** (CRS-aligned) to build a training table.
5. **Classify pixels** (GBM/RF on bands + indices) or patches (CNN, [13_computer_vision](../13_computer_vision/image_classification.md)).
6. **Validate with spatial blocks;** compare with random pixel CV to see the optimism.
7. **Predict a map;** post-process (majority filter); estimate the areas with uncertainty.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Synthetic 4-band scene (blue, green, red, NIR) at 10 m: water, vegetation, bare soil, built-up patches
rng = np.random.default_rng(0)
H = W = 200
def blobs(seed, sigma=12, frac=0.3):
    """Smooth random field thresholded so that `frac` of the scene is covered (patchy, spatially coherent)."""
    f = gaussian_filter(np.random.default_rng(seed).normal(size=(H, W)), sigma)
    return f > np.quantile(f, 1 - frac)
land = np.full((H, W), 2)                                         # 2 = bare soil
land[blobs(1, frac=0.40)] = 1                                     # vegetation
land[blobs(2, sigma=20, frac=0.12)] = 0                           # water
land[blobs(3, sigma=6, frac=0.10)] = 3                            # built-up
signature = {0: [0.06, 0.07, 0.05, 0.02], 1: [0.04, 0.08, 0.05, 0.40], 2: [0.10, 0.14, 0.18, 0.25], 3: [0.14, 0.15, 0.16, 0.18]}
bands = np.stack([np.vectorize(lambda c: signature[c][b])(land) for b in range(4)]).astype(np.float32)
bands += rng.normal(0, 0.015, bands.shape).astype(np.float32)     # sensor noise
bands *= gaussian_filter(rng.normal(1, 0.08, (H, W)), 15)[None]   # illumination/atmospheric variation
cloud = blobs(4, sigma=10, frac=0.08)                            # a cloud mask (8% of the scene)
bands[:, cloud] = np.nan
blue, green, red, nir = bands

ndvi = (nir - red) / (nir + red)
ndwi = (green - nir) / (green + nir)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].imshow(np.clip(np.nan_to_num(np.dstack([red, green, blue]) / 0.25), 0, 1)); ax[0].set_title("true colour (clouds = black)")
im = ax[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.5, vmax=0.9); ax[1].set_title("NDVI"); plt.colorbar(im, ax=ax[1])
ax[2].imshow(land, cmap="tab10", vmin=0, vmax=9); ax[2].set_title("reference land cover")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
print(pd.DataFrame({"class": land.ravel(), "ndvi": ndvi.ravel()}).groupby("class")["ndvi"].median().round(2))
```

```python
# Georeferencing a NumPy raster and writing/reading it with rasterio (in memory)
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
transform = from_origin(430000, 4480000, 10, 10)                 # top-left x, y (EPSG:25830) and 10 m pixels
profile = dict(driver="GTiff", height=H, width=W, count=4, dtype="float32", crs="EPSG:25830", transform=transform, nodata=np.nan)
with MemoryFile() as mem:
    with mem.open(**profile) as dst:
        dst.write(bands)
    with mem.open() as src:
        print(src.profile["crs"], src.res, src.bounds)
        pts = [(430000 + 10 * c + 5, 4480000 - 10 * r - 5) for r, c in [(10, 10), (100, 150)]]
        print("sampled band values at points:", [np.round(v, 3).tolist() for v in src.sample(pts)])   # value extraction at coordinates
```

```python
# Pixel classification: random-pixel CV vs SPATIAL BLOCK CV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
valid = ~np.isnan(bands).any(0)
rows, cols = np.nonzero(valid)
sample = rng.choice(len(rows), 6000, replace=False); r, c = rows[sample], cols[sample]
Xp = np.column_stack([bands[:, r, c].T, ndvi[r, c], ndwi[r, c]]); yp = land[r, c]
block = (r // 50) * 4 + (c // 50)                                  # 16 spatial blocks of 500 m × 500 m
clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
acc_rand = cross_val_score(clf, Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy")
acc_blk = cross_val_score(clf, Xp, yp, cv=GroupKFold(4), groups=block, scoring="balanced_accuracy")
print(f"balanced accuracy — random pixels: {acc_rand.mean():.3f}   spatial blocks: {acc_blk.mean():.3f} ± {acc_blk.std():.3f}")

# Predict the full map (cloud pixels stay unclassified) and estimate the areas
model = clf.fit(Xp, yp)
Xall = np.column_stack([bands[:, rows, cols].T, ndvi[rows, cols], ndwi[rows, cols]])
pred_map = np.full((H, W), -1); pred_map[rows, cols] = model.predict(Xall)
area_ha = pd.Series(pred_map[pred_map >= 0]).value_counts().sort_index() * 100 / 10_000   # 10 m pixel = 100 m²
print("predicted area (ha) by class:", area_ha.round(1).to_dict(), "| cloud-masked share:", round(cloud.mean(), 3))
```

## 7. Interpretation
- **NDVI medians by class:** high for vegetation, near 0 for bare soil/built-up, negative for water. The index separates vegetation and water well on its own, while built-up vs bare soil needs the full spectral information.
- **Random vs spatial-block accuracy:** both ≈ 0.997 in this synthetic scene, because every class has one distinct spectral signature everywhere in the scene, so nothing regional has to be learned. The block estimate is still the honest one, and its ± shows the variation between areas. In real imagery (mixed pixels, regional differences in soils and crops) the gap is often large. Random-pixel validation inflates the accuracy, because training and test pixels come from the same fields.
- **Area estimates:** pixel counts × the pixel area give the class areas. Report them with the classification uncertainty, and note the masked (cloud) fraction. Unbiased area estimation from a classified map requires reference sampling (e.g. Olofsson et al., 2014).

## 8. Decision framework
- **IF** the question concerns vegetation or water → start from spectral indices (NDVI/NDWI) and simple thresholds/regressions. That is a strong, interpretable baseline.
- **IF** the classes differ in spectral signature → pixel-wise GBM/RF on the bands + indices. **IF** they differ in *texture/shape* (buildings, solar panels, roads) → patch-based CNNs or segmentation ([13_computer_vision](../13_computer_vision/detection_and_segmentation.md)).
- **IF** labels come from a few polygons → spatial block (or polygon-grouped) CV; never random pixel splits.
- **IF** clouds/gaps → masks plus temporal compositing; report the masked fraction.
- **IF** the scenes are large → windowed reading (rasterio windows), tiling, xarray/dask.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Random pixel train/test split | Grossly optimistic accuracy | Spatial blocks / polygon groups |
| Ignoring nodata/cloud values | Garbage features (zeros or NaN as values) | Masks; nodata handling |
| Mixing band resolutions without resampling | Misaligned pixels | Resample to a common grid |
| Sampling rasters with points in another CRS | Wrong values | Reproject the points to the raster CRS |
| Area = pixel count without accuracy adjustment | Biased area estimates | Reference-sample-based estimation |

## 10. Checklist
- [ ] Raster profile inspected (CRS, transform, resolution, nodata, scale factors).
- [ ] Cloud/nodata masks applied.
- [ ] Indices computed and visualised.
- [ ] Training samples extracted in a consistent CRS.
- [ ] Spatial block validation vs random reported.
- [ ] Map predicted; the areas reported with caveats.

## 11. References
- rasterio documentation (reading, windows, sampling, profiles). https://rasterio.readthedocs.io/
- Copernicus Sentinel-2 user guide (bands, resolutions, processing levels). https://sentiwiki.copernicus.eu/web/s2-mission
- Tucker, C. J. (1979). "Red and photographic infrared linear combinations for monitoring vegetation." *Remote Sensing of Environment* 8(2), 127–150 (NDVI).
- Olofsson, P. et al. (2014). "Good practices for estimating area and assessing accuracy of land change." *Remote Sensing of Environment* 148, 42–57. https://doi.org/10.1016/j.rse.2014.02.015
- Rey, S. J. et al. *Geographic Data Science with Python*, chapters on raster data. https://geographicdata.science/book/
