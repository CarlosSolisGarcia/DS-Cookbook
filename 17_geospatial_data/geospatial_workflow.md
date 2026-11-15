# Geospatial Data Science Workflow: CRS, Vector Data, Spatial Joins, Features and Spatial Validation

> **Folder:** 17_geospatial_data · **Start here for location data.** · **Next:** [remote_sensing_intro.md](remote_sensing_intro.md) · **Related:** [clustering_algorithms.md](../09_unsupervised_learning/clustering_algorithms.md) (DBSCAN on coordinates), [validation_reference.md](../08_quick_references/validation_reference.md), [data_leakage.md](../00_methodology/data_leakage.md)

---

## 1. Purpose
Work correctly with location data: coordinate reference systems, vector formats, spatial joins and aggregation, distance/proximity features, maps, spatial clustering. Above all, validate models without **spatial leakage**, the main source of over-optimism in geospatial ML.

## 2. When to use it
Data with coordinates or geometries: monitoring stations (air quality, weather, groundwater), wells, wind turbines and solar parks, pipelines and grids, assets and incidents, administrative areas, satellite-derived layers.

## 3. Conceptual foundation

### 3.1 Coordinate reference systems (CRS)
- **Geographic CRS** (e.g. WGS84, **EPSG:4326**): longitude/latitude in **degrees**. Distances and areas computed directly on degrees are wrong (1° of longitude shrinks with latitude).
- **Projected CRS** (e.g. UTM zones, **EPSG:25830** = ETRS89 / UTM 30N for Spain, or national grids): coordinates in **metres**. Use one for distances, buffers and areas.
- **Rule:** store in WGS84 if needed for interchange; compute in a suitable projected CRS; always check `gdf.crs` before any geometric operation. GeoPandas `to_crs()` reprojects.

### 3.2 Data models and formats
| Model | Content | Formats | Python |
|---|---|---|---|
| **Vector** | Points, lines, polygons + attributes | GeoPackage (`.gpkg`, recommended), Shapefile (legacy), GeoJSON, GeoParquet | GeoPandas / Shapely |
| **Raster** | Grids of cell values (bands) | GeoTIFF / COG, NetCDF (multi-dimensional) | rasterio, xarray/rioxarray |

### 3.3 Core operations
- **Spatial join** (`gpd.sjoin`): attach the attributes of the polygon containing each point (`predicate="within"`), or of intersecting features. **Nearest join** (`sjoin_nearest`): attach the nearest feature and its distance.
- **Aggregation:** count/mean of points per polygon (`sjoin` + `groupby`), or `dissolve`.
- **Proximity features:** the distance to the nearest road/industry/river, the counts within buffers (r = 500 m, 1 km…), kernel-weighted sums (e.g. Σ emission · exp(−d/ℓ)).
- **Spatial autocorrelation:** nearby observations are similar (Tobler's first law). This creates both signal (spatial features, kriging) and **leakage risk** in validation.

### 3.4 Spatial leakage and validation
With random K-fold, a test station often has a training station a few hundred metres away with almost the same value. The model is rewarded for **interpolating** between neighbours, not for predicting at genuinely new locations. If the model will be used in **new areas**, use **spatial block CV** (groups = spatial blocks or clusters), with block sizes comparable to the autocorrelation range.

## 4. Assumptions and limitations
- The precision of the coordinates and the CRS metadata may be wrong (swapped lat/lon, missing CRS). Plot the data on a map early.
- Spatial block CV estimates performance for extrapolation. It can be pessimistic if the deployment is pure interpolation within the sampled area. Choose the scheme from the use case.
- Large vector joins need spatial indexes (GeoPandas uses them automatically). Very large rasters need windowed reading.

## 5. Step-by-step procedure
1. **Load and check the CRS;** plot the points/polygons; fix swapped axes or missing CRS.
2. **Reproject** to a projected CRS for metric operations.
3. **Join context:** spatial joins with administrative areas/zones; nearest joins with infrastructure.
4. **Engineer features:** distances, buffer counts, kernel-weighted sums, elevation, land-use shares.
5. **Explore the spatial structure:** maps of the target; a variogram-like analysis (the similarity of pairs vs distance) to estimate the autocorrelation range.
6. **Validate** with random CV *and* spatial block CV. Report both and explain which matches the deployment.
7. **Spatial clustering** (if useful): DBSCAN/HDBSCAN on projected coordinates with a physical `eps` in metres.
8. **Map the predictions and residuals;** look for spatial patterns in the residuals.

## 6. Python implementation

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point, box

from cookbook_data import make_air_quality_stations

st, roads, industry = make_air_quality_stations()
gdf = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st["x"], st["y"]), crs="EPSG:25830")   # metres (UTM 30N)
roads_gdf = gpd.GeoDataFrame({"road_id": range(len(roads))}, geometry=[LineString(r) for r in roads], crs="EPSG:25830")
ind_gdf = gpd.GeoDataFrame(industry, geometry=gpd.points_from_xy(industry["x"], industry["y"]), crs="EPSG:25830")
print(gdf.crs.name, "| bounds (m):", np.round(gdf.total_bounds, 0))
print("as WGS84 lon/lat:", np.round(gdf.to_crs(4326).total_bounds, 4))        # degrees, for maps/interchange

# A WRONG distance: Euclidean distance on lon/lat degrees vs metres
g_ll = gdf.to_crs(4326)
d_deg = g_ll.geometry.iloc[0].distance(g_ll.geometry.iloc[1]); d_m = gdf.geometry.iloc[0].distance(gdf.geometry.iloc[1])
print(f"distance station 0–1: {d_m:.0f} m (projected) vs {d_deg:.4f} 'degrees' (meaningless as a distance)")
```

```python
# Spatial join with districts (a 3×3 grid of polygons) and aggregation
x0, y0, x1, y1 = gdf.total_bounds
xs, ys = np.linspace(x0, x1, 4), np.linspace(y0, y1, 4)
districts = gpd.GeoDataFrame({"district": [f"D{i}{j}" for i in range(3) for j in range(3)]},
                             geometry=[box(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i in range(3) for j in range(3)], crs=gdf.crs)
joined = gpd.sjoin(gdf, districts, how="left", predicate="within")
print(joined.groupby("district")["no2"].agg(["count", "mean"]).round(1))

# Proximity features: distance to the nearest road, nearest industry; counts of industry within 3 km;
# a kernel-weighted industrial emission index
gdf["dist_road_m"] = gdf.geometry.apply(lambda p: roads_gdf.distance(p).min())
near = gpd.sjoin_nearest(gdf[["station_id", "geometry"]], ind_gdf[["emission", "geometry"]], distance_col="dist_ind_m")
gdf["dist_industry_m"] = near.groupby(level=0)["dist_ind_m"].min()
D = np.hypot(gdf["x"].to_numpy()[:, None] - industry["x"].to_numpy()[None], gdf["y"].to_numpy()[:, None] - industry["y"].to_numpy()[None])
gdf["industry_within_3km"] = (D < 3000).sum(1)
gdf["industry_index"] = (industry["emission"].to_numpy()[None] * np.exp(-D / 2500)).sum(1)
print(gdf[["dist_road_m", "dist_industry_m", "industry_within_3km", "industry_index", "no2"]].corr(method="spearman")["no2"].round(2))
```

```python
# Map: target and context layers
fig, ax = plt.subplots(figsize=(6, 6))
districts.boundary.plot(ax=ax, color="lightgrey", lw=0.8)
roads_gdf.plot(ax=ax, color="k", lw=1)
gdf.plot(ax=ax, column="no2", cmap="magma_r", markersize=25, legend=True, legend_kwds={"label": "NO₂ (µg/m³)"})
ind_gdf.plot(ax=ax, color="tab:blue", marker="^", markersize=80)
ax.set_title("NO₂ stations, roads (black), industry (▲)"); ax.set_axis_off(); plt.show()
```

```python
# Spatial autocorrelation: similarity of station pairs vs distance (an empirical semivariogram)
from scipy.spatial.distance import pdist
d_pairs = pdist(gdf[["x", "y"]].to_numpy()); g_pairs = 0.5 * pdist(gdf[["no2"]].to_numpy(), "sqeuclidean")
bins = np.arange(0, 15001, 1000)
semivar = pd.Series(g_pairs).groupby(pd.cut(d_pairs, bins)).mean()
print(semivar.round(1).head(10))       # rises with distance, then levels off: the 'range' of autocorrelation
```

```python
# Random K-fold vs SPATIAL BLOCK CV for a model with coordinate + proximity features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
feats = ["x", "y", "elevation_m", "dist_road_m", "dist_industry_m", "industry_index"]
X, y = gdf[feats], gdf["no2"]
blocks = (np.floor((gdf["x"] - x0) / 6000).astype(int).astype(str) + "_" + np.floor((gdf["y"] - y0) / 6000).astype(int).astype(str))
rf = RandomForestRegressor(400, min_samples_leaf=3, random_state=0, n_jobs=-1)
for name, cv, groups in [("random KFold", KFold(5, shuffle=True, random_state=0), None),
                         ("spatial blocks (6 km)", GroupKFold(5), blocks)]:
    for fset_name, fset in [("coords+context", feats), ("context only (no x, y)", feats[2:])]:
        s = -cross_val_score(rf, gdf[fset], y, cv=cv, groups=groups, scoring="neg_mean_absolute_error")
        print(f"{name:22s} {fset_name:24s} MAE = {s.mean():.2f} ± {s.std():.2f}")
```

```python
# Spatial clustering with a PHYSICAL distance scale (metres): DBSCAN on projected coordinates
from sklearn.cluster import DBSCAN
gdf["cluster"] = DBSCAN(eps=1500, min_samples=5).fit_predict(gdf[["x", "y"]])
print(gdf["cluster"].value_counts().sort_index().to_dict(), "(−1 = noise)")
```

## 7. Interpretation
- **CRS checks:** coordinates in metres in UTM; the reprojection to lon/lat gives degrees for maps. The "distance in degrees" is a classic bug: it is meaningless as a distance and distorts with latitude.
- **District aggregation:** the counts per district are very unequal (84 stations in the central district, 3 in another), so district means from a few stations are noisy. Show the counts with the means.
- **Proximity correlations:** NO₂ decreases with the distance to the road (a negative Spearman ρ) and increases with the industry index. These features encode the emission mechanism.
- **Semivariogram:** it rises steeply up to ≈ 4–5 km and then flattens. Stations closer than that range are strongly correlated, which tells you the block size for spatial CV (≥ the range).
- **Random vs spatial CV:** random K-fold gives MAE ≈ 2.9, while spatial 6-km blocks give ≈ 3.3. The estimate for **new areas** is ~15% worse, which is the honest number if the model will map unsampled districts. The gap is moderate here because the strongest driver (the distance to roads) is a physical feature that transfers. Models relying on coordinates or on dense local sampling show much larger gaps.
- **Coordinates vs context only:** in this run, dropping x/y costs ~0.2 MAE under *both* schemes, so the coordinates carry real information (the smooth regional field in the generator) and are not mere memorisation. That is not always the case. Run this comparison: if the coordinates help under random CV but not under spatial CV, they are memorising locations.
- **DBSCAN with eps = 1.5 km** finds the dense urban cluster, and labels the scattered rural stations as noise. These are the network design "zones".

## 8. Decision framework
- **IF** the coordinates are in degrees → reproject to a local projected CRS before distances, buffers or areas.
- **IF** the model will predict at **new locations/areas** → spatial block (or cluster) CV with blocks ≥ the autocorrelation range. **IF** it is used only for gap-filling between dense stations → random CV is closer to the deployment, but report both.
- **IF** raw coordinates help a lot under random CV but not under spatial CV → the model memorises the location. Prefer physically meaningful features (distances, land use, elevation).
- **IF** the residuals show spatial patterns → a missing spatial covariate. Add one, or model the residual spatially (kriging / Gaussian-process residuals).
- **IF** you are clustering locations → DBSCAN/HDBSCAN with eps in metres (projected), not K-Means on lon/lat.
- **IF** joining large datasets → spatial indexes (automatic in `sjoin`); simplify the geometries if needed.

## 9. Common mistakes
| Mistake | Consequence | Prevention |
|---|---|---|
| Distances/areas in lon/lat degrees | Wrong features, distorted with latitude | Projected CRS |
| Mixed CRSs in a join | Empty or incorrect joins | Check and align `crs` |
| Swapped lat/lon | Points in the ocean/another continent | Plot on a map; check the bounds |
| Random CV on spatial data | Optimistic errors for new areas | Spatial block CV |
| Raw coordinates as the main features | Memorisation; poor transfer | Physical context features |
| Shapefiles with long field names/encodings | Truncated attributes | GeoPackage/GeoParquet |

## 10. Checklist
- [ ] CRS checked and a projected CRS used for computations.
- [ ] Data plotted on a map; the geometry validity checked.
- [ ] Spatial joins and aggregations with counts.
- [ ] Proximity/kernel features from domain knowledge.
- [ ] Autocorrelation range estimated (semivariogram).
- [ ] Random vs spatial block CV reported; the deployment-relevant scheme chosen.
- [ ] Residual map checked.

## 11. References
- GeoPandas documentation (CRS, spatial joins, `sjoin_nearest`). https://geopandas.org/
- Rey, S. J., Arribas-Bel, D. & Wolf, L. J. (2023). *Geographic Data Science with Python*. CRC. Free online: https://geographicdata.science/book/
- Roberts, D. R. et al. (2017). "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure." *Ecography* 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- Ploton, P. et al. (2020). "Spatial validation reveals poor predictive performance of large-scale ecological mapping models." *Nature Communications* 11, 4540. https://doi.org/10.1038/s41467-020-18321-y
- Lovelace, R., Nowosad, J. & Muenchow, J. (2019). *Geocomputation with R* (concepts transfer to Python; spatial CV chapter). https://r.geocompx.org/
- EPSG registry (CRS definitions). https://epsg.io/
