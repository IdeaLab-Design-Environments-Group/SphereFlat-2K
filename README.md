# SphereFlat-2K

SphereFlat-2K is a dataset of 2,258 closed genus-0 surfaces (Thingi10K and procedural shapes) for learning where to place cone singularities when flattening curved surfaces, each remeshed to about 15k vertices and mapped conformally onto the unit sphere with Möbius centering. Every shape stores its sphere positions, the map's log scale factor psi, Gaussian and mean curvature, vertex areas, distance to sharp corners, synthetic visibility weights and seven 128×256 latitude-longitude grid channels. The data is in the v1.0 release and loads with `moe_data.dataset.MoebiusSphereDataset` (exact rotation augmentation); `uv run python -m moe_data.run --config config.yaml` rebuilds it from scratch.

| split | Thingi10K | procedural blobs | procedural protrusions | total |
|---|---:|---:|---:|---:|
| train | 488 | 479 | 461 | 1,428 |
| val | 89 | 64 | 52 | 205 |
| test | 101 | 57 | 66 | 224 |
| protrusion test (out of range) | 180 | – | 133 | 313 |
| protrusion test (in range) | – | – | 88 | 88 |
| **total** | **858** | **600** | **800** | **2,258** |

Code: MIT. Data: each Thingi10K shape keeps its source licence (listed per shape with author and thing id in `manifest.parquet`; NonCommercial shapes are for non-commercial use only, NoDerivatives models are excluded); procedural shapes are CC BY 4.0.
