# moe-nn dataset

A reproducible pipeline that turns Thingi10K and procedurally generated genus-0 shapes into a training set of Möbius-centred conformal sphere maps, with per-vertex log scale factor, curvature, visibility weights and latitude-longitude grid channels, plus a PyTorch `Dataset` that loads it with exact rotation augmentation. All thresholds, paths and seeds live in `config.yaml`, and `uv run python -m moe_data.run --config config.yaml` rebuilds the dataset (add `--pilot` for a small sample).
