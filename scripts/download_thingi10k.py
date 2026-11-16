"""Download the Thingi10K npz variant into data/raw/thingi10k (about 4.1 GB)."""
import sys
import thingi10k

cache_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/thingi10k"
thingi10k.init(variant="npz", cache_dir=cache_dir)
ds = thingi10k.dataset()
print("rows", len(ds))
