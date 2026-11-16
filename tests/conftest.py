"""Import torch before anything pulls in pymeshlab: pymeshlab bundles its own libomp, and importing it
first makes later torch ops segfault (reproduced: exit 139)."""
import torch  # noqa: F401
