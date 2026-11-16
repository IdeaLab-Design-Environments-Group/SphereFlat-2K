"""Minimal PLY I/O for MoebiusRegistration's SphereMap.

SphereMap (Include/Misha/Ply.h, PlyFile.inl) reads vertex x/y/z into floats
and face "vertex_indices" into ints; its PLY reader converts any scalar type
in the file, so the writer below just has to use standard names. Its output
has float x,y,z, double px,py,pz and uchar red,green,blue per vertex, which
generic readers such as libigl reject, so the reader here parses the header
and builds a numpy structured dtype from it.
"""
from pathlib import Path

import numpy as np

# PLY scalar type names (both the classic and the sized spellings).
_PLY_TYPES = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}
_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">"}


def write_ply(path, V, F, coord_dtype: str = "f4") -> None:
    """Write a triangle mesh as binary little-endian PLY.

    V: (n, 3) vertex positions, F: (m, 3) vertex indices. Coordinates are
    stored as float32 by default (SphereMap reads them as float anyway); pass
    coord_dtype="f8" for double. Faces are "list uchar int vertex_indices".
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V must have shape (n, 3), got {V.shape}")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"F must have shape (m, 3), got {F.shape}")
    if F.size and (F.min() < 0 or F.max() >= len(V)):
        raise ValueError("F contains out-of-range vertex indices")
    if F.size and F.max() > np.iinfo(np.int32).max:
        raise ValueError("vertex indices do not fit in int32")
    coord = np.dtype(coord_dtype).newbyteorder("<")
    if coord.kind != "f":
        raise ValueError(f"coord_dtype must be a float type, got {coord_dtype}")
    coord_name = {4: "float", 8: "double"}[coord.itemsize]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(V)}\n"
        f"property {coord_name} x\nproperty {coord_name} y\nproperty {coord_name} z\n"
        f"element face {len(F)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    vert = V.astype(coord)
    face = np.empty(len(F), dtype=[("n", "u1"), ("idx", "<i4", (3,))])
    face["n"] = 3
    face["idx"] = F
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vert.tobytes())
        f.write(face.tobytes())


def _parse_header(f):
    """Return (format, elements); elements is a list of (name, count, props).

    Each prop is (name, scalar_type) or (name, (count_type, item_type)) for lists.
    """
    if f.readline().strip() != b"ply":
        raise ValueError("not a PLY file")
    fmt = None
    elements = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("unexpected end of file in PLY header")
        words = line.decode("ascii").split()
        if not words or words[0] in ("comment", "obj_info"):
            continue
        if words[0] == "end_header":
            break
        if words[0] == "format":
            fmt = words[1]
        elif words[0] == "element":
            elements.append((words[1], int(words[2]), []))
        elif words[0] == "property":
            if not elements:
                raise ValueError("property before element in PLY header")
            if words[1] == "list":
                elements[-1][2].append((words[4], (words[2], words[3])))
            else:
                elements[-1][2].append((words[2], words[1]))
        else:
            raise ValueError(f"unknown PLY header line: {line!r}")
    if fmt not in ("ascii", *_ENDIAN):
        raise ValueError(f"unsupported PLY format {fmt!r}")
    return fmt, elements


def _scalar(name: str, endian: str) -> np.dtype:
    try:
        return np.dtype(_PLY_TYPES[name]).newbyteorder(endian)
    except KeyError:
        raise ValueError(f"unknown PLY type {name!r}") from None


def _read_binary_element(f, count, props, endian):
    """Read one element block; lists must be fixed-length triangles."""
    fields = []
    for name, typ in props:
        if isinstance(typ, tuple):
            fields.append((name + "__n", _scalar(typ[0], endian)))
            fields.append((name, _scalar(typ[1], endian), (3,)))
        else:
            fields.append((name, _scalar(typ, endian)))
    dt = np.dtype(fields)
    buf = f.read(dt.itemsize * count)
    if len(buf) != dt.itemsize * count:
        raise ValueError("PLY file truncated")
    arr = np.frombuffer(buf, dtype=dt, count=count)
    for name, typ in props:
        if isinstance(typ, tuple) and np.any(arr[name + "__n"] != 3):
            raise ValueError(f"element list {name!r} is not all triangles")
    return arr


def _read_ascii_element(f, count, props):
    rows = [f.readline().split() for _ in range(count)]
    out = {}
    col = 0
    for name, typ in props:
        if isinstance(typ, tuple):
            n = np.array([int(r[col]) for r in rows], dtype=np.int64)
            if np.any(n != 3):
                raise ValueError(f"element list {name!r} is not all triangles")
            out[name] = np.array([r[col + 1:col + 4] for r in rows], dtype=np.int64).reshape(count, 3)
            col += 4
        else:
            out[name] = np.array([r[col] for r in rows], dtype=np.float64)
            col += 1
    return out


def read_spheremap_ply(path):
    """Read SphereMap output: (X, P, F).

    X float64 (n, 3): original positions (x, y, z),
    P float64 (n, 3): spherical parametrization (px, py, pz),
    F int64 (m, 3): triangle vertex indices.
    Property types and order come from the header; binary (either endian)
    and ASCII files are supported.
    """
    with open(Path(path), "rb") as f:
        fmt, elements = _parse_header(f)
        data = {}
        for name, count, props in elements:
            if fmt == "ascii":
                data[name] = _read_ascii_element(f, count, props)
            else:
                data[name] = _read_binary_element(f, count, props, _ENDIAN[fmt])

    if "vertex" not in data or "face" not in data:
        raise ValueError("PLY file needs vertex and face elements")
    v, fc = data["vertex"], data["face"]
    for key in ("x", "y", "z", "px", "py", "pz"):
        if key not in (v.dtype.names if hasattr(v, "dtype") else v):
            raise ValueError(f"vertex property {key!r} missing (not SphereMap output?)")
    X = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("x", "y", "z")], axis=1)
    P = np.stack([np.asarray(v[k], dtype=np.float64) for k in ("px", "py", "pz")], axis=1)
    face_key = next((k for k in ("vertex_indices", "vertex_index") if k in
                     (fc.dtype.names if hasattr(fc, "dtype") else fc)), None)
    if face_key is None:
        raise ValueError("face element has no vertex_indices list")
    F = np.asarray(fc[face_key], dtype=np.int64).reshape(-1, 3)
    return X, P, F
