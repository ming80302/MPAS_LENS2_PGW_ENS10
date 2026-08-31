"""
Two regridding helpers used by stage 03:

1. SparseRegridder -- applies the pre-computed ESMF/SCRIP weight file to go
   from LENS2's POP gx1v6 ocean grid (SST) to the CAM fv0.9x1.25 grid. Same
   approach (and same weight file) as
   /glade/u/home/mingge/LENS2/MPAS_PGW/lib/regrid.py; kept as a small
   self-contained copy here rather than importing across pipeline
   directories.

2. bilinear_regular_grid -- CAM fv0.9x1.25 -> ERA5 0.25deg is a regular
   lat-lon grid to another regular lat-lon grid, so plain bilinear
   interpolation (scipy RegularGridInterpolator) is sufficient; no
   conservative/tripolar remap needed here, unlike step 1.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


class SparseRegridder:
    def __init__(self, weight_file):
        with xr.open_dataset(weight_file) as ds:
            row = ds["row"].values - 1
            col = ds["col"].values - 1
            S = ds["S"].values
            src_nx, src_ny = (int(v) for v in ds["src_grid_dims"].values)
            dst_nx, dst_ny = (int(v) for v in ds["dst_grid_dims"].values)
        self.src_shape = (src_ny, src_nx)
        self.dst_shape = (dst_ny, dst_nx)
        self.matrix = sp.csr_matrix((S, (row, col)),
                                     shape=(dst_ny * dst_nx, src_ny * src_nx))

    def __call__(self, field: np.ndarray) -> np.ndarray:
        if field.shape[-2:] != self.src_shape:
            raise ValueError(f"field spatial shape {field.shape[-2:]} != "
                              f"expected source shape {self.src_shape}")
        lead = field.shape[:-2]
        flat = field.reshape(*lead, self.src_shape[0] * self.src_shape[1])
        flat = np.nan_to_num(flat, nan=0.0)
        out = (self.matrix @ flat.reshape(-1, flat.shape[-1]).T).T
        return out.reshape(*lead, *self.dst_shape)


def bilinear_regular_grid(src_lat: np.ndarray, src_lon: np.ndarray, field: np.ndarray,
                           dst_lat: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    """field: (..., ny_src, nx_src) on a regular lat-lon grid (src_lat may be
    ascending or descending; src_lon assumed ascending, 0-360, uniform
    spacing, cyclic). Returns (..., ny_dst, nx_dst) on dst_lat/dst_lon
    (arbitrary order/spacing).

    Longitude is padded by one column on each side (wrapping around 360) so
    the interpolator can evaluate points near the 0/360 seam without going
    out of bounds.
    """
    lat_asc = src_lat[0] < src_lat[-1]
    lat = src_lat if lat_asc else src_lat[::-1]

    dlon = src_lon[1] - src_lon[0]
    lon_pad = np.concatenate([[src_lon[0] - dlon], src_lon, [src_lon[-1] + dlon]])

    lead_shape = field.shape[:-2]
    n_lead = int(np.prod(lead_shape)) if lead_shape else 1
    flat = field.reshape(n_lead, *field.shape[-2:])
    if not lat_asc:
        flat = flat[:, ::-1, :]

    out = np.empty((n_lead, len(dst_lat), len(dst_lon)), dtype=field.dtype)
    dst_lon_wrapped = np.mod(dst_lon, 360.0)
    pts = np.stack(np.meshgrid(dst_lat, dst_lon_wrapped, indexing="ij"), axis=-1)
    pts = pts.reshape(-1, 2)

    for i in range(n_lead):
        data_pad = np.concatenate(
            [flat[i][:, -1:], flat[i], flat[i][:, :1]], axis=1
        )
        interp = RegularGridInterpolator((lat, lon_pad), data_pad,
                                          bounds_error=False, fill_value=None)
        out[i] = interp(pts).reshape(len(dst_lat), len(dst_lon))

    return out.reshape(*lead_shape, len(dst_lat), len(dst_lon))
