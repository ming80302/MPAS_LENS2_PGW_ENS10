"""
Pure-Python reader/writer for the WPS intermediate ("IM") binary format,
built directly from the real WPS Fortran source
(/glade/work/mingge/WPSV4.1_intel_serial_large-file/util/src/{met_data_module,
write_met_module,read_met_module}.f90) rather than from memory of the
format spec, and cross-validated against a real file
(/glade/campaign/mmm/c3we/mingge/wpsv4.5_mpas/ERA5:1998-06-01_00) using the
genuine WPS util/rd_intermediate.exe as ground truth.

Key facts pulled from the Fortran source (not assumed):
  - Compiled with `-convert big_endian` (configure.wps FFLAGS) -- the files
    are big-endian, and standard ifort unformatted sequential records use a
    4-byte record-length marker before and after each record's data.
  - met_data_module.f90 field widths: field=9, hdate=24, units=25,
    map_source=32, desc=46 chars.
  - write_met_module.f90 silently renames field 'GHT' -> 'HGT' on disk (and
    read_met_module.f90 renames it back on read) -- rd_intermediate.exe
    still prints 'GHT', confirming the round-trip is transparent to readers
    that go through the module; we replicate the same rename so a file we
    write is byte-compatible with anything else that reads it.
  - version 5 (WPS format) + iproj=0 (PROJ_LATLON, what ERA5 uses here) has
    5 unformatted records per field: version; header info; SWCORNER/lat-lon
    block; is_wind_grid_rel flag; the (ny, nx) data slab.

Design choice: every record except the data slab is round-tripped as raw
bytes, not re-encoded from parsed values. This pipeline only ever adds a
delta to a slab's *values* for a handful of matching fields -- everything
else in the file (including all non-perturbed fields' slabs) is copied
byte-for-byte, so there is no way for this code to corrupt a field it
wasn't supposed to touch, and no dependency on correctly reproducing
WPS's per-projection header encoding.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field as dc_field

import numpy as np

_ENDIAN = ">"


def _read_record(f) -> bytes:
    head = f.read(4)
    if len(head) < 4:
        return None
    (reclen,) = struct.unpack(_ENDIAN + "i", head)
    data = f.read(reclen)
    tail = f.read(4)
    (reclen2,) = struct.unpack(_ENDIAN + "i", tail)
    if reclen2 != reclen:
        raise ValueError(f"corrupt record: lead len {reclen} != trail len {reclen2}")
    return data


def _write_record(f, data: bytes) -> None:
    reclen = struct.pack(_ENDIAN + "i", len(data))
    f.write(reclen)
    f.write(data)
    f.write(reclen)


# record 2 layout (version 5, all projections share this prefix)
_HDR_FMT = _ENDIAN + "24s f 32s 9s 25s 46s f i i i"
_HDR_LEN = struct.calcsize(_HDR_FMT)  # 156 bytes


@dataclass
class MetField:
    version_raw: bytes      # record 1, raw (4 bytes, int32)
    hdate: str
    xfcst: float
    map_source: str
    field: str
    units: str
    desc: str
    xlvl: float
    nx: int
    ny: int
    iproj: int
    proj_record_raw: bytes  # record 3, raw (projection-specific, passthrough)
    wind_flag_raw: bytes    # record 4, raw (is_wind_grid_rel, passthrough)
    slab: np.ndarray        # (ny, nx) float32, decoded from record 5


def read_file(path) -> list:
    """Parse every field in a WPS intermediate file. Returns a list of
    MetField, in on-disk order."""
    records = []
    with open(path, "rb") as f:
        while True:
            version_raw = _read_record(f)
            if version_raw is None:
                break
            hdr_raw = _read_record(f)
            (hdate_b, xfcst, map_source_b, field_b, units_b, desc_b,
             xlvl, nx, ny, iproj) = struct.unpack(_HDR_FMT, hdr_raw)
            field_name = field_b.decode("ascii").strip()
            if field_name == "HGT":
                field_name = "GHT"

            proj_record_raw = _read_record(f)
            wind_flag_raw = _read_record(f)
            slab_raw = _read_record(f)
            expected_bytes = nx * ny * 4
            if len(slab_raw) != expected_bytes:
                raise ValueError(
                    f"{path}: field {field_name} slab is {len(slab_raw)} bytes, "
                    f"expected {expected_bytes} ({nx}x{ny} float32)"
                )
            slab = np.frombuffer(slab_raw, dtype=_ENDIAN + "f4").reshape(ny, nx).copy()

            records.append(MetField(
                version_raw=version_raw,
                hdate=hdate_b.decode("ascii"),
                xfcst=xfcst,
                map_source=map_source_b.decode("ascii"),
                field=field_name,
                units=units_b.decode("ascii").strip(),
                desc=desc_b.decode("ascii"),
                xlvl=xlvl,
                nx=nx, ny=ny, iproj=iproj,
                proj_record_raw=proj_record_raw,
                wind_flag_raw=wind_flag_raw,
                slab=slab,
            ))
    return records


def write_file(path, records: list) -> None:
    with open(path, "wb") as f:
        for r in records:
            _write_record(f, r.version_raw)

            field_on_disk = "HGT" if r.field == "GHT" else r.field
            hdr = struct.pack(
                _HDR_FMT,
                r.hdate.encode("ascii").ljust(24)[:24],
                r.xfcst,
                r.map_source.encode("ascii").ljust(32)[:32],
                field_on_disk.encode("ascii").ljust(9)[:9],
                r.units.encode("ascii").ljust(25)[:25],
                r.desc.encode("ascii").ljust(46)[:46],
                r.xlvl,
                r.nx, r.ny, r.iproj,
            )
            _write_record(f, hdr)
            _write_record(f, r.proj_record_raw)
            _write_record(f, r.wind_flag_raw)
            _write_record(f, r.slab.astype(_ENDIAN + "f4", copy=False).tobytes())
