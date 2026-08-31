"""
LENS2 (CESM2-LE) file discovery, generalized over the 10-member ensemble
(config.MEMBERS) -- adapted from
/glade/u/home/mingge/MPAS/MPAS_LENS2_PGW/lib/catalog.py, which hardcodes a
single member. Same rationale as that module's docstring (globs the archive
and parses the real date range out of each filename instead of hardcoding
chunk boundaries) -- the only change here is that every function takes
(ens_mem, var_id) explicitly rather than reading them off config, since
stage 01a needs to call this once per ensemble member.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import config


@dataclass(frozen=True)
class Chunk:
    path: Path
    start: tuple
    end: tuple


_DATE_RE = re.compile(r"\.(\d{6}|\d{8}|\d{10})-(\d{6}|\d{8}|\d{10})\.nc$")


def _parse_token(tok: str, is_end: bool) -> tuple:
    if len(tok) == 6:
        y, m = int(tok[0:4]), int(tok[4:6])
        return (y, m, 31 if is_end else 1, 18 if is_end else 0)
    if len(tok) == 8:
        y, m, d = int(tok[0:4]), int(tok[4:6]), int(tok[6:8])
        return (y, m, d, 18 if is_end else 0)
    if len(tok) == 10:
        y, m, d, h = int(tok[0:4]), int(tok[4:6]), int(tok[6:8]), int(tok[8:10])
        return (y, m, d, h)
    raise ValueError(f"unrecognized date token: {tok}")


def var_dir(var: str) -> Path:
    component = config.COMPONENT_OF[var]
    freq = config.FREQ_OF[var]
    return Path(str(config.LENS2_TSERIES).format(component=component, freq=freq)) / var


def list_chunks(var: str, case: str, ens_mem: str, var_id: str) -> list:
    d = var_dir(var)
    casename = f"{case}-{ens_mem}.{var_id}"
    pattern = str(d / f"{casename}.*.{var}.*.nc")
    chunks = []
    for f in glob.glob(pattern):
        m = _DATE_RE.search(f)
        if not m:
            continue
        chunks.append(Chunk(path=Path(f),
                             start=_parse_token(m.group(1), False),
                             end=_parse_token(m.group(2), True)))
    if not chunks:
        raise FileNotFoundError(f"No files matched {pattern}")
    return sorted(chunks, key=lambda c: c.start)


def find_file(var: str, year: int, month: int, ens_mem: str, var_id: str,
              day: int = 1, hour: int = 0) -> Path:
    case = config.case_for_year(year)
    target = (year, month, day, hour)
    for c in list_chunks(var, case, ens_mem, var_id):
        if c.start <= target <= c.end:
            return c.path
    raise FileNotFoundError(
        f"No archived chunk of {var} for member {ens_mem}.{var_id} "
        f"({case}) covers {year}-{month:02d}-{day:02d}_{hour:02d}"
    )


def available_members(var: str = "T", case: Optional[str] = None) -> list:
    """Sanity check: confirm every member in config.MEMBERS actually has
    `var` archived, rather than assuming it."""
    case = case or config.HIST_CASE
    d = var_dir(var)
    pattern = str(d / f"{case}-*.{var}.*.nc")
    found = set()
    rx = re.compile(rf"{re.escape(case)}-(\d+)\.(\d+)\.")
    for f in glob.glob(pattern):
        m = rx.search(Path(f).name)
        if m:
            found.add((m.group(1), m.group(2)))
    return sorted(found)
