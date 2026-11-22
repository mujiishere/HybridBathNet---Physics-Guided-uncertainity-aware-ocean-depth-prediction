"""Download the three open multi-site datasets used by HybridBathNet v2 (pinned commits not
available offline, so files are verified by SHA-256 instead) and build harmonised site files.

python -m hbn2.download_open_data --raw data/v2_raw --out data/v2_sites

Licences/citations (cite all in the paper):
  Ponce, Puerto Rico  — Lumban-Gaol, Ohori & Peters (2022) Marine Geodesy 45(6) 615-644, MIT licence
                        (github.com/yustisiardhitasari/sdbcnn); reference = NOAA topobathy lidar.
  Kepulauan Seribu    — Harrys et al., SDB GUI sample data, MIT licence (github.com/rifqiharrys/sdb_gui).
  Langebaan / Saldanha — Payandeh et al. (2026) Front. Remote Sens. 7:1751006 (github.com/nasa-jpl/Satellite-Derived-Bathymetry);
                        ICESat-2 ATL24 (Magruder et al. 2025, doi:10.5067/ATLAS/ATL24.001). Caltech copyright: research use only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com"
FILES = {
    "ponce/s2.tif": ("yustisiardhitasari/sdbcnn/main/data/ponceRGBNSS_S2_190108_aoi1.tif",
                     "0644dc372f23b163ccf58150712c902b5236520a9326d0d1423b9486666ecb56"),
    "ponce/depth.tif": ("yustisiardhitasari/sdbcnn/main/data/ponce_depth_aoi1_10m_crct.tif",
                        "86518563b92c9e2258cb5bb114637269d26a2eac6671f12576ef2803f51f0645"),
    "sdbgui/image.tif": ("rifqiharrys/sdb_gui/main/docs/data/input/image.tif",
                         "9d95ad5b0950c6f9f4c45126ff5ca7e91679b81d730a3965db6e1c45d3862678"),
    **{f"sdbgui/depth_sample.{e}": (f"rifqiharrys/sdb_gui/main/docs/data/input/depth_sample.{e}", h) for e, h in {
        "cpg": "3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824",
        "dbf": "e5615348be4c84566611ed55a775ef1fd11744ea741c12c7791286eda27f4095",
        "prj": "6c3e4547d372d58b5261eb615d912ab9a2e169b36bf878ffe2fe841888b402db",
        "qix": "04fe7366707cf645062ce46abc2ae9fe42eef23f950775f8f7e018b815c8ed06",
        "shp": "d894a8cfa795f0b34f7bb76861a29855b242f988017d7b08f790a2100ae16a21",
        "shx": "10c7b7345bf7f2558099126661fecc0f290a80c97c000b346739df9bb1e6e408"}.items()},
    "jpl/s2_l2r.nc": ("nasa-jpl/Satellite-Derived-Bathymetry/main/S2B_MSI_2024_11_28_08_59_38_T33HYD_L2R_subset.nc",
                      "0291a7195502594cb4c52c255f04602e70bbee3f8db3327470992cfade2ee879"),
    "jpl/atl24.csv": ("nasa-jpl/Satellite-Derived-Bathymetry/main/atl24x_rgt_228_cycle_5_gt3r.csv",
                      "2f8e83b4ad33f80c10c54856a5b379af25ee81131478e75044615c3a241837d8"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/v2_raw"); ap.add_argument("--out", default="data/v2_sites")
    a = ap.parse_args()
    raw = Path(a.raw)
    for rel, (url, h) in FILES.items():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() or sha(p) != h:
            print("downloading", rel)
            urllib.request.urlretrieve(f"{RAW}/{url}", p)
        if sha(p) != h:
            raise SystemExit(f"checksum mismatch for {rel}: upstream file changed")
    from .sites import build_all
    print(json.dumps(build_all(raw, a.out), indent=1))


if __name__ == "__main__":
    main()
