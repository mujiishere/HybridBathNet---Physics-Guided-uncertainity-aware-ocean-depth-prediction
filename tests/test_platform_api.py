"""API tests against a small synthetic product set (file mode, no database)."""
import json
import os

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("products")
    H, W = 50, 60
    tr = rasterio.transform.from_origin(742860, 1988700, 10, 10)
    depth = np.tile(np.linspace(1, 15, W), (H, 1)).astype(np.float32)
    depth[:5, :5] = -9999
    lay = {"depth": depth, "sigma": np.full((H, W), 0.8, np.float32)}
    for t, k in (("90", 1.3), ("95", 1.7)):
        lay[f"lower{t}"] = np.where(depth == -9999, -9999, np.clip(depth - k, 0, None)).astype(np.float32)
        lay[f"upper{t}"] = np.where(depth == -9999, -9999, depth + k).astype(np.float32)
    lay["zmax"] = np.full((H, W), 12.0, np.float32)
    conf = np.where(depth > 12, 4, 1).astype(np.uint8); conf[depth == -9999] = 0
    lay["confidence"] = conf
    for k, a in lay.items():
        with rasterio.open(d / f"testsite_{k}.tif", "w", driver="GTiff", height=H, width=W, count=1, dtype=a.dtype.name,
                           crs="EPSG:32619", transform=tr, nodata=0 if k == "confidence" else -9999) as dst:
            dst.write(a, 1)
    (d / "manifest.json").write_text(json.dumps({"version": "test", "sites": ["testsite"], "metrics": {"testsite": {"rmse": 1.0}},
                                                 "site_meta": {"testsite": {"region": "Test"}}}))
    os.environ["HBN_PRODUCTS"] = str(d)
    os.environ.pop("HBN_DSN", None)
    import importlib
    import sdb_platform.api.main as m
    importlib.reload(m)
    return TestClient(m.app), tr


def _lonlat(tr, r, c):
    from pyproj import Transformer
    x, y = rasterio.transform.xy(tr, r, c)
    return Transformer.from_crs("EPSG:32619", "EPSG:4326", always_xy=True).transform(x, y)


def test_health_and_sites(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"
    s = c.get("/api/sites").json()["sites"][0]
    assert s["name"] == "testsite" and "confidence" in s["layers"] and len(s["bounds"]) == 4


def test_point_query(client):
    c, tr = client
    lon, lat = _lonlat(tr, 20, 30)
    r = c.get(f"/api/query/point?lon={lon}&lat={lat}").json()
    assert abs(r["depth"] - float(np.linspace(1, 15, 60)[30])) < 1e-3
    assert r["lower95"] < r["lower90"] < r["depth"] < r["upper90"] < r["upper95"]
    lon, lat = _lonlat(tr, 1, 1)
    assert c.get(f"/api/query/point?lon={lon}&lat={lat}").status_code == 404


def test_tiles_render_png(client):
    c, tr = client
    import math
    lon, lat = _lonlat(tr, 25, 30)
    z = 14; n = 2 ** z
    x = int((lon + 180) / 360 * n); y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    for layer in ("depth", "confidence", "sigma", "width90", "hazard"):
        r = c.get(f"/tiles/testsite/{layer}/{z}/{x}/{y}.png?draft=2")
        assert r.status_code == 200 and r.content[:4] == b"\x89PNG"


def test_profile_and_polygon_navigation(client):
    c, tr = client
    a = _lonlat(tr, 25, 6); b = _lonlat(tr, 25, 55)
    r = c.post("/api/query/profile", json={"site": "testsite", "geometry": {"type": "LineString", "coordinates": [a, b]},
                                           "draft_m": 3.0, "n_samples": 50}).json()
    assert r["summary"]["covered"] > 40 and r["summary"]["route_status"] == "HAZARD"
    p1, p2, p3 = _lonlat(tr, 10, 40), _lonlat(tr, 40, 40), _lonlat(tr, 40, 58)
    poly = {"type": "Polygon", "coordinates": [[p1, p2, p3, p1]]}
    r = c.post("/api/query/polygon", json={"site": "testsite", "geometry": poly, "draft_m": 2.0}).json()
    assert r["cells"] > 0 and r["navigation"]["hazard_pct"] == 0.0
    nav = c.get("/api/navigation/testsite?draft=5").json()
    assert sum(v["cells"] for v in nav["classes"].values()) == int((np.tile(np.linspace(1, 15, 60), (50, 1)) > -1).sum() - 25)
