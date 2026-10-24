"""
Search and download Sentinel-2 L2A products from the Copernicus Data Space
Ecosystem (CDSE) using the OData API.

Setup:
    1. Create a free account at https://dataspace.copernicus.eu
    2. Set credentials via environment variables (REQUIRED — no fallback):

         # Windows PowerShell:
         $env:CDSE_USERNAME="you@example.com"
         $env:CDSE_PASSWORD="your-password"

         # Linux / macOS:
         export CDSE_USERNAME="you@example.com"
         export CDSE_PASSWORD="your-password"

Usage:
    python -m src.download_sentinel2 --config config.yaml
    python -m src.download_sentinel2 --config config.yaml --limit 2  # cap at 2 downloads
"""
import argparse
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .utils import load_config, get_logger, ensure_dir

LOG = get_logger("download_sentinel2")

# §7 Security: credentials MUST be set via environment variables.
# No fallback constants — storing credentials in source code is a security risk.

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# CDSE download endpoint — note: download.dataspace.copernicus.eu, NOT catalogue.
DOWNLOAD_URL_TMPL = "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"


def get_access_token(username: str, password: str) -> str:
    """Authenticate against the CDSE Keycloak server and return a bearer token.
    Tokens expire after ~10 minutes — call this again if a download starts
    failing with a 401."""
    data = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    resp = requests.post(TOKEN_URL, data=data, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_products(aoi_wkt: str, start_date: str, end_date: str,
                     max_cloud_cover: float = 20.0,
                     collection: str = "SENTINEL-2",
                     product_type: str = "S2MSI2A") -> list:
    """Query the CDSE OData catalogue. No auth required for search itself."""
    filter_str = (
        f"Collection/Name eq '{collection}' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}') "
        f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
        f"and ContentDate/Start lt {end_date}T00:00:00.000Z "
        f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        f"and att/OData.CSC.StringAttribute/Value eq '{product_type}') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
    )
    params = {"$filter": filter_str, "$orderby": "ContentDate/Start asc", "$top": 100}
    resp = requests.get(CATALOG_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("value", [])


def download_product(product_id: str, product_name: str, token: str, out_dir: str) -> Path:
    """Stream-download one product .zip using the bearer token.

    CDSE's download endpoint redirects through several hops (catalogue →
    download.dataspace.copernicus.eu → CloudFront S3).  We must forward the
    Authorization header on every redirect, which the default requests
    behaviour strips.  We do this with a custom session + event hook.
    """
    out_path = Path(out_dir) / f"{product_name}.zip"
    if out_path.exists():
        LOG.info("Already downloaded: %s", out_path.name)
        return out_path

    url = DOWNLOAD_URL_TMPL.format(product_id=product_id)
    ensure_dir(out_dir)

    # Build a session that re-attaches the bearer token on every redirect.
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    def _keep_auth(r, *args, **kwargs):
        """Re-inject the Authorization header after every redirect."""
        if r.is_redirect:
            r.headers["Authorization"] = f"Bearer {token}"

    session.hooks["response"] = [_keep_auth]

    with session.get(url, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=product_name[:30]
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    LOG.info("Saved → %s  (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of products to download (useful for testing)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    username = os.environ.get("CDSE_USERNAME", "")
    password = os.environ.get("CDSE_PASSWORD", "")
    if not username or not password:
        raise SystemExit(
            "CDSE credentials not found.\n"
            "  Set CDSE_USERNAME and CDSE_PASSWORD as environment variables:\n"
            "    Windows PowerShell:  $env:CDSE_USERNAME='you@example.com'\n"
            "                        $env:CDSE_PASSWORD='your-password'\n"
            "    Linux/macOS:         export CDSE_USERNAME='you@example.com'\n"
            "                         export CDSE_PASSWORD='your-password'\n"
            "  Never commit credentials to version control."
        )
    LOG.info("Using CDSE account: %s", username)

    aoi_wkt = cfg["aoi"]["wkt"]
    s2_cfg = cfg["sentinel2"]

    LOG.info("Searching Copernicus Data Space catalogue...")
    products = search_products(
        aoi_wkt,
        s2_cfg["start_date"],
        s2_cfg["end_date"],
        max_cloud_cover=s2_cfg["max_cloud_cover"],
        product_type=s2_cfg["product_type"],
    )
    LOG.info("Found %d products matching AOI/date/cloud-cover filter", len(products))
    if not products:
        LOG.warning("No products found — try widening the date range or cloud threshold.")
        return

    token = get_access_token(username, password)
    token_time = time.time()

    out_dir = s2_cfg["raw_dir"]
    products_to_dl = products[:args.limit] if args.limit else products
    LOG.info("Downloading %d product(s) to %s", len(products_to_dl), out_dir)
    for p in products_to_dl:
        # CDSE tokens expire after ~10 minutes; refresh proactively
        if time.time() - token_time > 540:
            token = get_access_token(username, password)
            token_time = time.time()
        try:
            download_product(p["Id"], p["Name"], token, out_dir)
        except Exception as exc:
            LOG.error("Failed to download %s: %s — skipping", p["Name"], exc)

    LOG.info("Done. Downloaded %d product(s) to %s", len(products_to_dl), out_dir)


if __name__ == "__main__":
    main()
