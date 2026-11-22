-- HybridBathNet v2 spatial store (PostgreSQL >= 14 + PostGIS >= 3.2)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS sdb;

-- Study / production sites
CREATE TABLE IF NOT EXISTS sdb.site (
    site_id         SERIAL PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,          -- e.g. ponce_pr
    region          TEXT,
    native_crs      TEXT NOT NULL,                 -- e.g. EPSG:32619
    reference_type  TEXT,                          -- lidar | sonar | icesat2_atl24 | gebco
    water_type      TEXT,
    image_date      DATE,
    footprint       geometry(Polygon, 4326),
    meta            JSONB DEFAULT '{}'::jsonb
);

-- One row per trained/deployed model ensemble
CREATE TABLE IF NOT EXISTS sdb.model_run (
    run_id          SERIAL PRIMARY KEY,
    model_name      TEXT NOT NULL DEFAULT 'HybridBathNetV2',
    version         TEXT NOT NULL,
    n_members       INT,
    config          JSONB,
    calibration     JSONB,                         -- conformal quantiles per alpha / depth bin
    metrics         JSONB,                         -- cross-validated accuracy & calibration
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (model_name, version)
);

-- Raster products (Cloud-Optimised GeoTIFFs on disk / object storage)
CREATE TABLE IF NOT EXISTS sdb.product (
    product_id      SERIAL PRIMARY KEY,
    site_id         INT REFERENCES sdb.site ON DELETE CASCADE,
    run_id          INT REFERENCES sdb.model_run ON DELETE CASCADE,
    layer           TEXT NOT NULL,                 -- depth | sigma | lower90 | ... | confidence
    uri             TEXT NOT NULL,
    units           TEXT,
    stats           JSONB,
    UNIQUE (site_id, run_id, layer)
);

-- Per-pixel predictions (10 m cells) for SQL analytics: point, polygon & route queries
CREATE TABLE IF NOT EXISTS sdb.depth_cell (
    site_id         INT REFERENCES sdb.site ON DELETE CASCADE,
    run_id          INT REFERENCES sdb.model_run ON DELETE CASCADE,
    row_i           INT,
    col_j           INT,
    geom            geometry(Point, 4326) NOT NULL,
    depth_m         REAL,
    sigma_m         REAL,
    aleatoric_sd    REAL,
    epistemic_sd    REAL,
    lower90         REAL,
    upper90         REAL,
    lower95         REAL,
    upper95         REAL,
    zmax_m          REAL,
    confidence      SMALLINT,                      -- 1 A high, 2 B medium, 3 C low, 4 D optically deep
    PRIMARY KEY (site_id, run_id, row_i, col_j)
);
CREATE INDEX IF NOT EXISTS depth_cell_gix ON sdb.depth_cell USING GIST (geom);
CREATE INDEX IF NOT EXISTS depth_cell_conf ON sdb.depth_cell (site_id, confidence);

-- Independent reference observations used for validation
CREATE TABLE IF NOT EXISTS sdb.reference_obs (
    obs_id          BIGSERIAL PRIMARY KEY,
    site_id         INT REFERENCES sdb.site ON DELETE CASCADE,
    geom            geometry(Point, 4326) NOT NULL,
    depth_m         REAL NOT NULL,
    n_raw           INT,
    source          TEXT
);
CREATE INDEX IF NOT EXISTS reference_obs_gix ON sdb.reference_obs USING GIST (geom);

-- Planner areas of interest saved from the dashboard
CREATE TABLE IF NOT EXISTS sdb.planner_aoi (
    aoi_id          SERIAL PRIMARY KEY,
    name            TEXT,
    geom            geometry(Geometry, 4326) NOT NULL,
    draft_m         REAL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    summary         JSONB
);

-- Convenience view: navigation-safety screen. A cell is 'hazard' for a vessel when the
-- conservative (shoal-side) 95 % bound is shallower than draft + under-keel clearance.
DROP FUNCTION IF EXISTS sdb.navigation_screen(TEXT, REAL, REAL);
CREATE OR REPLACE FUNCTION sdb.navigation_screen(p_site TEXT, p_draft DOUBLE PRECISION, p_ukc DOUBLE PRECISION DEFAULT 0.5)
RETURNS TABLE (status TEXT, n_cells BIGINT, area_km2 DOUBLE PRECISION) AS $$
    SELECT CASE
             WHEN c.confidence = 4 THEN 'unknown_optically_deep'
             WHEN c.lower95 < p_draft + p_ukc THEN 'hazard'
             WHEN c.depth_m < p_draft + p_ukc + 2 THEN 'caution'
             ELSE 'clear' END AS status,
           count(*), count(*) * 100.0 / 1e6
    FROM sdb.depth_cell c JOIN sdb.site s USING (site_id)
    WHERE s.name = p_site
      AND c.run_id = (SELECT max(run_id) FROM sdb.model_run)
    GROUP BY 1;
$$ LANGUAGE sql STABLE;
