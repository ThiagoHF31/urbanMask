# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The UrbanMask authors
# Licensed under the Apache License, Version 2.0 — see the LICENSE file in the repository root.
"""
Urban Mask Detection Pipeline for Kubeflow
Dual classification: Heuristic + ML (PCA + Random Forest)
14 spectral/textural features from Sentinel-2

Architecture:
    GEE extraction (parallel batches, service account)
    → Dual classification (urban_heu heuristic + urban_pca PCA/RF)
    → PostgreSQL persistence (bulk insert)
"""

import gc
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ee
import geemap  # noqa: F401 – kept for downstream compatibility
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

EXECUTION_INTERVAL_DAYS = 15
TEMPORAL_WINDOW_DAYS    = 150
SCALE                   = 10
MAX_CLOUD_COVERAGE      = 30

# Service-account quotas permitem lotes muito maiores que os 5 000 do usuário padrão.
# Default elevado para 10 000 (limite seguro abaixo do payload de 10 MB do GEE).
BATCH_SIZE             = int(os.getenv('BATCH_SIZE',             '5000'))
TILE_SCALE             = int(os.getenv('GEE_TILE_SCALE',         '8'))
MAX_CONCURRENT_BATCHES = int(os.getenv('GEE_CONCURRENT_BATCHES', '4'))
DB_INSERT_CHUNK        = int(os.getenv('DB_INSERT_CHUNK',        '5000'))

# Paths
POINTS_CACHE_PATH      = '/home/jovyan/sentinel_pixel_grid.parquet'
POINTS_CACHE_HASH_PATH = '/home/jovyan/sentinel_pixel_grid_gpkg_hash.txt'
DEFAULT_GPKG_PATH      = '/home/jovyan/areas.gpkg'
CHECKPOINT_DIR    = Path(os.getenv('CHECKPOINT_DIR', '/home/jovyan/checkpoints'))

# Pasta com os artefatos do modelo (pipeline_final_cv.joblib + best_threshold_cv.pkl)
MODEL_DIR = Path(os.getenv('MODEL_DIR', Path(__file__).parent / 'model'))

# Ordem exata exigida pelo pipeline scikit-learn
ML_FEATURES: List[str] = [
    'ndvi', 'ndvi_re', 'gndvi', 'savi', 'mndwi',
    'ndbi', 'ui', 'nbi', 'bsi', 'baei', 'ibi',
    'ndvi_std', 'entropy', 'ndbi_std',
]

DB_CONFIG = {
    'host':     os.getenv('DB_HOST',     ''),
    'port':     os.getenv('DB_PORT',     '5432'),
    'user':     os.getenv('DB_USER',     ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME',     ''),
}

# ─── GEE Helpers ──────────────────────────────────────────────────────────────

def safe_get_info(ee_object, description: str = "GEE object", max_retries: int = 3):
    """
    Wrapper seguro para ee.ComputedObject.getInfo() com retry e backoff exponencial.
    Lida com timeouts, erros de quota e falhas transitórias de rede.
    """
    for attempt in range(max_retries):
        try:
            return ee_object.getInfo()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 10  # 10 s → 20 s → 40 s
                logger.warning(
                    f"GEE getInfo falhou para '{description}' "
                    f"(tentativa {attempt + 1}/{max_retries}): {e}. "
                    f"Retentando em {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"GEE getInfo falhou definitivamente para '{description}' "
                    f"após {max_retries} tentativas: {e}"
                )
                raise


def initialize_earth_engine() -> None:
    """
    Inicializa GEE via Service Account.

    Métodos suportados (em ordem de prioridade):
      1. Variável de ambiente GEE_SERVICE_ACCOUNT (JSON completo) — preferencial
      2. Arquivo montado em GCP_CREDENTIALS_PATH — fallback

    Com service account, configura deadline de 10 minutos para operações pesadas.
    """
    from google.oauth2 import service_account as sa

    try:
        gee_json = os.getenv('GEE_SERVICE_ACCOUNT')

        if gee_json:
            logger.info("Usando credenciais da variável de ambiente GEE_SERVICE_ACCOUNT")
            info  = json.loads(gee_json)
            creds = sa.Credentials.from_service_account_info(
                info,
                scopes=["https://www.googleapis.com/auth/earthengine"],
            )
            ee.Initialize(creds)
            logger.info(f"Earth Engine inicializado: {info.get('client_email')}")

        else:
            creds_path = os.getenv('GCP_CREDENTIALS_PATH', '/secret/gcp-credentials.json')
            if not os.path.exists(creds_path):
                raise FileNotFoundError(
                    "Credenciais não encontradas. Configure:\n"
                    "  1. GEE_SERVICE_ACCOUNT (JSON completo), ou\n"
                    "  2. GCP_CREDENTIALS_PATH (caminho do arquivo JSON)"
                )
            with open(creds_path) as f:
                info = json.load(f)
            email = info.get('client_email')
            if not email:
                raise ValueError("Campo 'client_email' ausente no JSON de credenciais")
            creds = ee.ServiceAccountCredentials(email, creds_path)
            ee.Initialize(creds)
            logger.info(f"Earth Engine inicializado: {email}")

        # Service account: aumenta timeout para operações de sampleRegions com lotes grandes
        ee.data.setDeadline(600_000)  # 10 minutos em ms
        logger.info("GEE deadline configurado: 10 minutos (service account)")

    except Exception as e:
        logger.error(f"Erro crítico na inicialização do Earth Engine: {e}")
        raise


def cloud_mask_s2(image: ee.Image) -> ee.Image:
    """
    Aplica máscara de nuvens (banda QA60) e renomeia bandas do Sentinel-2 SR.
    Retorna imagem normalizada para reflectância (÷ 10000).
    """
    qa   = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return (
        image
        .select(
            ['B2',    'B3',    'B4',  'B5',  'B8',  'B11',  'B12'],
            ['Blue', 'Green', 'Red', 'RE1', 'NIR', 'SWIR1', 'SWIR2'],
        )
        .updateMask(mask)
        .divide(10000)
    )

# ─── Classification ───────────────────────────────────────────────────────────

def urban_classification_logic(indices: Dict[str, float]) -> Optional[int]:
    """
    Classifica pixel como urbano por heurística espectral (modelo legado).

    Critérios (AND):
      - NBI   > 0.18  (área construída)
      - UI    > 0.05  (urban index)
      - NDVI  < 0.20  (baixa vegetação)
      - MNDWI > -0.50 (descarta água)

    Returns: 1 (urbano), 0 (não urbano), None (dados inválidos)
    """
    nbi   = indices.get('nbi')
    ui    = indices.get('ui')
    ndvi  = indices.get('ndvi')
    mndwi = indices.get('mndwi')

    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [nbi, ui, ndvi, mndwi]):
        return None

    return 1 if (nbi > 0.18 and ui > 0.05 and ndvi < 0.20 and mndwi > -0.50) else 0


def load_ml_artifacts() -> Tuple[Any, float]:
    """
    Carrega pipeline scikit-learn (PCA + Random Forest) e threshold de decisão.

    Deve ser chamado UMA VEZ fora do loop de processamento para evitar I/O repetido.

    Returns:
        (ml_pipeline, best_threshold)
    """
    pipeline_path  = MODEL_DIR / 'pipeline_final_cv.joblib'
    threshold_path = MODEL_DIR / 'best_threshold_cv.pkl'

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline ML não encontrado: {pipeline_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold ML não encontrado: {threshold_path}")

    ml_pipeline    = joblib.load(pipeline_path)
    best_threshold = float(joblib.load(threshold_path))

    logger.info(f"Modelo ML carregado : {pipeline_path.name}")
    logger.info(f"Threshold decisão   : {best_threshold:.6f}")
    return ml_pipeline, best_threshold


def classify_urban_ml(
    df: pd.DataFrame,
    ml_pipeline,
    best_threshold: float,
) -> pd.Series:
    """
    Aplica pipeline PCA + Random Forest sobre as 14 features.

    Tratamento defensivo: np.inf e NaN → 0 antes de predict_proba.
    Classificação binária: probabilidade >= threshold → urban_pca = 1.

    Returns:
        pd.Series com valores inteiros 0/1 (urban_pca)
    """
    X = df[ML_FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = ml_pipeline.predict_proba(X)[:, 1]
    return pd.Series((proba >= best_threshold).astype(int), index=df.index)

# ─── Grid Generation ──────────────────────────────────────────────────────────

def generate_pixel_grid_from_polygons(
    polygons_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Gera grade de pontos (pixels Sentinel-2 de 10 m) sobre os polígonos.

    Operação custosa executada UMA VEZ — resultado cacheado em parquet.
    Usa passo angular de ~0.00009° (~10 m no Equador) para simular a grade S2.
    """
    logger.info("Gerando grade de pixels sobre os polígonos (operação única)...")
    all_points: List[Dict] = []
    total = len(polygons_gdf)

    fallback_count = 0
    for idx, row in polygons_gdf.iterrows():
        if idx % 100 == 0:
            logger.info(f"  Geometria {idx}/{total}")
        geom        = row['geometry']
        degree_step = 0.00009  # ~10 m no Equador, equivalente ao SCALE global
        minx, miny, maxx, maxy = geom.bounds

        pts_antes = len(all_points)
        for x in np.arange(minx, maxx, degree_step):
            for y in np.arange(miny, maxy, degree_step):
                pt = Point(x, y)
                if geom.contains(pt):
                    pt_dict: Dict = {
                        'longitude': x,
                        'latitude':  y,
                        'centroide': f'POINT({x} {y})',
                    }
                    for col in polygons_gdf.columns:
                        if col != 'geometry':
                            pt_dict[col] = row.get(col)
                    all_points.append(pt_dict)

        # Fallback: APMs menores que o step (~10 m) ou de geometria fina não
        # contêm nenhum ponto da grade. representative_point() sempre cai dentro
        # do polígono — garante cobertura mínima de 1 pixel por APM.
        if len(all_points) == pts_antes:
            rep = geom.representative_point()
            pt_dict = {
                'longitude': rep.x,
                'latitude':  rep.y,
                'centroide': f'POINT({rep.x} {rep.y})',
            }
            for col in polygons_gdf.columns:
                if col != 'geometry':
                    pt_dict[col] = row.get(col)
            all_points.append(pt_dict)
            fallback_count += 1

    df_pts = pd.DataFrame(all_points)
    df_pts['pixel_id'] = range(1, len(df_pts) + 1)
    logger.info(f"Grade gerada: {len(df_pts)} pixels únicos "
                f"({fallback_count} via fallback representative_point)")
    return df_pts

# ─── Pixel Processor ─────────────────────────────────────────────────────────

class PixelProcessor:
    """
    Extrai 14 features espectrais/texturais do Sentinel-2 via GEE (lotes paralelos).

    Features calculadas:
        ndvi, ndvi_re, gndvi, savi, mndwi, ndbi, ui, nbi, bsi, baei, ibi,
        ndvi_std (desvio temporal), entropy (GLCM), ndbi_std (desvio temporal)
    """

    def __init__(
        self,
        gpkg_path: str,
        target_date: str = None,
        execution_timestamp: datetime = None,
    ):
        self.gpkg_path           = gpkg_path
        self.target_date         = target_date or datetime.now().strftime('%Y-%m-%d')
        self.execution_timestamp = execution_timestamp or datetime.now()
        self.points_df: Optional[pd.DataFrame] = None

    def load_or_generate_points(self) -> pd.DataFrame:
        """Carrega cache parquet ou gera a grade de pixels a partir do GPKG."""
        if os.path.exists(POINTS_CACHE_PATH):
            logger.info(f"Cache de pontos encontrado: {POINTS_CACHE_PATH}")
            self.points_df = pd.read_parquet(POINTS_CACHE_PATH)
        else:
            logger.info(f"Cache não encontrado. Gerando a partir de: {self.gpkg_path}")
            if not os.path.exists(self.gpkg_path):
                raise FileNotFoundError(f"GPKG não encontrado: {self.gpkg_path}")

            gdf = gpd.read_file(self.gpkg_path)
            gdf = gdf.rename(columns={col: col.lower() for col in gdf.columns if col != 'geometry'})
            for col in gdf.select_dtypes(include=['datetime']).columns:
                gdf[col] = gdf[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                logger.info(f"Reprojetando {gdf.crs} → EPSG:4326")
                gdf = gdf.to_crs(epsg=4326)

            self.points_df = generate_pixel_grid_from_polygons(gdf)
            logger.info(f"Salvando cache em: {POINTS_CACHE_PATH}")
            self.points_df.to_parquet(POINTS_CACHE_PATH, index=False)

        logger.info(f"{len(self.points_df)} pixels prontos para envio ao GEE")
        return self.points_df

    def extract_indices_batch(
        self,
        batch_df: pd.DataFrame,
        batch_label: str = "",
    ) -> pd.DataFrame:
        """
        Extrai as 14 features do Sentinel-2 via GEE para um lote de pontos.

        Fluxo GEE:
          1. FeatureCollection dos pontos do lote
          2. ImageCollection S2 filtrada + mascarada
          3. Desvio padrão temporal (ndvi_std, ndbi_std) via stdDev reducer
          4. Textura GLCM (entropy) a partir da banda NIR mediana
          5. Índices espectrais da imagem mediana
          6. sampleRegions (paralelo no lado GEE, tileScale configurável)
          7. Merge com metadados locais via pixel_id

        Args:
            batch_df:    DataFrame com 'pixel_id', 'longitude', 'latitude'
            batch_label: Prefixo para logs

        Returns:
            DataFrame enriquecido com as 14 features; vazio se falhar ou sem dados.
        """
        end_date   = self.target_date
        start_date = (
            datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=TEMPORAL_WINDOW_DAYS)
        ).strftime('%Y-%m-%d')

        try:
            # ── 1. FeatureCollection de pontos ─────────────────────────────
            pts_fc = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry.Point([float(row['longitude']), float(row['latitude'])]),
                    {'pixel_id': int(row['pixel_id'])},
                )
                for _, row in batch_df.iterrows()
            ])

            # ── 2. Coleção S2 filtrada e mascarada ─────────────────────────
            s2_col = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(pts_fc.geometry().bounds())
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_COVERAGE))
                .map(cloud_mask_s2)
            )

            col_size = safe_get_info(
                s2_col.size(),
                description=f"{batch_label} S2 size ({start_date}→{end_date})",
            )
            if col_size == 0:
                logger.warning(f"{batch_label}: sem imagens S2 disponíveis no período")
                return pd.DataFrame()
            logger.info(f"{batch_label}: {col_size} imagens S2 encontradas")

            # ── 3. Imagem mediana e seleção de bandas base ─────────────────
            s2_med = s2_col.median()
            blue   = s2_med.select('Blue')
            green  = s2_med.select('Green')
            red    = s2_med.select('Red')
            nir    = s2_med.select('NIR')
            swir1  = s2_med.select('SWIR1')
            swir2  = s2_med.select('SWIR2')

            # ── 4. Desvio padrão temporal (sobre a coleção inteira) ────────
            ndvi_col  = s2_col.map(lambda img: img.normalizedDifference(['NIR', 'Red']).rename('ndvi'))
            ndbi_col  = s2_col.map(lambda img: img.normalizedDifference(['SWIR1', 'NIR']).rename('ndbi'))
            ndvi_std  = ndvi_col.reduce(ee.Reducer.stdDev()).rename('ndvi_std')
            ndbi_std  = ndbi_col.reduce(ee.Reducer.stdDev()).rename('ndbi_std')

            # ── 5. Textura GLCM – Entropia da banda NIR ───────────────────
            # Escalonamos para int16 (×10000) para o GLCM funcionar corretamente
            glcm    = nir.multiply(10000).toInt().glcmTexture(size=3)
            entropy = glcm.select('NIR_ent').rename('entropy')

            # ── 6. Índices espectrais (imagem mediana) ────────────────────

            # Diferenças normalizadas
            ndvi    = s2_med.normalizedDifference(['NIR', 'Red']).rename('ndvi')
            ndvi_re = s2_med.normalizedDifference(['NIR', 'RE1']).rename('ndvi_re')
            gndvi   = s2_med.normalizedDifference(['NIR', 'Green']).rename('gndvi')
            mndwi   = s2_med.normalizedDifference(['Green', 'SWIR1']).rename('mndwi')
            ndbi    = s2_med.normalizedDifference(['SWIR1', 'NIR']).rename('ndbi')
            ui      = s2_med.normalizedDifference(['SWIR2', 'NIR']).rename('ui')

            # Expressões com múltiplas bandas
            nbi = s2_med.expression(
                '(RED * SWIR1) / NIR',
                {'RED': red, 'SWIR1': swir1, 'NIR': nir},
            ).rename('nbi')

            savi = s2_med.expression(
                '((NIR - RED) * 1.5) / (NIR + RED + 0.5)',
                {'NIR': nir, 'RED': red},
            ).rename('savi')

            bsi = s2_med.expression(
                '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
                {'SWIR1': swir1, 'RED': red, 'NIR': nir, 'BLUE': blue},
            ).rename('bsi')

            baei = s2_med.expression(
                '(RED + 0.3) / (GREEN + SWIR1)',
                {'RED': red, 'GREEN': green, 'SWIR1': swir1},
            ).rename('baei')

            # IBI depende de ndbi, savi e mndwi já calculados acima
            ibi = s2_med.expression(
                '(NDBI - (SAVI + MNDWI) / 2.0) / (NDBI + (SAVI + MNDWI) / 2.0)',
                {'NDBI': ndbi, 'SAVI': savi, 'MNDWI': mndwi},
            ).rename('ibi')

            # ── 7. Stack das 14 features ──────────────────────────────────
            stack = ee.Image.cat([
                ndvi, ndvi_re, gndvi, savi, mndwi,
                ndbi, ui, nbi, bsi, baei, ibi,
                ndvi_std, entropy, ndbi_std,
            ])

            # ── 8. sampleRegions paralelo no GEE ─────────────────────────
            sampled_fc = stack.sampleRegions(
                collection=pts_fc,
                scale=SCALE,
                tileScale=TILE_SCALE,
                geometries=False,
            )

            sampled_info = safe_get_info(
                sampled_fc,
                description=f"{batch_label} sampleRegions ({len(batch_df)} pontos)",
            )
            feats = sampled_info.get('features', [])
            if not feats:
                logger.warning(f"{batch_label}: sampleRegions retornou vazio")
                return pd.DataFrame()

            raw_df = pd.DataFrame([f['properties'] for f in feats])
            if raw_df.empty:
                return pd.DataFrame()

            # ── 9. Merge com metadados locais via pixel_id ────────────────
            result = batch_df.merge(raw_df, on='pixel_id', how='inner')
            if result.empty:
                return pd.DataFrame()

            result['date_time']   = end_date
            result['target_date'] = end_date
            return result

        except Exception as e:
            logger.error(f"{batch_label}: Erro na extração GEE: {e}")
            return pd.DataFrame()

# ─── Coalesce helpers (garantia "sem campos null") ────────────────────────────

def _s(v) -> str:
    """Converte para str, substituindo None/NaN por ''."""
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v)


def _i(v) -> int:
    """Converte para int, substituindo None/NaN por 0."""
    if v is None:
        return 0
    try:
        if pd.isna(v):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _f(v) -> float:
    """Converte para float, substituindo None/NaN/inf por 0.0."""
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        fv = float(v)
        if np.isinf(fv):
            return 0.0
        return fv
    except (TypeError, ValueError):
        return 0.0


# ─── Database Manager ─────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Gerenciador PostgreSQL (SQLAlchemy).

    Suporta:
      - Pool de conexões otimizado para alto volume
      - Esquema com 14 features espectrais/texturais
      - Colunas urban_heu (heurístico) e urban_pca (ML)
      - Bulk insert em chunks com ON CONFLICT DO NOTHING (pixel_date_hash MD5)
      - Migração incremental (ADD COLUMN IF NOT EXISTS) para tabelas legadas
    """

    def __init__(self, config: Dict[str, str] = None):
        self.config  = config or DB_CONFIG
        self.engine  = None
        self.session = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def connect(self) -> None:
        """Estabelece pool de conexões com PostgreSQL."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from urllib.parse import quote_plus

        pwd = quote_plus(self.config['password'])
        url = (
            f"postgresql://{self.config['user']}:{pwd}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
        self.engine  = create_engine(
            url,
            echo=False,
            pool_size=5,         # conexões persistentes no pool
            max_overflow=10,     # extras sob demanda
            pool_pre_ping=True,  # valida conexão antes de usar (evita stale connections)
            pool_recycle=3600,   # recicla conexões ociosas a cada 1h
        )
        self.session = sessionmaker(bind=self.engine)()
        logger.info("Conexão PostgreSQL estabelecida (pool_size=5, max_overflow=10)")

    def create_tables(self) -> None:
        """
        Cria tabelas e índices caso não existam.
        Executa migração incremental (ADD COLUMN IF NOT EXISTS) para instâncias legadas.
        """
        from sqlalchemy import text

        # Schema alinhado ao GPKG de entrada (EPSG:4674).
        # Todas as colunas são NOT NULL com DEFAULT seguro para garantir "sem campos null".
        create_pixel = """
        CREATE TABLE IF NOT EXISTS pixel (
            id          SERIAL PRIMARY KEY,
            geom        TEXT             NOT NULL,
            objectid    BIGINT           NOT NULL DEFAULT 0,
            nr_apm      BIGINT           NOT NULL DEFAULT 0,
            nr_iptu     TEXT             NOT NULL DEFAULT '',
            cd_geo      TEXT             NOT NULL DEFAULT '',
            cd_log      BIGINT           NOT NULL DEFAULT 0,
            nm_log      TEXT             NOT NULL DEFAULT '',
            nr_log      TEXT             NOT NULL DEFAULT '',
            nm_qdr      TEXT             NOT NULL DEFAULT '',
            nm_lot      TEXT             NOT NULL DEFAULT '',
            nm_apm      TEXT             NOT NULL DEFAULT '',
            compl       TEXT             NOT NULL DEFAULT '',
            cd_bai      INTEGER          NOT NULL DEFAULT 0,
            nm_bai      TEXT             NOT NULL DEFAULT '',
            cd_destin   BIGINT           NOT NULL DEFAULT 0,
            nm_destin   TEXT             NOT NULL DEFAULT '',
            cd_grupo_d  BIGINT           NOT NULL DEFAULT 0,
            nm_grupo_d  TEXT             NOT NULL DEFAULT '',
            cd_uso      BIGINT           NOT NULL DEFAULT 0,
            nm_uso      TEXT             NOT NULL DEFAULT '',
            cd_grupo_u  BIGINT           NOT NULL DEFAULT 0,
            nm_grupo_u  TEXT             NOT NULL DEFAULT '',
            area_orig   DOUBLE PRECISION NOT NULL DEFAULT 0,
            area_atual  DOUBLE PRECISION NOT NULL DEFAULT 0,
            area_edif   DOUBLE PRECISION NOT NULL DEFAULT 0,
            cd_tipo     TEXT             NOT NULL DEFAULT '',
            nm_proprie  TEXT             NOT NULL DEFAULT '',
            ci          TEXT             NOT NULL DEFAULT '',
            dg_ferific  INTEGER          NOT NULL DEFAULT 0,
            nr_refiptu  DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at  TIMESTAMP        DEFAULT CURRENT_TIMESTAMP
        );"""

        create_pixel_urban = """
        CREATE TABLE IF NOT EXISTS pixel_urban (
            id              SERIAL PRIMARY KEY,
            date_time       TIMESTAMP        NOT NULL,
            longitude       DOUBLE PRECISION NOT NULL,
            latitude        DOUBLE PRECISION NOT NULL,
            centroide       TEXT             NOT NULL,
            -- 14 features espectrais/texturais
            ndvi            DOUBLE PRECISION NOT NULL DEFAULT 0,
            ndvi_re         DOUBLE PRECISION NOT NULL DEFAULT 0,
            gndvi           DOUBLE PRECISION NOT NULL DEFAULT 0,
            savi            DOUBLE PRECISION NOT NULL DEFAULT 0,
            mndwi           DOUBLE PRECISION NOT NULL DEFAULT 0,
            ndbi            DOUBLE PRECISION NOT NULL DEFAULT 0,
            ui              DOUBLE PRECISION NOT NULL DEFAULT 0,
            nbi             DOUBLE PRECISION NOT NULL DEFAULT 0,
            bsi             DOUBLE PRECISION NOT NULL DEFAULT 0,
            baei            DOUBLE PRECISION NOT NULL DEFAULT 0,
            ibi             DOUBLE PRECISION NOT NULL DEFAULT 0,
            ndvi_std        DOUBLE PRECISION NOT NULL DEFAULT 0,
            entropy         DOUBLE PRECISION NOT NULL DEFAULT 0,
            ndbi_std        DOUBLE PRECISION NOT NULL DEFAULT 0,
            -- classificações duais
            urban_heu       INTEGER          NOT NULL DEFAULT 0,
            urban_pca       INTEGER          NOT NULL DEFAULT 0,
            -- metadados
            periodo         TEXT             NOT NULL DEFAULT '',
            id_pixel        INTEGER          NOT NULL REFERENCES pixel(id),
            pixel_date_hash TEXT             NOT NULL UNIQUE,
            created_at      TIMESTAMP        DEFAULT CURRENT_TIMESTAMP
        );"""

        # Migração incremental — colunas novas no esquema do GPKG e nas features ML.
        # ADD COLUMN ... DEFAULT ... NOT NULL preenche linhas existentes com o default.
        pixel_new_cols = [
            ("objectid",   "BIGINT",           "0"),
            ("nr_apm",     "BIGINT",           "0"),
            ("nr_iptu",    "TEXT",             "''"),
            ("cd_geo",     "TEXT",             "''"),
            ("cd_log",     "BIGINT",           "0"),
            ("nm_log",     "TEXT",             "''"),
            ("nr_log",     "TEXT",             "''"),
            ("nm_qdr",     "TEXT",             "''"),
            ("nm_lot",     "TEXT",             "''"),
            ("nm_apm",     "TEXT",             "''"),
            ("compl",      "TEXT",             "''"),
            ("cd_bai",     "INTEGER",          "0"),
            ("nm_bai",     "TEXT",             "''"),
            ("cd_destin",  "BIGINT",           "0"),
            ("nm_destin",  "TEXT",             "''"),
            ("cd_grupo_d", "BIGINT",           "0"),
            ("nm_grupo_d", "TEXT",             "''"),
            ("cd_uso",     "BIGINT",           "0"),
            ("nm_uso",     "TEXT",             "''"),
            ("cd_grupo_u", "BIGINT",           "0"),
            ("nm_grupo_u", "TEXT",             "''"),
            ("area_orig",  "DOUBLE PRECISION", "0"),
            ("area_atual", "DOUBLE PRECISION", "0"),
            ("area_edif",  "DOUBLE PRECISION", "0"),
            ("cd_tipo",    "TEXT",             "''"),
            ("nm_proprie", "TEXT",             "''"),
            ("ci",         "TEXT",             "''"),
            ("dg_ferific", "INTEGER",          "0"),
            ("nr_refiptu", "DOUBLE PRECISION", "0"),
        ]
        migrations = [
            *[
                f"ALTER TABLE pixel ADD COLUMN IF NOT EXISTS {c} {t} NOT NULL DEFAULT {d};"
                for (c, t, d) in pixel_new_cols
            ],
            *[
                f"ALTER TABLE pixel_urban ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION NOT NULL DEFAULT 0;"
                for col in ['ndvi_re', 'gndvi', 'savi', 'ndbi', 'bsi', 'baei', 'ibi',
                            'ndvi_std', 'entropy', 'ndbi_std']
            ],
            "ALTER TABLE pixel_urban ADD COLUMN IF NOT EXISTS urban_heu INTEGER NOT NULL DEFAULT 0;",
            "ALTER TABLE pixel_urban ADD COLUMN IF NOT EXISTS urban_pca INTEGER NOT NULL DEFAULT 0;",
            "ALTER TABLE pixel_urban ADD COLUMN IF NOT EXISTS pixel_date_hash TEXT;",
        ]

        indexes = """
        CREATE INDEX        IF NOT EXISTS idx_pxu_date     ON pixel_urban(date_time);
        CREATE INDEX        IF NOT EXISTS idx_pxu_pixel    ON pixel_urban(id_pixel);
        CREATE INDEX        IF NOT EXISTS idx_pxu_location ON pixel_urban(longitude, latitude);
        CREATE UNIQUE INDEX IF NOT EXISTS uidx_pxu_hash   ON pixel_urban(pixel_date_hash);
        """

        with self.engine.connect() as conn:
            conn.execute(text(create_pixel))
            conn.execute(text(create_pixel_urban))
            for step in migrations:
                conn.execute(text(step))
            conn.execute(text(indexes))
            conn.commit()

        logger.info("Tabelas, colunas e índices verificados/atualizados")

    def save_pixel_data(self, df: pd.DataFrame, periodo: str = 'QUINZENAL') -> int:
        """
        Persiste batch completo (14 features + urban_heu + urban_pca) no PostgreSQL.

        Estratégia:
          1. Lookup em lote de pixels existentes (tabela pai)
          2. INSERT individual apenas para pixels novos
          3. Bulk INSERT em chunks (DB_INSERT_CHUNK) na tabela pixel_urban
          4. ON CONFLICT (pixel_date_hash) DO NOTHING — idempotência garantida

        Args:
            df:      DataFrame com features + urban_heu + urban_pca já calculados
            periodo: Label do período (ex: 'QUINZENAL')

        Returns:
            Número de registros enviados ao banco
        """
        from sqlalchemy import text

        if df.empty:
            logger.warning("DataFrame vazio, nada a salvar")
            return 0

        # ── 1. Lookup em lote de pixels existentes ─────────────────────────
        # SRID=4326 — pontos foram reprojetados (EPSG:4674 → EPSG:4326) antes do GEE.
        geoms       = [f"SRID=4326;POINT({r['longitude']} {r['latitude']})" for _, r in df.iterrows()]
        geom_params = {f"g{i}": g for i, g in enumerate(geoms)}
        placeholders = ', '.join(f":g{i}" for i in range(len(geoms)))

        existing  = self.session.execute(
            text(f"SELECT id, geom::text FROM pixel WHERE geom IN ({placeholders})"),
            geom_params,
        ).fetchall()
        pixel_map = {row[1]: row[0] for row in existing}

        # ── 2. Inserir pixels novos (tabela pai) com schema do GPKG ───────
        urban_records: List[Dict] = []

        insert_pixel_sql = text("""
            INSERT INTO pixel (
                geom, objectid, nr_apm, nr_iptu, cd_geo, cd_log,
                nm_log, nr_log, nm_qdr, nm_lot, nm_apm, compl,
                cd_bai, nm_bai, cd_destin, nm_destin,
                cd_grupo_d, nm_grupo_d, cd_uso, nm_uso, cd_grupo_u, nm_grupo_u,
                area_orig, area_atual, area_edif,
                cd_tipo, nm_proprie, ci, dg_ferific, nr_refiptu
            ) VALUES (
                :geom, :objectid, :nr_apm, :nr_iptu, :cd_geo, :cd_log,
                :nm_log, :nr_log, :nm_qdr, :nm_lot, :nm_apm, :compl,
                :cd_bai, :nm_bai, :cd_destin, :nm_destin,
                :cd_grupo_d, :nm_grupo_d, :cd_uso, :nm_uso, :cd_grupo_u, :nm_grupo_u,
                :area_orig, :area_atual, :area_edif,
                :cd_tipo, :nm_proprie, :ci, :dg_ferific, :nr_refiptu
            ) RETURNING id
        """)

        for _, row in df.iterrows():
            geom_wkt = f"SRID=4326;POINT({row['longitude']} {row['latitude']})"
            pixel_id = pixel_map.get(geom_wkt)

            if not pixel_id:
                result = self.session.execute(
                    insert_pixel_sql,
                    {
                        "geom":       geom_wkt,
                        "objectid":   _i(row.get('objectid')),
                        "nr_apm":     _i(row.get('nr_apm')),
                        "nr_iptu":    _s(row.get('nr_iptu')),
                        "cd_geo":     _s(row.get('cd_geo')),
                        "cd_log":     _i(row.get('cd_log')),
                        "nm_log":     _s(row.get('nm_log')),
                        "nr_log":     _s(row.get('nr_log')),
                        "nm_qdr":     _s(row.get('nm_qdr')),
                        "nm_lot":     _s(row.get('nm_lot')),
                        "nm_apm":     _s(row.get('nm_apm')),
                        "compl":      _s(row.get('compl')),
                        "cd_bai":     _i(row.get('cd_bai')),
                        "nm_bai":     _s(row.get('nm_bai')),
                        "cd_destin":  _i(row.get('cd_destin')),
                        "nm_destin":  _s(row.get('nm_destin')),
                        "cd_grupo_d": _i(row.get('cd_grupo_d')),
                        "nm_grupo_d": _s(row.get('nm_grupo_d')),
                        "cd_uso":     _i(row.get('cd_uso')),
                        "nm_uso":     _s(row.get('nm_uso')),
                        "cd_grupo_u": _i(row.get('cd_grupo_u')),
                        "nm_grupo_u": _s(row.get('nm_grupo_u')),
                        "area_orig":  _f(row.get('area_orig')),
                        "area_atual": _f(row.get('area_atual')),
                        "area_edif":  _f(row.get('area_edif')),
                        "cd_tipo":    _s(row.get('cd_tipo')),
                        "nm_proprie": _s(row.get('nm_proprie')),
                        "ci":         _s(row.get('ci')),
                        "dg_ferific": _i(row.get('dg_ferific')),
                        "nr_refiptu": _f(row.get('nr_refiptu')),
                    },
                )
                pixel_id = result.fetchone()[0]
                pixel_map[geom_wkt] = pixel_id

            hash_str        = f"{pixel_id}_{row.get('target_date', datetime.now().strftime('%Y-%m-%d'))}"
            pixel_date_hash = hashlib.md5(hash_str.encode()).hexdigest()

            urban_records.append({
                "date_time":  row.get('date_time', datetime.now()),
                "longitude":  _f(row['longitude']),
                "latitude":   _f(row['latitude']),
                "centroide":  _s(row.get('centroide')),
                # 14 features espectrais/texturais — COALESCE para 0.0
                "ndvi":     _f(row.get('ndvi')),     "ndvi_re": _f(row.get('ndvi_re')),
                "gndvi":    _f(row.get('gndvi')),    "savi":    _f(row.get('savi')),
                "mndwi":    _f(row.get('mndwi')),    "ndbi":    _f(row.get('ndbi')),
                "ui":       _f(row.get('ui')),       "nbi":     _f(row.get('nbi')),
                "bsi":      _f(row.get('bsi')),      "baei":    _f(row.get('baei')),
                "ibi":      _f(row.get('ibi')),
                "ndvi_std": _f(row.get('ndvi_std')), "entropy": _f(row.get('entropy')),
                "ndbi_std": _f(row.get('ndbi_std')),
                # classificações duais — None (heurística sem dados) vira 0
                "urban_heu": _i(row.get('urban_heu')),
                "urban_pca": _i(row.get('urban_pca')),
                # metadados
                "periodo":          _s(periodo),
                "id_pixel":         pixel_id,
                "pixel_date_hash":  pixel_date_hash,
            })

        # ── 3. Bulk INSERT em chunks (evita parâmetros demais por query) ───
        insert_sql = text("""
            INSERT INTO pixel_urban (
                date_time, longitude, latitude, centroide,
                ndvi, ndvi_re, gndvi, savi, mndwi, ndbi, ui, nbi,
                bsi, baei, ibi, ndvi_std, entropy, ndbi_std,
                urban_heu, urban_pca,
                periodo, id_pixel, pixel_date_hash
            ) VALUES (
                :date_time, :longitude, :latitude, :centroide,
                :ndvi, :ndvi_re, :gndvi, :savi, :mndwi, :ndbi, :ui, :nbi,
                :bsi, :baei, :ibi, :ndvi_std, :entropy, :ndbi_std,
                :urban_heu, :urban_pca,
                :periodo, :id_pixel, :pixel_date_hash
            )
            ON CONFLICT (pixel_date_hash) DO NOTHING
        """)

        try:
            for chunk_start in range(0, len(urban_records), DB_INSERT_CHUNK):
                chunk = urban_records[chunk_start:chunk_start + DB_INSERT_CHUNK]
                self.session.execute(insert_sql, chunk)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.error(f"Erro no bulk insert: {e}")
            raise

        logger.info(f"{len(urban_records)} registros enviados (ON CONFLICT DO NOTHING ativo)")
        return len(urban_records)

    def close(self) -> None:
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Conexão com banco fechada")

# ─── Batch Worker (thread-safe) ───────────────────────────────────────────────

def _process_batch(
    processor: PixelProcessor,
    batch_df: pd.DataFrame,
    batch_num: int,
    total_batches: int,
    target_date: str,
    checkpoint_dir: Path,
) -> Optional[pd.DataFrame]:
    """
    Wrapper thread-safe: extração GEE + checkpoint para um lote.

    Retorna DataFrame com as 14 features (SEM classificações).
    Classificações são aplicadas na thread principal após o retorno.
    """
    label           = f"BATCH {batch_num}/{total_batches}"
    checkpoint_file = checkpoint_dir / f"checkpoint_batch_{batch_num}_{target_date}.parquet"

    if checkpoint_file.exists():
        logger.info(f"{label}: Checkpoint encontrado — lendo do disco")
        return pd.read_parquet(checkpoint_file)

    result = processor.extract_indices_batch(batch_df, batch_label=label)
    if result is None or result.empty:
        logger.warning(f"{label}: GEE retornou vazio")
        return None

    # Arredondamento de precisão antes de salvar no checkpoint
    for col in ML_FEATURES:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').round(6)

    result.to_parquet(checkpoint_file, index=False)
    logger.info(f"{label}: Checkpoint salvo em {checkpoint_file}")
    return result

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_urban_detection(
    gpkg_path: str = None,
    target_date: str = None,
    periodo: str = 'QUINZENAL',
    save_to_db: bool = True,
    ml_pipeline=None,
    best_threshold: float = None,
) -> Dict[str, Any]:
    """
    Pipeline completo: extração GEE → classificação dupla → PostgreSQL.

    Parâmetros ml_pipeline e best_threshold podem ser pré-carregados externamente
    para evitar I/O repetido em processamentos retroativos multi-data.

    Args:
        gpkg_path:      Caminho para o GeoPackage
        target_date:    Data alvo (YYYY-MM-DD)
        periodo:        Label do período (ex: 'QUINZENAL')
        save_to_db:     Persistir no PostgreSQL
        ml_pipeline:    Pipeline scikit-learn pré-carregado (opcional)
        best_threshold: Threshold de decisão pré-carregado (opcional)

    Returns:
        Dicionário com estatísticas da execução
    """
    run_id = f"urban_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    logger.info("=" * 70)
    logger.info("URBAN MASK DETECTION (Dual Model: Heurístico + PCA/RF)")
    logger.info("=" * 70)
    logger.info(f"Run ID              : {run_id}")
    logger.info(f"Intervalo execução  : {EXECUTION_INTERVAL_DAYS} dias")
    logger.info(f"Batch size          : {BATCH_SIZE:,} pontos/submissão")
    logger.info(f"Concurrent batches  : {MAX_CONCURRENT_BATCHES}")
    logger.info(f"Tile scale GEE      : {TILE_SCALE}")
    logger.info("=" * 70)

    stats: Dict[str, Any] = {
        'run_id':     run_id,
        'start_time': datetime.now().isoformat(),
        'gpkg_path':  gpkg_path or DEFAULT_GPKG_PATH,
        'target_date': target_date,
        'periodo':    periodo,
        'total_polygons': 0,
        'total_pixels':   0,
        'records_saved':  0,
        'status':    'running',
    }

    try:
        # ── [1/5] Earth Engine ────────────────────────────────────────────
        logger.info("[1/5] Inicializando Google Earth Engine")
        initialize_earth_engine()
        execution_timestamp = datetime.now()

        # ── [2/5] Artefatos ML ────────────────────────────────────────────
        logger.info("[2/5] Carregando artefatos do modelo ML")
        if ml_pipeline is None or best_threshold is None:
            ml_pipeline, best_threshold = load_ml_artifacts()

        # ── [3/5] Grade de Pixels ─────────────────────────────────────────
        logger.info("[3/5] Carregando grade fixa de pixels")
        processor = PixelProcessor(
            gpkg_path=gpkg_path or DEFAULT_GPKG_PATH,
            target_date=target_date,
            execution_timestamp=execution_timestamp,
        )
        points_df    = processor.load_or_generate_points()
        total_pixels = len(points_df)
        stats['total_pixels_gerados'] = total_pixels

        for col in ['ci', 'id_lote', 'inscricao_']:
            if col in points_df.columns:
                stats['total_polygons'] = int(points_df[col].nunique())
                break

        logger.info(f"Polígonos únicos: {stats['total_polygons']:,}")

        # ── [4/5] Extração GEE (paralela) + Classificação ─────────────────
        logger.info("[4/5] Extração GEE paralela + Classificação Dupla")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        total_batches = (total_pixels + BATCH_SIZE - 1) // BATCH_SIZE

        db = None
        if save_to_db:
            db = DatabaseManager()
            db.connect()
            db.create_tables()

        total_processed = 0
        total_saved     = 0

        # Prepara todos os lotes
        batch_slices = [
            (points_df.iloc[s: min(s + BATCH_SIZE, total_pixels)].copy(), (s // BATCH_SIZE) + 1)
            for s in range(0, total_pixels, BATCH_SIZE)
        ]

        # ThreadPoolExecutor: GEE extraction em paralelo; DB save na thread principal
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_BATCHES) as executor:
            future_map = {
                executor.submit(
                    _process_batch,
                    processor, batch_df, batch_num, total_batches,
                    target_date, CHECKPOINT_DIR,
                ): batch_num
                for batch_df, batch_num in batch_slices
            }

            for future in as_completed(future_map):
                batch_num = future_map[future]
                label     = f"BATCH {batch_num}/{total_batches}"

                try:
                    batch_result = future.result()
                except Exception as exc:
                    logger.error(f"{label}: Falha na extração GEE: {exc}")
                    continue

                if batch_result is None or batch_result.empty:
                    continue

                # ── Classificação Heurística (urban_heu) ─────────────────
                batch_result['urban_heu'] = batch_result.apply(
                    lambda row: urban_classification_logic({
                        'ui':    row.get('ui'),
                        'nbi':   row.get('nbi'),
                        'ndvi':  row.get('ndvi'),
                        'mndwi': row.get('mndwi'),
                    }),
                    axis=1,
                )

                # ── Classificação ML (urban_pca) ──────────────────────────
                batch_result['urban_pca'] = classify_urban_ml(
                    batch_result, ml_pipeline, best_threshold
                )

                n = len(batch_result)
                logger.info(
                    f"{label}: {n:,} pixels | "
                    f"urban_heu=1: {int(batch_result['urban_heu'].sum()):,} | "
                    f"urban_pca=1: {int(batch_result['urban_pca'].sum()):,}"
                )

                total_processed += n

                if save_to_db and db:
                    saved = db.save_pixel_data(batch_result, periodo)
                    total_saved += saved
                    logger.info(f"{label}: {saved:,} registros persistidos")

                del batch_result
                gc.collect()

        if db:
            db.close()

        # ── [5/5] Sumário ─────────────────────────────────────────────────
        stats.update({
            'status':       'success',
            'end_time':     datetime.now().isoformat(),
            'total_pixels': total_processed,
            'records_saved': total_saved,
        })

        logger.info("=" * 70)
        logger.info("PROCESSAMENTO CONCLUÍDO")
        logger.info(f"Run ID              : {run_id}")
        logger.info(f"Polígonos           : {stats['total_polygons']:,}")
        logger.info(f"Pixels analisados   : {total_processed:,}")
        logger.info(f"Registros salvos    : {total_saved:,}")
        logger.info(f"Próxima execução    : {EXECUTION_INTERVAL_DAYS} dias")
        logger.info("=" * 70)

        return stats

    except Exception as e:
        stats.update({
            'status':   'error',
            'error':    str(e),
            'end_time': datetime.now().isoformat(),
        })
        logger.error(f"Erro na execução: {e}")
        import traceback
        traceback.print_exc()
        raise

# ─── Utilities ────────────────────────────────────────────────────────────────

def test_database_connection() -> bool:
    """Testa conexão com PostgreSQL. Retorna True se bem-sucedida."""
    from sqlalchemy import text
    logger.info("Testando conexão com PostgreSQL")
    try:
        db = DatabaseManager()
        db.connect()
        result = db.session.execute(text("SELECT version();")).fetchone()
        logger.info(f"PostgreSQL: {result[0].split(',')[0]}")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Erro na conexão: {e}")
        logger.info("Verifique: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME")
        return False


def print_instructions():
    """Imprime instruções de uso."""
    print(f"""
{'=' * 70}
URBAN MASK DETECTION (Dual Model)
{'=' * 70}

USO VIA LINHA DE COMANDO:
  python urbanMask.py --gpkg /path/to/file.gpkg --periodo QUINZENAL
  python urbanMask.py --start-date 2024-01-01 --end-date 2024-12-31

USO COMO MÓDULO:
  from urbanMask import run_urban_detection, load_ml_artifacts
  pipeline, threshold = load_ml_artifacts()           # carrega uma vez
  stats = run_urban_detection(                         # reutiliza artefatos
      gpkg_path='/path/to/file.gpkg',
      ml_pipeline=pipeline,
      best_threshold=threshold
  )

CONFIGURAÇÕES (via variáveis de ambiente):
  BATCH_SIZE             = {BATCH_SIZE:,}     pontos por submissão ao GEE
  GEE_CONCURRENT_BATCHES = {MAX_CONCURRENT_BATCHES}       extração paralela de lotes
  GEE_TILE_SCALE         = {TILE_SCALE}       tileScale do sampleRegions
  DB_INSERT_CHUNK        = {DB_INSERT_CHUNK:,}    linhas por executemany
  MODEL_DIR              = {MODEL_DIR}

MODELO ML:
  {MODEL_DIR}/pipeline_final_cv.joblib
  {MODEL_DIR}/best_threshold_cv.pkl

BANCO DE DADOS:
  Credenciais via Kubernetes Secret → DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

TESTAR CONEXÃO:
  python -c "from urbanMask import test_database_connection; test_database_connection()"

KUBEFLOW CRON:
  0 0 */{EXECUTION_INTERVAL_DAYS} * *
{'=' * 70}
""")

# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Urban Mask Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--gpkg',       type=str, default=None,
                        help='Caminho para GeoPackage')
    parser.add_argument('--date',       type=str, default=None,
                        help='Data alvo execução única (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Data inicial para processamento retroativo (YYYY-MM-DD)')
    parser.add_argument('--end-date',   type=str, default=None,
                        help='Data final para processamento retroativo (YYYY-MM-DD)')
    parser.add_argument('--periodo',    type=str, default='QUINZENAL',
                        help='Período da análise')
    parser.add_argument('--no-db',      action='store_true',
                        help='Não salvar no banco')
    parser.add_argument('--test-db',    action='store_true',
                        help='Testar conexão com banco')
    parser.add_argument('--help-full',  action='store_true',
                        help='Instruções completas')

    args = parser.parse_args()

    if args.help_full:
        print_instructions()

    elif args.test_db:
        test_database_connection()

    elif args.start_date:
        # Processamento retroativo: itera datas no intervalo
        start_dt      = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date_str  = args.end_date or datetime.now().strftime('%Y-%m-%d')
        end_dt        = datetime.strptime(end_date_str, '%Y-%m-%d')

        if start_dt > end_dt:
            logger.error(
                f"--start-date ({args.start_date}) posterior a --end-date ({end_date_str})"
            )
            sys.exit(1)

        logger.info(
            f"Processamento retroativo: {args.start_date} → {end_date_str} "
            f"(intervalo: {EXECUTION_INTERVAL_DAYS} dias)"
        )

        # Carrega GEE e artefatos ML UMA VEZ fora do loop
        initialize_earth_engine()
        ml_pipeline, best_threshold = load_ml_artifacts()

        current_dt = start_dt
        resultados = []

        while current_dt <= end_dt:
            target_str = current_dt.strftime('%Y-%m-%d')
            logger.info(f"=== ITERAÇÃO: {target_str} ===")
            try:
                result = run_urban_detection(
                    gpkg_path=args.gpkg,
                    target_date=target_str,
                    periodo=args.periodo,
                    save_to_db=not args.no_db,
                    ml_pipeline=ml_pipeline,
                    best_threshold=best_threshold,
                )
                resultados.append({
                    'data':         target_str,
                    'status':       'success',
                    'linhas_salvas': result.get('records_saved', 0),
                })
            except Exception as e:
                logger.error(f"Falha em {target_str}: {e}")
                resultados.append({'data': target_str, 'status': 'error', 'erro': str(e)})

            current_dt += timedelta(days=EXECUTION_INTERVAL_DAYS)

        logger.info("=== RESUMO RETROATIVO ===")
        for r in resultados:
            info = (
                f"Salvos: {r.get('linhas_salvas'):,}"
                if r['status'] == 'success'
                else f"Erro: {r.get('erro')}"
            )
            logger.info(f"  {r['data']} | {r['status']} | {info}")

    else:
        if args.gpkg is None:
            logger.warning(f"Usando GPKG padrão: {DEFAULT_GPKG_PATH}")

        run_urban_detection(
            gpkg_path=args.gpkg,
            target_date=args.date,
            periodo=args.periodo,
            save_to_db=not args.no_db,
        )
