# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Centro de Excelência em Inteligência Artificial (CEIA), Universidade Federal de Goiás
"""Passo 1 do treino: extrai, para cada polígono e cada ano, os 14 índices Sentinel-2 e o rótulo do Google Open Buildings
Temporal (presença >= 0,56 e altura >= 2 m), um registro por pixel de 10 m, salvando CSVs de 100 polígonos.
Portado do notebook script_colab.ipynb (Google Colab) para linha de comando.

Uso:
  python training/01_extract_pixels_gee.py --gpkg poligonos.gpkg --out ./batches_pixel --ee-project SEU_PROJETO
Requer: earthengine-api autenticado (ee.Authenticate()), geopandas, pandas, numpy.
"""
import argparse, os
import ee
import geopandas as gpd
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--gpkg", required=True, help="GeoPackage com os polígonos (lotes ou APMs)")
ap.add_argument("--out", required=True, help="pasta de saída dos CSVs (batch_N.csv)")
ap.add_argument("--ee-project", required=True, help="projeto do Google Earth Engine")
ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023])
ap.add_argument("--save-every", type=int, default=100, help="polígonos por arquivo")
args = ap.parse_args()

ee.Initialize(project=args.ee_project, opt_url="https://earthengine-highvolume.googleapis.com")
apms_polygons = gpd.read_file(args.gpkg)
for col in apms_polygons.select_dtypes(include=["datetime"]).columns:
    apms_polygons[col] = apms_polygons[col].dt.strftime("%Y-%m-%d %H:%M:%S")
if apms_polygons.crs.to_epsg() != 4326:
    apms_polygons = apms_polygons.to_crs(epsg=4326)

# IMPORTANTE: se o GeoPackage tiver linhas repetidas (mesma geometria), os mesmos pixels serão extraídos várias vezes.
n_rep = int(apms_polygons.geometry.to_wkb().duplicated().sum())
if n_rep:
    print(f"ATENÇÃO: {n_rep} linhas com geometria repetida; considere drop_duplicates antes da extração.")

OUTPUT_PATH = args.out
YEARS = args.years
SCALE = 10
SAVE_EVERY_N_POLYGONS = args.save_every
METADATA_COLS = [col for col in apms_polygons.columns if col != "geometry"]
os.makedirs(f"{OUTPUT_PATH}/batches_pixel", exist_ok=True)

# 1. FUNÇÕES DE SUPORTE
# ==========================================

def cloudMaskS2(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11', 'B12'],
                        ['Blue', 'Green', 'Red', 'RE1', 'NIR', 'SWIR1', 'SWIR2']) \
                .updateMask(mask).divide(10000)

def limpar_e_estilizar(df):
    if df.empty: return df

    # Índices para arredondamento de precisão
    cols_float = [
        'ndvi', 'ndvi_re', 'gndvi', 'savi', 'mndwi', 'ndbi',
        'ui', 'nbi', 'bsi', 'baei', 'ibi', 'ndvi_std', 'entropy',
        'google_conf', 'google_height'
    ]

    for col in cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(5)

    if 'contrast' in df.columns:
        df['contrast'] = pd.to_numeric(df['contrast'], errors='coerce').round(2)

    df['centroide'] = 'POINT(' + df['long'].astype(str) + ' ' + df['lat'].astype(str) + ')'
    return df

# ==========================================
# 2. FUNÇÃO PRINCIPAL: EXTRAÇÃO PIXEL A PIXEL
# ==========================================

def get_data_for_polygon(row, years):
    geom = row['geometry']
    ee_geom = ee.Geometry(geom.__geo_interface__)
    dfs = []

    for year in years:
        try:
            start_date, end_date = f'{year}-01-01', f'{year}-12-31'

            # --- A. Sentinel-2 (Coleção e Estatísticas) ---
            s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                .filterBounds(ee_geom).filterDate(start_date, end_date) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(cloudMaskS2)

            if s2_col.size().getInfo() == 0: continue

            # NDVI Temporal (Desvio Padrão)
            ndvi_std = s2_col.map(lambda img: img.normalizedDifference(['NIR', 'Red'])).reduce(ee.Reducer.stdDev()).rename('ndvi_std')
            ndbi_std = s2_col.map(lambda img: img.normalizedDifference(['SWIR1', 'NIR'])).reduce(ee.Reducer.stdDev()).rename('ndbi_std')

            # Imagem Mediana para índices fixos
            s2 = s2_col.median().clip(ee_geom)
            blue = s2.select('Blue')
            green = s2.select('Green')
            red = s2.select('Red')
            re1 = s2.select('RE1')
            nir = s2.select('NIR')
            swir1 = s2.select('SWIR1')
            swir2 = s2.select('SWIR2')

            # --- B. Textura (GLCM) ---
            glcm = nir.multiply(10000).toInt().glcmTexture(size=3)
            contrast = glcm.select('NIR_contrast').rename('contrast')
            entropy = glcm.select('NIR_ent').rename('entropy')

            # --- C. Índices Espectrais ---
            ndvi = s2.normalizedDifference(['NIR', 'Red']).rename('ndvi')
            ndvire = s2.normalizedDifference(['NIR', 'RE1']).rename('ndvi_re')
            gndvi = s2.normalizedDifference(['NIR', 'Green']).rename('gndvi')
            mndwi = s2.normalizedDifference(['Green', 'SWIR1']).rename('mndwi')
            ndbi = s2.normalizedDifference(['SWIR1', 'NIR']).rename('ndbi')
            ui = s2.normalizedDifference(['SWIR2', 'NIR']).rename('ui')

            savi = s2.expression('((NIR - RED) * 1.5) / (NIR + RED + 0.5)',
                                {'NIR': nir, 'RED': red}).rename('savi')

            nbi = s2.expression('(RED * SWIR1) / NIR',
                                {'RED': red, 'SWIR1': swir1, 'NIR': nir}).rename('nbi')

            bsi = s2.expression('((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
                                {'SWIR1': swir1, 'RED': red, 'NIR': nir, 'BLUE': blue}).rename('bsi')

            baei = s2.expression('(RED + 0.3) / (GREEN + SWIR1)',
                                {'RED': red, 'GREEN': green, 'SWIR1': swir1}).rename('baei')

            ibi = s2.expression('(NDBI - (SAVI + MNDWI) / 2) / (NDBI + (SAVI + MNDWI) / 2)',
                                {'NDBI': ndbi, 'SAVI': savi, 'MNDWI': mndwi}).rename('ibi')

            # --- D. Google Open Buildings ---
            google_col = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1') \
                .filterBounds(ee_geom).filterDate(start_date, end_date)

            has_google = google_col.size().gt(0)
            google_mosaic = google_col.mosaic().select(['building_presence', 'building_height'])

            #  Confiança >= 0.56 E Altura >= 2m
            target_refined = google_mosaic.select('building_presence').gte(0.56) \
                .And(google_mosaic.select('building_height').gte(2.0)).rename('google_target')

            # --- E. Empilhamento e Extração ---
            stack = ee.Image.cat([
                target_refined,
                google_mosaic.select('building_presence').rename('google_conf'),
                google_mosaic.select('building_height').rename('google_height'),
                ndvi, ndvire, gndvi, savi, mndwi, ndbi, ui, nbi, bsi, baei, ibi,
                contrast, entropy, ndvi_std, ndbi_std,
                ee.Image.pixelLonLat().rename(['long', 'lat'])
            ])

            raw_data = stack.reduceRegion(
                reducer=ee.Reducer.toList(), geometry=ee_geom, scale=SCALE,
                maxPixels=1e8, tileScale=4
            ).getInfo()

            if raw_data.get('long') is not None:
                df_year = pd.DataFrame(raw_data)
                df_year['google_target'] = df_year['google_target'].replace(-1, np.nan)
                df_year['ano'] = year
                df_year['pixel_id'] = range(1, len(df_year) + 1)

                # Adiciona metadados do polígono original
                for col in METADATA_COLS:
                    df_year[col] = row.get(col, None)

                dfs.append(df_year)

        except Exception as e:
            print(f"Erro Polígono {row.name} Ano {year}: {e}")
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def limpar_e_estilizar(df):
    if df.empty: return df

    if 'urban_mask' in df.columns:
        df['urban_mask'] = df['urban_mask'].fillna(0).astype(int)

    # Lista de todos os índices para arredondamento
    cols_float = [
        'ndvi', 'ndvi_re', 'gndvi', 'savi', 'mndwi', 'ndbi',
        'ui', 'nbi', 'bsi', 'baei', 'ibi', 'ndvi_std', 'entropy', 'ndbi_std'
    ]

    for col in cols_float:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(5)

    # O contraste costuma ter valores altos, arredondamos para 2 casas
    if 'contrast' in df.columns:
        df['contrast'] = pd.to_numeric(df['contrast'], errors='coerce').round(2)

    df['centroide'] = 'POINT(' + df['long'].astype(str) + ' ' + df['lat'].astype(str) + ')'
    return df

# ==========================================
# 3. LOOP PRINCIPAL
# ==========================================
current_batch_data = []

print("="*70)
print(f"INICIANDO PROCESSAMENTO: TODOS OS ÍNDICES + TEXTURA")
print("="*70)

for idx, row in apms_polygons.iterrows():
    current_batch_number = idx // SAVE_EVERY_N_POLYGONS
    batch_filename = f'{OUTPUT_PATH}/batches_pixel/batch_{current_batch_number}.csv'

    if os.path.exists(batch_filename):
        continue

    print(f"Processando: {idx} | Lote: {row.name}", end="\r")

    df_poly = get_data_for_polygon(row, YEARS)

    if not df_poly.empty:
        df_clean = limpar_e_estilizar(df_poly)
        current_batch_data.append(df_clean)

    is_last = (idx == len(apms_polygons) - 1)
    if (len(current_batch_data) > 0) and ((idx + 1) % SAVE_EVERY_N_POLYGONS == 0 or is_last):
        df_concat = pd.concat(current_batch_data, ignore_index=True)
        df_concat.to_csv(batch_filename, index=False)
        print(f"\nBatch {current_batch_number} salvo! ({len(df_concat)} pixels)")
        current_batch_data = []

print("\nProcesso Finalizado.")