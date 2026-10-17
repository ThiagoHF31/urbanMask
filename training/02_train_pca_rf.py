# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The UrbanMask authors
"""Passo 2 do treino: consolida os CSVs do passo 1, treina StandardScaler -> PCA(95%) -> RandomForest(100, prof. 12)
e escolhe o limiar de decisão que maximiza o F1 (média de 5 partições). Salva pipeline_final_cv.joblib e best_threshold_cv.pkl.
Portado do notebook modeloFinal.ipynb.

Uso:  python training/02_train_pca_rf.py --input ./batches_pixel --out ./model
Atenção: divisão aleatória por pixel (como no original). Para uma avaliação que respeite polígonos, anos ou blocos
geográficos, veja docs/evaluation.md.
"""
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="pasta com os batch_*.csv do passo 1")
ap.add_argument("--out", default="./model", help="pasta de saída dos artefatos")
args = ap.parse_args()
# Importação de Bibliotecas
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
import joblib
import warnings

# Sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONSOLIDAÇÃO DOS DADOS
# ============================================================================
PATH_ENTRADA = args.input
arquivos_csv = glob.glob(os.path.join(PATH_ENTRADA, "*.csv"))

if not arquivos_csv:
    print("Nenhum arquivo CSV encontrado.")
else:
    print(f"Lendo {len(arquivos_csv)} arquivos...")
    lista_dfs = [pd.read_csv(f) for f in tqdm(arquivos_csv, desc="Processando")]
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    print(f"Total de linhas: {len(df_completo):,}")


# ============================================================================
# 2. PREPARAÇÃO DE VARIÁVEIS (CRÍTICO)
# ============================================================================
# Defina aqui exatamente os nomes das suas colunas
features = ['ndvi', 'ndvi_re', 'gndvi', 'savi', 'mndwi', 'ndbi',
            'ui', 'nbi', 'bsi', 'baei', 'ibi', 'ndvi_std', 'entropy', 'ndbi_std']
target = 'google_target'

# Criando X e y e tratando valores nulos/infinitos
X = df_completo[features].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df_completo[target].astype(int)


# ============================================================================
# 3. SEPARAÇÃO DO TESTE FINAL (HOLD-OUT)
# ============================================================================
X_train_cv, X_test, y_train_cv, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ============================================================================
# 4. CROSS-VALIDATION (5-FOLD)
# ============================================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_thresholds = []

print(f"Iniciando Cross-Validation em {X_train_cv.shape[0]:,} registros...")

for i, (train_index, val_index) in enumerate(skf.split(X_train_cv, y_train_cv)):
    X_tr, X_val = X_train_cv.iloc[train_index], X_train_cv.iloc[val_index]
    y_tr, y_val = y_train_cv.iloc[train_index], y_train_cv.iloc[val_index]

    fold_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42))
    ])

    fold_pipeline.fit(X_tr, y_tr)

    probs_val = fold_pipeline.predict_proba(X_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, probs_val)
    f1_s = 2 * prec * rec / (prec + rec + 1e-10)

    best_t = thresholds[np.argmax(f1_s)]
    fold_thresholds.append(best_t)
    print(f"   Fold {i+1}: Threshold = {best_t:.3f}")

final_threshold = np.mean(fold_thresholds)
print(f"\nThreshold Médio: {final_threshold:.3f}")

# ============================================================================
# 5. MODELO FINAL E SALVAMENTO
# ============================================================================
print("Treinando modelo final...")
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42))
])
final_pipeline.fit(X_train_cv, y_train_cv)

# Avaliação
y_probs_test = final_pipeline.predict_proba(X_test)[:, 1]
y_pred_test = (y_probs_test >= final_threshold).astype(int)

print("\nRELATÓRIO FINAL")
print(classification_report(y_test, y_pred_test))
print(f"ROC-AUC: {roc_auc_score(y_test, y_probs_test):.4f}")

import os; os.makedirs(args.out, exist_ok=True)
joblib.dump(final_pipeline, os.path.join(args.out, 'pipeline_final_cv.joblib'))
joblib.dump(final_threshold, os.path.join(args.out, 'best_threshold_cv.pkl'))
print(f"\nPROCESSO CONCLUÍDO!")
