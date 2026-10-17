# Modelo PCA + Random Forest

| Item | Valor |
|---|---|
| Arquivos | `pipeline_final_cv.joblib` (scikit-learn 1.6.1), `best_threshold_cv.pkl` |
| Entrada | 14 variáveis (ordem em `ML_FEATURES` de `urbanMask.py`) |
| Pipeline | `StandardScaler` → `PCA(n_components=0.95)` (6 componentes, 95,9 % da variância) → `RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)`; sem `class_weight` |
| Limiar | 0,291 = média de cinco partições (0,295; 0,277; 0,289; 0,303; 0,291), cada uma o ponto que maximiza o F1 na curva precisão-recall |
| Rótulo de treino | Google Open Buildings Temporal v1: presença ≥ 0,56 **e** altura ≥ 2 m (remove solo compactado, calçada e entulho) |
| Dados de treino | pixels de 2020–2023 de polígonos de lotes vagos de um município (13.732.832 registros pixel-ano; 10.986.265 para treino e validação cruzada, 2.746.567 reservados para teste) |
| Composição no treino | mediana **anual** (nuvens < 20 %); em produção a composição é uma janela móvel de 150 dias (nuvens < 30 %) |
| Pressuposto | "uma construção é uma construção, independentemente da base": treinou-se em lotes vagos para detectar em APMs |
