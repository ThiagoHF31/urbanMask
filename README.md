# UrbanMask

Desenvolvido pelo CEIA/UFG no projeto *Inteligência Artificial Aplicada para promover uma Gestão eficiente do Dinamismo do Município de Goiânia* (SEFAZ e SEPLAN, Prefeitura de Goiânia), com os resultados exibidos na plataforma ScanUrb Gyn.

Pipeline em lote para detecção e monitoramento de novas construções em áreas públicas e lotes vagos a partir de imagens Sentinel-2, com dupla classificação: uma regra espectral (heurística) e um modelo de aprendizado de máquina (PCA + Random Forest). Pensado para execução quinzenal em Kubernetes.

> **English summary.** UrbanMask is a batch pipeline that reprocesses Sentinel-2 imagery every 15 days over a set of polygons, flags new constructions at 10 m pixel level with two classifiers (a four-index threshold rule and PCA + Random Forest), keeps the cadastral key of each pixel end to end, and stores one PostgreSQL row per pixel and date. Code: Apache-2.0. The documentation below is in Portuguese; see also `docs/`.

## Arquitetura

```
GeoPackage (.gpkg)
      |
      v
Grade de pixels de 10 m (cache .parquet)
      |
      v
Google Earth Engine: extração paralela (ThreadPoolExecutor)
      |   14 variáveis espectrais, de textura e temporais por pixel
      v
+----------------------------------+
|  Classificação dupla             |
|  - urban_heu  (regra espectral)  |
|  - urban_pca  (PCA + RF)         |
+----------------------------------+
      |
      v
PostgreSQL (pixel + pixel_urban)
```

1. **Ingestão e grade:** cada polígono vira uma grade de pontos de cerca de 10 m (polígonos muito pequenos usam um ponto representativo). A grade é guardada em parquet e reutilizada enquanto o GeoPackage não mudar (hash do arquivo).
2. **Composição:** mediana das cenas Sentinel-2 SR harmonizadas dos **150 dias anteriores** à data da rodada, com menos de 30 % de nuvem e máscara de nuvem/cirro (QA60).
3. **Variáveis:** 14 por pixel (tabela abaixo).
4. **Classificação:** duas saídas independentes, `urban_heu` e `urban_pca`.
5. **Persistência:** tabelas `pixel` (geometria e metadados do polígono) e `pixel_urban` (variáveis, rótulos e data), com hash MD5 de pixel + data para que repetir uma rodada não duplique linhas.

## Estrutura do repositório

```
.
├── urbanMask.py                    # pipeline principal
├── model/
│   ├── pipeline_final_cv.joblib    # pipeline scikit-learn (StandardScaler + PCA + Random Forest)
│   ├── best_threshold_cv.pkl       # limiar de decisão otimizado por validação cruzada
│   └── README.md                   # ficha do modelo
├── training/
│   ├── 01_extract_pixels_gee.py    # extração de pixels de treino no Earth Engine
│   └── 02_train_pca_rf.py          # treino do PCA + Random Forest
├── deploy/
│   ├── Dockerfile                  # imagem do pipeline
│   ├── run.sh                      # execução local ou em depuração
│   └── cronjob.example.yaml        # exemplo de CronJob Kubernetes
├── docs/
│   ├── input_format.md             # formato esperado do GeoPackage
│   ├── database_schema.md          # esquema das tabelas
│   └── limitations.md              # limitações conhecidas
├── data/
│   └── apms.gpkg                   # APMs de exemplo, sem identificadores de pessoas
├── tests/                          # testes automatizados (pytest)
├── requirements.txt
├── CITATION.cff
├── NOTICE
└── LICENSE
```

## Variáveis extraídas do Sentinel-2

| # | Variável | Tipo | Descrição |
|---|----------|------|-----------|
| 1 | `ndvi` | Mediana | Índice de vegetação por diferença normalizada |
| 2 | `ndvi_re` | Mediana | NDVI com borda do vermelho (NIR/RE1) |
| 3 | `gndvi` | Mediana | NDVI com banda verde (NIR/Green) |
| 4 | `savi` | Mediana | Índice de vegetação ajustado ao solo |
| 5 | `mndwi` | Mediana | Índice de água por diferença normalizada modificado |
| 6 | `ndbi` | Mediana | Índice de área construída por diferença normalizada |
| 7 | `ui` | Mediana | Índice urbano (SWIR2/NIR) |
| 8 | `nbi` | Mediana | Índice normalizado de construções |
| 9 | `bsi` | Mediana | Índice de solo exposto |
| 10 | `baei` | Mediana | Índice de extração de área construída |
| 11 | `ibi` | Mediana | Índice de construção baseado em índices |
| 12 | `ndvi_std` | Temporal | Desvio-padrão do NDVI na janela de 150 dias |
| 13 | `entropy` | Textura | Entropia GLCM da banda NIR (size = 3) |
| 14 | `ndbi_std` | Temporal | Desvio-padrão do NDBI na janela de 150 dias |

## Classificação dupla

### `urban_heu`: regra espectral

Aplicada à imagem mediana:

```
NBI > 0.18  AND  UI > 0.05  AND  NDVI < 0.20  AND  MNDWI > -0.50
```

Retorna `1` (superfície construída) ou `0`. É mais sensível: detecta mais, com mais falsos positivos.

### `urban_pca`: PCA + Random Forest

1. O pipeline `model/pipeline_final_cv.joblib` é carregado uma única vez, fora do laço.
2. A probabilidade vem de `predict_proba(X)[:, 1]`.
3. O limiar de decisão (0,291) está em `model/best_threshold_cv.pkl` e foi escolhido por validação cruzada.
4. `probabilidade >= limiar` resulta em `1`, caso contrário `0`.

É mais preciso que a regra. Detalhes do treino, dos dados e das limitações em [`model/README.md`](model/README.md).

## Banco de dados

### Tabela `pixel_urban`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `id` | SERIAL (PK) | Identificador |
| `date_time` | TIMESTAMP | Data da análise |
| `longitude`, `latitude` | FLOAT | Coordenadas |
| `ndvi` ... `ndbi_std` | FLOAT x14 | Variáveis espectrais, de textura e temporais |
| `urban_heu` | INTEGER | Classificação da regra (0/1) |
| `urban_pca` | INTEGER | Classificação do modelo (0/1) |
| `periodo` | TEXT | Rótulo do período (ex.: `QUINZENAL`) |
| `id_pixel` | INTEGER (FK) | Referência à tabela `pixel` |
| `pixel_date_hash` | TEXT UNIQUE | MD5 de `pixel_id + target_date` |

A tabela `pixel` guarda os metadados do GeoPackage de entrada. O esquema completo está em [`docs/database_schema.md`](docs/database_schema.md).

### Consultas úteis

**Comparação entre os modelos por data:**
```sql
SELECT
    DATE(date_time)                                    AS data,
    COUNT(*) FILTER (WHERE urban_heu = 1)              AS heu_urbano,
    COUNT(*) FILTER (WHERE urban_pca = 1)              AS ml_urbano,
    COUNT(*)                                           AS total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE urban_heu = 1) / COUNT(*), 2) AS pct_heu,
    ROUND(100.0 * COUNT(*) FILTER (WHERE urban_pca = 1) / COUNT(*), 2) AS pct_ml
FROM pixel_urban
GROUP BY DATE(date_time)
ORDER BY data DESC;
```

**Concordância entre os modelos nos últimos 30 dias:**
```sql
SELECT urban_heu, urban_pca, COUNT(*) AS pixels
FROM pixel_urban
WHERE date_time >= NOW() - INTERVAL '30 days'
GROUP BY urban_heu, urban_pca;
```

## Instalação

Python 3.11. O modelo foi serializado com scikit-learn 1.6.1.

```bash
pip install -r requirements.txt
pytest          # opcional: roda os testes
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `GEE_SERVICE_ACCOUNT` | - | JSON completo da conta de serviço do Earth Engine (**preferencial**) |
| `GCP_CREDENTIALS_PATH` | `/secret/gcp-credentials.json` | Alternativa: caminho do arquivo JSON |
| `DB_HOST` | - | Host do PostgreSQL |
| `DB_PORT` | `5432` | Porta do PostgreSQL |
| `DB_USER` | - | Usuário do banco |
| `DB_PASSWORD` | - | Senha do banco |
| `DB_NAME` | - | Nome do banco |
| `BATCH_SIZE` | `5000` | Pontos por submissão ao Earth Engine |
| `GEE_TILE_SCALE` | `8` | `tileScale` do `sampleRegions` |
| `GEE_CONCURRENT_BATCHES` | `4` | Lotes do Earth Engine em paralelo |
| `DB_INSERT_CHUNK` | `5000` | Linhas por `executemany` |
| `MODEL_DIR` | `./model` | Pasta com os artefatos do modelo |
| `CHECKPOINT_DIR` | `/home/jovyan/checkpoints` | Pasta dos checkpoints `.parquet` |

> Credenciais nunca vão no código nem no repositório. Em Kubernetes, injete-as por Secrets (ver `deploy/cronjob.example.yaml`).

## Execução

### Linha de comando

```bash
# testar a conexão com o banco
python urbanMask.py --test-db

# execução de uma data
python urbanMask.py --gpkg /caminho/areas.gpkg --date 2026-06-04 --periodo QUINZENAL

# processamento retroativo (série histórica, de 15 em 15 dias)
python urbanMask.py --gpkg /caminho/areas.gpkg --start-date 2024-01-01 --end-date 2024-12-31

# sem gravar no banco (teste)
python urbanMask.py --gpkg /caminho/areas.gpkg --date 2026-06-04 --no-db
```

Tempo de referência: cerca de 5 horas por rodada para o conjunto de áreas do projeto original, em uma conta de serviço do Earth Engine. Cada rodada recalcula tudo do zero.

### Como módulo

```python
from urbanMask import run_urban_detection, load_ml_artifacts

# carrega pipeline e limiar uma única vez
pipeline, threshold = load_ml_artifacts()

# processa várias datas sem recarregar o modelo
for date in ["2024-10-01", "2024-10-16", "2024-11-01"]:
    run_urban_detection(
        gpkg_path="/caminho/areas.gpkg",
        target_date=date,
        ml_pipeline=pipeline,
        best_threshold=threshold,
    )
```

### Docker

```bash
docker build -f deploy/Dockerfile -t urbanmask .
docker run -v /caminho/dados:/data \
  -e GEE_SERVICE_ACCOUNT='...' -e DB_HOST=... -e DB_PORT=5432 \
  -e DB_USER=... -e DB_PASSWORD=... -e DB_NAME=... \
  urbanmask --gpkg /data/areas.gpkg
```

## Deploy em Kubernetes

O exemplo `deploy/cronjob.example.yaml` agenda a execução quinzenal (`0 2 */15 * *`, às 02h00 dos dias 1, 16 e 31). Ajuste namespace, imagem, volumes e segredos.

```bash
# forçar uma execução imediata, fora do agendamento
kubectl create job --from=cronjob/urbanmask urbanmask-manual-$(date +%s) -n <NAMESPACE>

# acompanhar os logs
kubectl logs -n <NAMESPACE> -l app=urbanmask --follow
```

## Resiliência e desempenho

- **Checkpoints:** cada lote é salvo em `.parquet` antes do INSERT; em caso de falha, a execução retoma do ponto anterior.
- **Idempotência:** `ON CONFLICT DO NOTHING` com o hash MD5 de `pixel_id + data`.
- **Repetição no Earth Engine:** espera exponencial (10 s, 20 s, 40 s) em falhas de rede.
- **Limite do Earth Engine:** 10 minutos por operação (conta de serviço).
- **Pool do PostgreSQL:** `pool_size=5`, `max_overflow=10`, `pool_pre_ping=True`.

## Treino e modelo

`training/` traz os dois passos usados para gerar o modelo: extração dos pixels no Earth Engine e treino do PCA + Random Forest (StandardScaler, PCA com 95 % da variância e seis componentes, Random Forest de 100 árvores com profundidade 12). Os rótulos vêm do Google Open Buildings Temporal. A divisão do treino é aleatória por pixel.

## O que não está neste repositório

Os identificadores que ligam um polígono a uma pessoa (nome do proprietário, inscrição cadastral e campos derivados) foram removidos do dado de exemplo. O formato esperado da sua própria camada está em [`docs/input_format.md`](docs/input_format.md).

## Limitações

Veja [`docs/limitations.md`](docs/limitations.md). Em resumo: o Sentinel-2 tem 10 m (100 m² por pixel); solo exposto é a maior fonte de erro; a detecção indica superfície construída, **não** irregularidade jurídica; a decisão final é sempre humana.

## Licença, citação e dados de terceiros

Código sob **Apache License 2.0** (`LICENSE`). Os rótulos de treino derivam do Google Open Buildings Temporal (CC BY 4.0); veja `NOTICE`. Para citar, use `CITATION.cff`.

---

**Instituição**: CEIA, Universidade Federal de Goiás (UFG)
