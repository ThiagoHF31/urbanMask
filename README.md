# UrbanMask

Ferramenta em lote que **reprocessa imagens Sentinel-2 a cada 15 dias** sobre áreas públicas municipais (APMs) e sinaliza pixels de 10 m onde surgiu **nova construção**, mantendo a chave cadastral de cada pixel da ingestão à publicação. Usa dois classificadores sobre as mesmas 14 variáveis espectrais, de textura e temporais:

- **HEU**: regra de limiares com quatro índices;
- **PCA+RF**: PCA seguido de Random Forest.

Os resultados são gravados em PostgreSQL (uma linha por pixel e data).

> **English summary.** UrbanMask is a batch pipeline that reprocesses Sentinel-2 imagery every 15 days over municipal public areas, flags new constructions at 10 m pixel level with two classifiers (a four-index threshold rule and PCA + Random Forest), keeps the cadastral key of each pixel end to end, and stores one PostgreSQL row per pixel and date. Code: Apache-2.0. See the sections below (in Portuguese) and `docs/`.

## Como funciona

```
GeoPackage de polígonos ─► grade de pixels (10 m, em cache) ─► Google Earth Engine (lotes paralelos)
        ─► 14 variáveis por pixel ─► urban_heu (regra)  +  urban_pca (PCA+RF)  ─► PostgreSQL (pixel, pixel_urban)
```

1. **Ingestão e grade:** cada polígono vira uma grade de pontos de ≈ 10 m (polígonos muito pequenos usam um ponto representativo). A grade é guardada em parquet e reutilizada.
2. **Composição:** mediana das cenas Sentinel-2 SR harmonizadas dos **150 dias anteriores** à data da rodada, com menos de 30 % de nuvem e máscara de nuvem/cirro (QA60).
3. **Variáveis (14):** NDVI, NDVI de borda do vermelho, GNDVI, SAVI, MNDWI, NDBI, UI, NBI, BSI, BAEI, IBI; entropia GLCM do infravermelho próximo; desvio-padrão temporal do NDVI e do NDBI.
4. **Classificação:** `urban_heu = 1` se NBI > 0,18, UI > 0,05, NDVI < 0,20 e MNDWI > −0,50; `urban_pca = 1` se a probabilidade do Random Forest ≥ 0,291.
5. **Persistência:** tabelas `pixel` (geometria e metadados cadastrais) e `pixel_urban` (variáveis, rótulos e data), com hash MD5 de pixel+data para que repetir uma rodada não duplique linhas.

## Uso rápido

```bash
pip install -r requirements.txt

# 1) credenciais (nunca no código nem no repositório)
export GEE_SERVICE_ACCOUNT='{...json da conta de serviço do Earth Engine...}'
export DB_HOST=... DB_PORT=5432 DB_USER=... DB_PASSWORD=... DB_NAME=...

# 2) testar a conexão com o banco
python urbanMask.py --test-db

# 3) rodar uma data (ou um intervalo retroativo, de 15 em 15 dias)
python urbanMask.py --gpkg /caminho/apms.gpkg --date 2026-06-04
python urbanMask.py --gpkg /caminho/apms.gpkg --start-date 2024-01-01 --end-date 2026-08-02
python urbanMask.py --gpkg /caminho/apms.gpkg --date 2026-06-04 --no-db   # sem gravar no banco
```

Tempo de referência: ≈ 5 horas por rodada para as APMs do município em uma conta de serviço do Earth Engine; cada rodada recalcula tudo do zero. Em produção roda como CronJob (`deploy/cronjob.example.yaml`).

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `GEE_SERVICE_ACCOUNT` | — | JSON completo da conta de serviço do Earth Engine (preferencial) |
| `GCP_CREDENTIALS_PATH` | `/secret/gcp-credentials.json` | alternativa: caminho do arquivo JSON |
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | — / `5432` | PostgreSQL |
| `BATCH_SIZE` | `5000` | pontos por submissão ao Earth Engine |
| `GEE_TILE_SCALE` | `8` | `tileScale` do `sampleRegions` |
| `GEE_CONCURRENT_BATCHES` | `4` | lotes em paralelo |
| `DB_INSERT_CHUNK` | `5000` | linhas por inserção |
| `MODEL_DIR` | `./model` | pasta com `pipeline_final_cv.joblib` e `best_threshold_cv.pkl` |
| `CHECKPOINT_DIR` | `/home/jovyan/checkpoints` | checkpoints `.parquet` por lote |

## O que **não** está neste repositório

Os dados de entrada. O GeoPackage de entrada do projeto original contém dados pessoais e por isso **não é publicado**; use a sua própria camada. O formato esperado está em [`docs/input_format.md`](docs/input_format.md). O esquema das tabelas está em [`docs/database_schema.md`](docs/database_schema.md).

## Modelo

`model/pipeline_final_cv.joblib` (StandardScaler → PCA com 95 % da variância, seis componentes → Random Forest de 100 árvores, profundidade 12) e `model/best_threshold_cv.pkl` (limiar 0,291). Detalhes, dados de treino e limitações em [`model/README.md`](model/README.md).

## Treino e avaliação

`training/` traz os dois passos usados para gerar o modelo (extração dos pixels no Earth Engine e treino). A divisão do treino é aleatória por pixel. Testes: `pytest`.

## Limitações

Veja [`docs/limitations.md`](docs/limitations.md). Em resumo: o Sentinel-2 tem 10 m (100 m² por pixel); solo exposto é a maior fonte de erro; a detecção indica superfície construída, **não** irregularidade jurídica; a decisão final é sempre humana.

## Licença, citação e dados de terceiros

Código sob **Apache License 2.0** (`LICENSE`). Os rótulos de treino derivam do Google Open Buildings Temporal (CC BY 4.0); veja `NOTICE`. Para citar, use `CITATION.cff`.

Projeto relacionado: pipeline para lotes vagos (`lotesVagos_detection`, só PCA, a cada 45 dias).
