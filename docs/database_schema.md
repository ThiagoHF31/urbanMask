# Esquema do banco (PostgreSQL)

**`pixel`** — um registro por pixel de 10 m e polígono: `id` (serial), `geom` (ponto WKT), mais os atributos do polígono (ver `docs/input_format.md`), todos `NOT NULL` com valor padrão seguro.

**`pixel_urban`** — um registro por pixel e data de análise:

| Coluna | Tipo | Descrição |
|---|---|---|
| `date_time` | timestamp | data da rodada |
| `longitude`, `latitude`, `centroide` | float, texto | posição do pixel |
| `ndvi, ndvi_re, gndvi, savi, mndwi, ndbi, ui, nbi, bsi, baei, ibi, ndvi_std, entropy, ndbi_std` | float | as 14 variáveis |
| `urban_heu`, `urban_pca` | 0/1 | classificação da regra e do PCA+RF |
| `periodo` | texto | rótulo do período (ex.: `QUINZENAL`) |
| `id_pixel` | inteiro | chave estrangeira para `pixel.id` |
| `pixel_date_hash` | texto único | MD5 de pixel + data; `ON CONFLICT DO NOTHING` torna a gravação idempotente |

Consulta típica (concordância entre os modelos nos últimos 30 dias):
```sql
SELECT urban_heu, urban_pca, COUNT(*) FROM pixel_urban
WHERE date_time >= NOW() - INTERVAL '30 days' GROUP BY 1, 2;
```
