# Formato do GeoPackage de entrada

- Camada de polígonos em qualquer CRS (o pipeline reprojeta para EPSG:4326).
- Cada linha vira uma grade de pixels de ≈ 10 m. **Atributos** da linha são herdados por todos os pixels dela e gravados na tabela `pixel`.
- Colunas esperadas: uma geometria (polígono) e um identificador por feição; os demais atributos são opcionais. Para usar outra camada, adapte `create_tables` e `save_pixel_data` em `urbanMask.py`.
