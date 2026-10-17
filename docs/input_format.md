# Formato do GeoPackage de entrada

- Camada de polígonos em qualquer CRS (o pipeline reprojeta para EPSG:4326).
- Cada linha vira uma grade de pixels de ≈ 10 m. **Atributos** da linha são herdados por todos os pixels dela e gravados na tabela `pixel`.
- Colunas esperadas (esquema das APMs): `objectid, nr_apm, cd_geo, cd_log, nm_log, nr_log, nm_qdr, nm_lot, nm_apm, compl, cd_bai, nm_bai, cd_destin, nm_destin, cd_grupo_d, nm_grupo_d, cd_uso, nm_uso, cd_grupo_u, nm_grupo_u, area_orig, area_atual, area_edif, cd_tipo`. Colunas que identificam pessoas (nome do proprietário, inscrição cadastral, CPF/CNPJ) não são necessárias e devem ser removidas antes de publicar. Para usar outra camada, adapte `create_tables` e `save_pixel_data` em `urbanMask.py`.
