# Limitações conhecidas

- **Resolução:** um pixel de 10 m cobre 100 m², maior que muitas ampliações; construções pequenas podem não ser vistas.
- **Solo exposto** e vegetação seca são a principal fonte de falsos positivos; a heurística é a mais sensível.
- **Borda das APMs:** pixels a poucos metros da divisa misturam o lote vizinho com a APM e têm precisão bem menor.
- **Indicação, não irregularidade:** a detecção mostra superfície construída; equipamentos públicos legítimos também aparecem. A decisão é sempre da fiscalização.
- **Base geográfica:** uma área ausente do GeoPackage nunca é observada; uma destinação desatualizada pode classificar mal uma obra autorizada.
- **Latência:** a resolução temporal é a da rodada (15 dias); a obra só aparece quando passa a dominar a mediana da janela.

