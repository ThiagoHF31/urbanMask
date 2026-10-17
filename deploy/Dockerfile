# Imagem do UrbanMask. Os dados de entrada NÃO vão na imagem: monte-os como volume.
# docker run -v /caminho/dados:/data -e GEE_SERVICE_ACCOUNT=... -e DB_HOST=... urbanmask --gpkg /data/apms.gpkg
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgdal-dev && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgdal36 libgeos-c1t64 libproj25 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home jovyan
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /home/jovyan
COPY --chown=jovyan:jovyan urbanMask.py .
COPY --chown=jovyan:jovyan model/ ./model/
USER jovyan
ENTRYPOINT ["python", "urbanMask.py"]
CMD ["--help"]
