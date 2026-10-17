#!/bin/bash

set -e

echo "======================================================================"
echo "URBAN MASK DETECTION - Automated Execution"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

cd /home/jovyan

echo "Instalando dependências..."
pip install -q earthengine-api geemap geopandas sqlalchemy psycopg2-binary pandas numpy shapely pyarrow "scikit-learn==1.6.1" joblib
echo "Dependências instaladas"

echo "Iniciando processamento..."
python urbanMask.py \
    --gpkg /home/jovyan/areas.gpkg \
    --periodo QUINZENAL

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo "Execution completed successfully"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "Execution failed with exit code: $EXIT_CODE"
    echo "======================================================================"
    exit $EXIT_CODE
fi