import re
from pathlib import Path
import joblib, pytest

MODEL = Path(__file__).parent.parent / "model"
SRC = (Path(__file__).parent.parent / "urbanMask.py").read_text()

@pytest.fixture(scope="module")
def pipe(): return joblib.load(MODEL / "pipeline_final_cv.joblib")

def test_limiar():
    assert joblib.load(MODEL / "best_threshold_cv.pkl") == pytest.approx(0.291, abs=1e-3)

def test_estrutura(pipe):
    assert list(pipe.named_steps) == ["scaler", "pca", "rf"]
    assert pipe.named_steps["pca"].n_components_ == 6
    rf = pipe.named_steps["rf"]
    assert len(rf.estimators_) == 100 and rf.max_depth == 12

def test_ordem_das_variaveis(pipe):
    ml = re.search(r"ML_FEATURES: List\[str\] = \[(.*?)\]", SRC, re.S).group(1)
    nomes = re.findall(r"'(\w+)'", ml)
    assert list(pipe.named_steps["scaler"].feature_names_in_) == nomes and len(nomes) == 14

def test_predicao(pipe):
    import numpy as np, pandas as pd
    X = pd.DataFrame(np.zeros((3, 14)), columns=pipe.named_steps["scaler"].feature_names_in_)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (3,) and ((p >= 0) & (p <= 1)).all()
