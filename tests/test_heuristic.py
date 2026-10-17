"""Testa a regra HEU sem importar as dependências pesadas (Earth Engine etc.): extrai a função do código-fonte."""
import ast
from pathlib import Path
from typing import Dict, Optional
import numpy as np

SRC = (Path(__file__).parent.parent / "urbanMask.py").read_text()
fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "urban_classification_logic")
ns = {"np": np, "Dict": Dict, "Optional": Optional}
exec(compile(ast.Module([fn], []), "urbanMask.py", "exec"), ns)
heu = ns["urban_classification_logic"]

def ind(**kw):
    base = dict(nbi=0.25, ui=0.10, ndvi=0.05, mndwi=-0.30); base.update(kw); return base

def test_construcao():            assert heu(ind()) == 1
def test_vegetacao_alta():        assert heu(ind(ndvi=0.30)) == 0
def test_nbi_baixo():             assert heu(ind(nbi=0.10)) == 0
def test_ui_baixo():              assert heu(ind(ui=0.00)) == 0
def test_mndwi_muito_baixo():     assert heu(ind(mndwi=-0.60)) == 0
def test_valores_invalidos():     assert heu(ind(ndvi=float("nan"))) is None
