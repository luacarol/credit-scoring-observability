"""Detecção de Data Drift e Concept Drift com Evidently AI 0.7.

Gera relatórios HTML comparando dataset de referência vs produção.

Métricas:
  - Data drift (numéricas): PSI — Population Stability Index (drift se PSI > 0.2)
  - Data drift (categóricas): teste chi-square (drift se p-value < 0.05)
  - Concept/label drift: drift na variável alvo ``inadimplente`` (chi-square)

O concept drift de performance (degradação de accuracy) é calculado no
pipeline, complementando o drift de rótulo detectado aqui.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

NUMERIC_FEATURES = [
    "idade",
    "anos_emprego",
    "renda_mensal",
    "valor_emprestimo",
    "score_credito",
]
CATEGORICAL_FEATURES = [
    "historico_pagamentos",
    "finalidade",
    "possui_conta_poupanca",
]
TARGET_COL = "inadimplente"

# Limiar convencional de mercado para PSI (Evidently usa 0.1 por padrão).
PSI_THRESHOLD = 0.2
P_VALUE_THRESHOLD = 0.05
# Flag de dataset drift quando a fração de features com drift atinge este valor.
DATASET_DRIFT_SHARE = 0.3

# Métodos baseados em "distância": quanto maior, mais drift.
DISTANCE_METHODS = {"psi", "hellinger", "jensenshannon", "wasserstein", "kl_div", "tvd"}

MONITORED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]


def _build_dataset(df: pd.DataFrame) -> Dataset:
    data_def = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES + [TARGET_COL],
    )
    return Dataset.from_pandas(df[MONITORED_COLUMNS].dropna(), data_definition=data_def)


def run_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_path: Path | None = None,
) -> tuple[Path, dict]:
    """Executa relatório de drift e retorna (caminho_html, métricas_json)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (REPORTS_DIR / "drift_report.html")

    ref_ds = _build_dataset(reference_df)
    prod_ds = _build_dataset(production_df)

    report = Report(
        [
            DataDriftPreset(
                num_method="psi",
                num_threshold=PSI_THRESHOLD,
                cat_method="chisquare",
                cat_threshold=P_VALUE_THRESHOLD,
            ),
            DataSummaryPreset(),
        ]
    )
    result = report.run(reference_data=ref_ds, current_data=prod_ds)
    result.save_html(str(output_path))
    logger.info("Relatório de drift salvo em %s", output_path)

    metrics = _extract_drift_metrics(result)
    metrics["advanced_distances"] = _compute_advanced_distances(reference_df, production_df)

    metrics_path = REPORTS_DIR / "drift_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        "Métricas de drift: %d features com drift detectado",
        metrics.get("n_drifted_features", 0),
    )

    return output_path, metrics


def _parse_value_drift(name: str) -> tuple[str, str]:
    """Extrai (coluna, método) do nome da métrica ValueDrift."""
    col = name.split("column=")[1].split(",")[0]
    method = name.split("method=")[1].split(",")[0]
    return col, method


def _compute_advanced_distances(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    bins: int = 20,
) -> dict:
    """Distâncias de distribuição avançadas por feature numérica.

    Complementam PSI/KS (métricas "simples") com:
      - Jensen-Shannon distance (base sqrt da JS divergence, ∈ [0, 1])
      - Hellinger distance (∈ [0, 1])

    Ambas comparam histogramas normalizados; quanto maior, mais drift.
    """
    advanced: dict = {}
    for col in NUMERIC_FEATURES:
        ref_vals = reference_df[col].dropna().values
        prod_vals = production_df[col].dropna().values

        if len(ref_vals) == 0 or len(prod_vals) == 0:
            advanced[col] = {"jensen_shannon": None, "hellinger": None}
            continue

        mn = min(ref_vals.min(), prod_vals.min())
        mx = max(ref_vals.max(), prod_vals.max())
        edges = np.linspace(mn, mx, bins + 1)

        ref_hist, _ = np.histogram(ref_vals, bins=edges, density=True)
        prod_hist, _ = np.histogram(prod_vals, bins=edges, density=True)

        # density=True soma à largura do bin, não a 1; normaliza para distribuição
        ref_p = ref_hist / ref_hist.sum()
        prod_p = prod_hist / prod_hist.sum()

        # Evita log(0) — small epsilon
        ref_p = ref_p + 1e-12
        prod_p = prod_p + 1e-12

        js_divergence = jensenshannon(ref_p, prod_p) ** 2
        hellinger = float(np.sqrt(np.sum((np.sqrt(ref_p) - np.sqrt(prod_p)) ** 2) / 2))

        advanced[col] = {
            "jensen_shannon": round(float(js_divergence), 6),
            "hellinger": round(hellinger, 6),
        }

    return advanced


def _extract_drift_metrics(result) -> dict:
    """Extrai métricas de drift do resultado do Evidently 0.7."""
    try:
        d = result.dict()
        metrics_raw = d.get("metrics", [])

        drift_metrics: dict = {
            "drifted_features": [],
            "n_drifted_features": 0,
            "share_drifted_features": 0.0,
            "dataset_drift": False,
            "feature_scores": {},
            "psi_by_feature": {},
            "target_drift": None,
        }

        for m in metrics_raw:
            name = m.get("metric_name", "")
            value = m.get("value")

            if "DriftedColumnsCount" in name:
                if isinstance(value, dict):
                    drift_metrics["n_drifted_features"] = int(value.get("count", 0))
                    drift_metrics["share_drifted_features"] = float(value.get("share", 0.0))
                    drift_metrics["dataset_drift"] = (
                        drift_metrics["share_drifted_features"] >= DATASET_DRIFT_SHARE
                    )
                continue

            if "ValueDrift" not in name:
                continue

            try:
                col, method = _parse_value_drift(name)
            except (IndexError, ValueError):
                continue

            if value is None:
                continue

            statistic = float(value)
            # Distância (PSI etc.): maior = mais drift. p-value: menor = mais drift.
            if method in DISTANCE_METHODS:
                drift_detected = statistic > PSI_THRESHOLD
            else:
                drift_detected = statistic < P_VALUE_THRESHOLD

            feature_info = {
                "method": method,
                "statistic": round(statistic, 6),
                "drift_detected": drift_detected,
            }
            drift_metrics["feature_scores"][col] = feature_info

            if method == "psi":
                drift_metrics["psi_by_feature"][col] = round(statistic, 6)

            if drift_detected:
                drift_metrics["drifted_features"].append(col)

            if col == TARGET_COL:
                drift_metrics["target_drift"] = feature_info

        return drift_metrics

    except Exception as exc:
        logger.warning("Não foi possível extrair métricas detalhadas: %s", exc)
        return {"error": str(exc), "n_drifted_features": 0, "dataset_drift": False}
