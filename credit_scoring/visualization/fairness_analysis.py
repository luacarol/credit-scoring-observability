"""Análise de fairness/equidade por faixa etária.

Verifica se o modelo tem desempenho similar entre grupos demográficos,
detectando possíveis vieses na concessão de crédito.

Métricas avaliadas por faixa etária:
  - Taxa de aprovação (1 - inadimplente previsto)
  - Taxa de falsos positivos (bom pagador classificado como inadimplente)
  - Taxa de falsos negativos (inadimplente classificado como bom pagador)
  - Accuracy por grupo
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

AGE_BINS = [18, 30, 45, 60, 100]
AGE_LABELS = ["18-30", "31-45", "46-60", "61+"]
COLORS = ["#1565C0", "#2E7D32", "#F57F17", "#C62828"]


def _assign_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    )
    return df


def _compute_group_metrics(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Calcula métricas de fairness por faixa etária."""
    df = df.copy()
    df["predicao"] = predictions
    df["y_true"] = df["inadimplente"]

    rows = []
    for group in AGE_LABELS:
        mask = df["faixa_etaria"] == group
        subset = df[mask]

        if len(subset) == 0:
            continue

        y_true = subset["y_true"].values
        y_pred = subset["predicao"].values

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = (tp + tn) / len(subset) if len(subset) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        approval_rate = np.mean(y_pred == 0)
        default_rate = np.mean(y_true == 1)

        rows.append(
            {
                "faixa_etaria": group,
                "n_amostras": len(subset),
                "taxa_aprovacao": approval_rate,
                "taxa_inadimplencia_real": default_rate,
                "accuracy": accuracy,
                "fpr": fpr,
                "fnr": fnr,
            }
        )

    return pd.DataFrame(rows)


def _plot_fairness(metrics_ref: pd.DataFrame, metrics_prod: pd.DataFrame) -> str:
    """Gera figura comparando fairness na referência vs produção. Retorna base64."""
    import base64
    import io

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    groups = AGE_LABELS
    x = np.arange(len(groups))
    width = 0.35

    plots = [
        ("taxa_aprovacao", "Taxa de Aprovação", "Taxa (%)"),
        ("accuracy", "Accuracy por Grupo", "Accuracy"),
        ("fpr", "False Positive Rate", "Taxa"),
        ("fnr", "False Negative Rate", "Taxa"),
    ]

    for idx, (col, title, ylabel) in enumerate(plots):
        ax = axes[idx]

        ref_vals = [
            metrics_ref.loc[metrics_ref["faixa_etaria"] == g, col].values[0]
            if g in metrics_ref["faixa_etaria"].values
            else 0
            for g in groups
        ]
        prod_vals = [
            metrics_prod.loc[metrics_prod["faixa_etaria"] == g, col].values[0]
            if g in metrics_prod["faixa_etaria"].values
            else 0
            for g in groups
        ]

        ax.bar(x - width / 2, ref_vals, width, label="Referência", color="#2196F3", alpha=0.8)
        ax.bar(x + width / 2, prod_vals, width, label="Produção", color="#F44336", alpha=0.8)

        # Linha de paridade (ideal)
        overall_ref = np.mean(ref_vals)
        ax.axhline(overall_ref, color="#2196F3", linestyle="--", alpha=0.5, linewidth=1)
        overall_prod = np.mean(prod_vals)
        ax.axhline(overall_prod, color="#F44336", linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Destaca grupos com desvio > 10% da média
        for i, (_rv, pv) in enumerate(zip(ref_vals, prod_vals, strict=False)):
            if abs(pv - overall_prod) > 0.10:
                ax.get_xticklabels()[i].set_color("#D32F2F")
                ax.get_xticklabels()[i].set_fontweight("bold")

    fig.suptitle(
        "Análise de Fairness por Faixa Etária\nReferência vs Produção",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def _build_html(
    img_b64: str,
    metrics_ref: pd.DataFrame,
    metrics_prod: pd.DataFrame,
) -> str:
    def _rows(metrics: pd.DataFrame) -> str:
        html = ""
        overall_acc = metrics["accuracy"].mean()
        for _, row in metrics.iterrows():
            deviation = abs(row["accuracy"] - overall_acc)
            flag = " ⚠" if deviation > 0.10 else ""
            flag_color = "#D32F2F" if flag else "inherit"
            html += f"""
            <tr>
                <td>{row["faixa_etaria"]}</td>
                <td>{int(row["n_amostras"])}</td>
                <td>{row["taxa_aprovacao"]:.1%}</td>
                <td>{row["taxa_inadimplencia_real"]:.1%}</td>
                <td style="color:{flag_color};font-weight:{"bold" if flag else "normal"}">{row["accuracy"]:.1%}{flag}</td>
                <td>{row["fpr"]:.1%}</td>
                <td>{row["fnr"]:.1%}</td>
            </tr>"""
        return html

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Fairness Analysis — Credit Scoring</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
        h2 {{ color: #283593; margin-top: 40px; }}
        .context {{ background: #E3F2FD; border-left: 4px solid #1565C0; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
        .warning {{ background: #FFEBEE; border-left: 4px solid #C62828; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        th {{ background: #1a237e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .lgpd-box {{ background: #F3E5F5; border-left: 4px solid #7B1FA2; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
    </style>
</head>
<body>
    <h1>Fairness Analysis — Credit Scoring por Faixa Etária</h1>

    <div class="context">
        <strong>Objetivo:</strong> Verificar se o modelo toma decisões equitativas entre grupos etários.
        Desvios de accuracy acima de 10% da média geral são sinalizados como potencial viés (<span style="color:#D32F2F">⚠</span>).
    </div>

    <div class="lgpd-box">
        <strong>LGPD — Art. 20:</strong> O titular tem direito à revisão de decisões automatizadas.
        Esta análise garante que o modelo não discrimina sistematicamente por faixa etária,
        em conformidade com a Lei Geral de Proteção de Dados.
    </div>

    <h2>Gráficos Comparativos</h2>
    <img src="data:image/png;base64,{img_b64}" alt="Fairness plots" />

    <h2>Métricas — Dataset de Referência (Treino)</h2>
    <table>
        <thead>
            <tr>
                <th>Faixa Etária</th>
                <th>Amostras</th>
                <th>Taxa Aprovação</th>
                <th>Inadimplência Real</th>
                <th>Accuracy</th>
                <th>FPR</th>
                <th>FNR</th>
            </tr>
        </thead>
        <tbody>{_rows(metrics_ref)}</tbody>
    </table>

    <h2>Métricas — Dataset de Produção (Drift)</h2>
    <table>
        <thead>
            <tr>
                <th>Faixa Etária</th>
                <th>Amostras</th>
                <th>Taxa Aprovação</th>
                <th>Inadimplência Real</th>
                <th>Accuracy</th>
                <th>FPR</th>
                <th>FNR</th>
            </tr>
        </thead>
        <tbody>{_rows(metrics_prod)}</tbody>
    </table>

    <div class="warning">
        <strong>Interpretação:</strong>
        <ul style="margin:8px 0">
            <li><strong>FPR (False Positive Rate)</strong>: bom pagador classificado como inadimplente — impacto negativo para o cliente</li>
            <li><strong>FNR (False Negative Rate)</strong>: inadimplente aprovado — impacto financeiro para a instituição</li>
            <li>Grupos com accuracy ⚠ indicam possível viés que deve ser investigado antes do deploy</li>
        </ul>
    </div>
</body>
</html>"""


def generate_fairness_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    output_path: Path | None = None,
) -> Path:
    """Gera relatório HTML de fairness por faixa etária."""
    from credit_scoring.models.train import _encode_categoricals

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (REPORTS_DIR / "fairness_report.html")

    ref_enc = _assign_age_group(_encode_categoricals(reference_df))
    prod_enc = _assign_age_group(_encode_categoricals(production_df))

    cols = [c for c in feature_cols if c in ref_enc.columns]
    preds_ref = model.predict(ref_enc[cols])
    preds_prod = model.predict(prod_enc[cols])

    metrics_ref = _compute_group_metrics(ref_enc, preds_ref)
    metrics_prod = _compute_group_metrics(prod_enc, preds_prod)

    img_b64 = _plot_fairness(metrics_ref, metrics_prod)
    html = _build_html(img_b64, metrics_ref, metrics_prod)
    output_path.write_text(html, encoding="utf-8")

    logger.info("Relatório de fairness salvo em %s", output_path)
    return output_path
