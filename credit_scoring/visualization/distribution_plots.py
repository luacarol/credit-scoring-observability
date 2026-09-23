"""Geração de gráficos de distribuição antes e depois do drift.

Gera um relatório HTML standalone com histogramas e KDE comparando
o dataset de referência (treino) com o dataset de produção (drift).
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

FEATURES_TO_PLOT = [
    ("renda_mensal", "Renda Mensal (R$)"),
    ("score_credito", "Score de Crédito"),
    ("anos_emprego", "Anos de Emprego"),
    ("valor_emprestimo", "Valor do Empréstimo (R$)"),
    ("inadimplente", "Taxa de Inadimplência"),
]

COLORS = {"reference": "#2196F3", "production": "#F44336"}


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    mn = min(expected.min(), actual.min())
    mx = max(expected.max(), actual.max())
    bins = np.linspace(mn, mx, buckets + 1)

    expected_perc, _ = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bins)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc) / len(expected)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc) / len(actual)

    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def _plot_feature(
    ax: plt.Axes,
    ref: np.ndarray,
    prod: np.ndarray,
    feature_label: str,
    is_binary: bool = False,
) -> dict:
    """Plota histograma + KDE para uma feature. Retorna métricas calculadas."""
    if is_binary:
        categories = [0, 1]
        ref_counts = [np.mean(ref == c) for c in categories]
        prod_counts = [np.mean(prod == c) for c in categories]
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width / 2, ref_counts, width, label="Referência", color=COLORS["reference"], alpha=0.8)
        ax.bar(x + width / 2, prod_counts, width, label="Produção", color=COLORS["production"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Adimplente (0)", "Inadimplente (1)"])
        ax.set_ylabel("Proporção")
        _, p_value = stats.chisquare(
            [np.sum(prod == c) for c in categories],
            f_exp=[np.sum(ref == c) * len(prod) / len(ref) for c in categories],
        )
    else:
        bins = 30
        ax.hist(ref, bins=bins, alpha=0.5, color=COLORS["reference"], label="Referência", density=True)
        ax.hist(prod, bins=bins, alpha=0.5, color=COLORS["production"], label="Produção", density=True)

        # KDE curves
        for data, color in [(ref, COLORS["reference"]), (prod, COLORS["production"])]:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)

        _, p_value = stats.ks_2samp(ref, prod)

    psi = _compute_psi(ref, prod)
    drift_detected = p_value < 0.05

    title_color = "#D32F2F" if drift_detected else "#388E3C"
    status = "⚠ DRIFT" if drift_detected else "✓ Estável"
    ax.set_title(
        f"{feature_label}\n{status} | PSI={psi:.3f} | p-value={p_value:.4f}",
        fontsize=10,
        color=title_color,
        fontweight="bold",
    )
    ax.set_xlabel(feature_label, fontsize=9)
    ax.set_ylabel("Densidade", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    return {"psi": psi, "p_value": p_value, "drift_detected": drift_detected}


def generate_distribution_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Gera relatório HTML com gráficos de distribuição antes/depois do drift."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path or (REPORTS_DIR / "distribution_report.html")

    n_features = len(FEATURES_TO_PLOT)
    fig, axes = plt.subplots(
        nrows=(n_features + 1) // 2,
        ncols=2,
        figsize=(16, 4 * ((n_features + 1) // 2)),
    )
    axes = axes.flatten()

    metrics_summary = {}
    for i, (feature, label) in enumerate(FEATURES_TO_PLOT):
        ref_vals = reference_df[feature].dropna().values
        prod_vals = production_df[feature].dropna().values
        is_binary = feature == "inadimplente"
        metrics_summary[feature] = _plot_feature(axes[i], ref_vals, prod_vals, label, is_binary)

    # Remove eixo extra se número de features for ímpar
    if n_features % 2 != 0:
        axes[-1].set_visible(False)

    n_drifted = sum(1 for m in metrics_summary.values() if m["drift_detected"])
    fig.suptitle(
        f"Distribuição: Referência vs Produção\n"
        f"{n_drifted}/{n_features} features com drift detectado",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    # Salva figura e embute no HTML
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    html = _build_html(img_b64, metrics_summary, reference_df, production_df)
    output_path.write_text(html, encoding="utf-8")

    logger.info("Relatório de distribuição salvo em %s", output_path)
    return output_path


def _build_html(img_b64: str, metrics: dict, ref_df: pd.DataFrame, prod_df: pd.DataFrame) -> str:
    """Monta o HTML completo com gráfico e tabela de métricas."""
    rows = ""
    for feature, label in FEATURES_TO_PLOT:
        m = metrics[feature]
        status_html = (
            '<span style="color:#D32F2F;font-weight:bold">⚠ Detectado</span>'
            if m["drift_detected"]
            else '<span style="color:#388E3C;font-weight:bold">✓ Não detectado</span>'
        )
        ref_mean = ref_df[feature].mean()
        prod_mean = prod_df[feature].mean()
        delta = ((prod_mean - ref_mean) / ref_mean) * 100
        delta_html = (
            f'<span style="color:#D32F2F">▲ +{delta:.1f}%</span>'
            if delta > 0
            else f'<span style="color:#1565C0">▼ {delta:.1f}%</span>'
        )
        rows += f"""
        <tr>
            <td>{label}</td>
            <td>{status_html}</td>
            <td>{m['psi']:.4f}</td>
            <td>{m['p_value']:.6f}</td>
            <td>{ref_mean:.2f}</td>
            <td>{prod_mean:.2f}</td>
            <td>{delta_html}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Distribution Drift Report — Credit Scoring</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
        h2 {{ color: #283593; margin-top: 40px; }}
        .summary-box {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 14px; }}
        .badge-alert {{ background: #FFEBEE; color: #C62828; }}
        .badge-ok {{ background: #E8F5E9; color: #2E7D32; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th {{ background: #1a237e; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f5f5f5; }}
        img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .context {{ background: #E3F2FD; border-left: 4px solid #1565C0; padding: 12px 16px; border-radius: 4px; margin: 16px 0; }}
    </style>
</head>
<body>
    <h1>Distribution Drift Report — Credit Scoring</h1>

    <div class="summary-box">
        <p class="context">
            <strong>Contexto:</strong> Simulação de crise econômica — inflação (+85% na renda),
            queda do score de crédito (-30%) e instabilidade no emprego.
            O modelo foi treinado no dataset de <strong>Referência</strong> e encontra em produção
            um perfil de clientes completamente diferente.
        </p>
        <p>
            <strong>Datasets:</strong> {len(ref_df):,} amostras de referência vs {len(prod_df):,} amostras de produção
            &nbsp;|&nbsp;
            <strong>Threshold:</strong> PSI &gt; 0.2 ou p-value &lt; 0.05
        </p>
    </div>

    <h2>Gráficos de Distribuição</h2>
    <img src="data:image/png;base64,{img_b64}" alt="Distribution plots" />

    <h2>Tabela de Métricas por Feature</h2>
    <table>
        <thead>
            <tr>
                <th>Feature</th>
                <th>Status</th>
                <th>PSI</th>
                <th>p-value (KS)</th>
                <th>Média Referência</th>
                <th>Média Produção</th>
                <th>Variação</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>

    <div class="summary-box" style="margin-top:30px">
        <h3 style="margin-top:0">Interpretação</h3>
        <ul>
            <li><strong>PSI &gt; 0.2</strong>: drift significativo — modelo precisa ser retreinado</li>
            <li><strong>p-value &lt; 0.05</strong>: distribuições estatisticamente diferentes (teste KS)</li>
            <li><strong>Variação</strong>: diferença percentual entre a média de referência e produção</li>
        </ul>
    </div>
</body>
</html>"""
