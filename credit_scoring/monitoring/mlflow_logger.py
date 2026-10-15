"""Centraliza o logging de observabilidade no MLflow.

Loga métricas de drift, qualidade e saúde do modelo como um run separado.
"""

import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def log_drift_run(
    drift_metrics: dict,
    model_metrics_ref: dict,
    model_metrics_prod: dict | None = None,
    experiment_name: str = "credit_scoring_monitoring",
) -> str:
    """Cria um run MLflow com todas as métricas de observabilidade e artefatos."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="drift_monitoring") as run:
        # Métricas de drift do dataset
        mlflow.log_metric("n_drifted_features", drift_metrics.get("n_drifted_features", 0))
        mlflow.log_metric(
            "share_drifted_features", drift_metrics.get("share_drifted_features", 0.0)
        )
        mlflow.log_metric("dataset_drift_detected", int(drift_metrics.get("dataset_drift", False)))

        # Estatística de drift por feature (método + valor)
        for feat, info in drift_metrics.get("feature_scores", {}).items():
            mlflow.log_metric(f"drift_{feat}", info.get("statistic", 0.0))

        # PSI por feature numérica (Population Stability Index)
        for feat, psi in drift_metrics.get("psi_by_feature", {}).items():
            mlflow.log_metric(f"psi_{feat}", psi)

        # Distâncias avançadas por feature (Jensen-Shannon, Hellinger)
        for feat, distances in drift_metrics.get("advanced_distances", {}).items():
            for name, value in distances.items():
                if value is not None:
                    mlflow.log_metric(f"{name}_{feat}", value)

        # Drift da variável alvo (concept/label drift)
        target_drift = drift_metrics.get("target_drift")
        if target_drift:
            mlflow.log_metric("target_drift_statistic", target_drift.get("statistic", 0.0))
            mlflow.log_metric(
                "target_drift_detected", int(target_drift.get("drift_detected", False))
            )

        # Métricas do modelo em referência vs produção
        for key, value in model_metrics_ref.items():
            mlflow.log_metric(f"ref_{key}", value)

        if model_metrics_prod:
            for key, value in model_metrics_prod.items():
                mlflow.log_metric(f"prod_{key}", value)

            # Degradação de performance
            if "accuracy" in model_metrics_ref and "accuracy" in model_metrics_prod:
                delta = model_metrics_prod["accuracy"] - model_metrics_ref["accuracy"]
                mlflow.log_metric("accuracy_delta", delta)
                if delta < -0.05:
                    logger.warning("ALERTA: degradação de accuracy de %.3f", abs(delta))

        # Tags de alerta
        alert = drift_metrics.get("dataset_drift", False) or (
            (model_metrics_prod or {}).get("accuracy", 1.0)
            < model_metrics_ref.get("accuracy", 1.0) - 0.05
        )
        mlflow.set_tag("drift_alert", str(alert))
        mlflow.set_tag("drifted_features", ", ".join(drift_metrics.get("drifted_features", [])))
        mlflow.set_tag("n_drifted", str(drift_metrics.get("n_drifted_features", 0)))

        # Artefatos HTML e logs de execução
        for html_file in REPORTS_DIR.glob("*.html"):
            mlflow.log_artifact(str(html_file), artifact_path="reports")
        for json_file in REPORTS_DIR.glob("*.json"):
            mlflow.log_artifact(str(json_file), artifact_path="reports")
        for log_file in REPORTS_DIR.glob("*.log"):
            mlflow.log_artifact(str(log_file), artifact_path="logs")

        run_id = run.info.run_id
        logger.info(
            "MLflow run '%s' registrado — drift_alert=%s, features com drift: %s",
            run_id,
            alert,
            drift_metrics.get("drifted_features", []),
        )
        return run_id


def log_pipeline_health(
    stage_statuses: dict,
    latencies: dict,
    data_volumes: dict,
    alert_count: int,
    contract_violations: int,
    experiment_name: str = "credit_scoring_monitoring",
) -> str:
    """Registra as métricas de saúde operacional do pipeline no MLflow.

    Centraliza status de execução, latência, volume de dados processados e
    quantidade de alertas, além de anexar um resumo estruturado como artefato.
    """
    mlflow.set_experiment(experiment_name)

    stages_total = len(stage_statuses)
    stages_succeeded = sum(1 for status in stage_statuses.values() if status == "success")
    status = "success" if stages_succeeded == stages_total else "failed"

    with mlflow.start_run(run_name="pipeline_health") as run:
        mlflow.set_tag("pipeline_status", status)

        mlflow.log_metric("pipeline_status_success", int(status == "success"))
        mlflow.log_metric("n_stages", stages_total)
        mlflow.log_metric("n_stages_succeeded", stages_succeeded)
        mlflow.log_metric("n_stages_failed", stages_total - stages_succeeded)

        total_latency = sum(latencies.values())
        mlflow.log_metric("pipeline_latency_seconds", round(total_latency, 4))
        for stage, seconds in latencies.items():
            mlflow.log_metric(f"latency_{stage}_seconds", round(seconds, 4))

        for name, value in data_volumes.items():
            mlflow.log_metric(f"n_samples_{name}", value)

        mlflow.log_metric("n_alerts", alert_count)
        mlflow.log_metric("n_contract_violations", contract_violations)

        summary = {
            "status": status,
            "stage_statuses": stage_statuses,
            "latencies_seconds": {k: round(v, 4) for k, v in latencies.items()},
            "total_latency_seconds": round(total_latency, 4),
            "data_volumes": data_volumes,
            "alert_count": alert_count,
            "contract_violations": contract_violations,
        }
        mlflow.log_dict(summary, "pipeline_run_summary.json")

        run_id = run.info.run_id
        logger.info(
            "MLflow run '%s' registrado — status=%s, latência=%.3fs, alertas=%d",
            run_id,
            status,
            total_latency,
            alert_count,
        )
        return run_id
