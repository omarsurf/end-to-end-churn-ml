from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import load_typed_config, log_loaded_config, project_root, resolve_path
from .exceptions import DataValidationError, ModelNotFoundError
from .logging_config import setup_logging
from .model_registry import ModelRegistry
from .monitoring import ProductionMetricsTracker
from .prepare import clean_total_charges, engineer_features
from .schemas import validate_batch_input, validate_prediction_outputs

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def load_model_with_retry(model_path: Path):
    return joblib.load(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score new customers.")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "config" / "default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV of customers.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV with scores.")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Optional threshold for label."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any input rows are invalid.",
    )
    parser.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="Allow scoring without production model (dev/test only).",
    )
    return parser.parse_args()


def load_threshold(results_path: Path, model_id: str | None = None) -> float | None:
    if not results_path.exists():
        return None
    try:
        results = pd.read_csv(results_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if results.empty or "final_threshold" not in results.columns:
        return None

    selected = results
    if model_id is not None and "model_id" in results.columns:
        model_rows = results[results["model_id"].astype(str) == str(model_id)]
        if not model_rows.empty:
            selected = model_rows
        else:
            logger.warning(
                "No matching threshold row for active model_id; using latest threshold entry.",
                extra={
                    "model_id": model_id,
                    "results_rows": int(len(results)),
                },
            )

    value = selected["final_threshold"].iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _resolve_registry_model_path(root: Path, raw_model_path: str) -> Path:
    model_path = Path(raw_model_path)
    if not model_path.is_absolute():
        model_path = resolve_path(root, model_path)
    return model_path


def _resolve_model_path(
    root: Path, models_dir: Path, cfg, *, allow_unregistered: bool = False
) -> tuple[Path, str | None]:
    fallback_path = models_dir / cfg.artifacts.model_file
    if allow_unregistered:
        logger.warning(
            "Scoring with unregistered model (--allow-unregistered). "
            "This should only be used for dev/test.",
            extra={"model_path": str(fallback_path), "model_id": None},
        )
        return fallback_path, None

    if cfg.registry.enabled and cfg.registry.use_current:
        registry = ModelRegistry(resolve_path(root, cfg.registry.file))
        try:
            production = registry.get_production_model()
        except ModelNotFoundError as exc:
            raise SystemExit(
                "No production model found in registry. Promote a model before running predictions."
            ) from exc

        candidate_path = _resolve_registry_model_path(root, production.model_path)
        if candidate_path.is_absolute():
            try:
                candidate_path.resolve().relative_to(root.resolve())
            except ValueError as exc:
                raise SystemExit(
                    "Production model path is external to project; prediction fallback is disabled."
                ) from exc

        if not candidate_path.exists():
            raise SystemExit(
                "Production model artifact is missing; prediction fallback is disabled."
            )

        return candidate_path, production.model_id

    if cfg.registry.enabled:
        raise SystemExit(
            "registry.enabled=True but registry.use_current=False. "
            "Set use_current=True or use --allow-unregistered for dev/test."
        )

    raise SystemExit(
        "Registry disabled. Production predictions require registry.enabled=True "
        "and a promoted production model. Use --allow-unregistered for dev/test."
    )


def _prediction_required_columns(cfg) -> list[str]:
    target_col = cfg.validation.target_column
    return [
        col for col in cfg.validation.required_columns if col not in {target_col, "customerID"}
    ]


def _prepare_features_for_prediction(df: pd.DataFrame, cfg, models_dir: Path) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = clean_total_charges(df)

    if not cfg.engineering.enabled:
        return df

    medians_path = models_dir / cfg.artifacts.train_medians_file
    if not medians_path.exists():
        raise FileNotFoundError(f"Train medians not found: {medians_path}")

    train_medians = pd.read_json(medians_path, typ="series").to_dict()
    engineered, _ = engineer_features(df, train_medians=train_medians, cfg=cfg)
    return engineered


def main() -> None:
    args = parse_args()
    root = project_root()
    cfg = load_typed_config(args.config)
    pipeline_logger = setup_logging(
        log_file=resolve_path(root, cfg.logging.file),
        level=cfg.logging.level,
        logger_name=cfg.logging.logger_name,
    )
    log_loaded_config(pipeline_logger, cfg, args.config)

    batch_started = time.perf_counter()
    input_rows = 0
    failed_rows = 0
    model_id: str | None = None

    try:
        models_dir = resolve_path(root, cfg.paths.models)
        preprocessor_path = models_dir / cfg.artifacts.preprocessor_file
        if not preprocessor_path.exists():
            raise SystemExit("Preprocessor not found. Run churn-prepare first.")

        preprocessor = joblib.load(preprocessor_path)
        model_required_columns = list(getattr(preprocessor, "feature_names_in_", []))

        model_path, model_id = _resolve_model_path(
            root, models_dir, cfg, allow_unregistered=args.allow_unregistered
        )
        if not model_path.exists():
            raise SystemExit("Model not found. Run churn-train first.")

        input_df = pd.read_csv(args.input)
        input_rows = int(len(input_df))
        if "Churn" in input_df.columns:
            input_df = input_df.drop(columns=["Churn"])

        customer_ids = input_df["customerID"] if "customerID" in input_df.columns else None
        features_df = (
            input_df.drop(columns=["customerID"]) if "customerID" in input_df.columns else input_df
        )

        required_columns = _prediction_required_columns(cfg) if args.strict else []

        valid_df, issues = validate_batch_input(
            features_df,
            required_columns=required_columns,
            strict=args.strict,
        )
        if issues:
            has_batch_issue = any(issue.get("row") is None for issue in issues)
            failed_rows += input_rows if has_batch_issue else len(issues)
            logger.warning("Input validation issues detected", extra={"issues_count": len(issues)})

        # Track which rows failed validation/prediction for transparency
        row_status = pd.Series("failed", index=features_df.index, dtype=str)
        proba_all = pd.Series(float("nan"), index=features_df.index, dtype=float)
        if not valid_df.empty:
            # Mark validated rows as pending prediction
            row_status.loc[valid_df.index] = "pending"
            valid_features = pd.DataFrame()
            try:
                prepared_features = _prepare_features_for_prediction(valid_df, cfg, models_dir)
                if model_required_columns:
                    missing_model_columns = [
                        col for col in model_required_columns if col not in prepared_features.columns
                    ]
                    if missing_model_columns:
                        message = (
                            "Prepared features missing columns required by preprocessor: "
                            f"{missing_model_columns}"
                        )
                        if args.strict:
                            raise DataValidationError(message)
                        failed_rows += len(valid_df)
                        row_status.loc[valid_df.index] = "failed"
                        logger.warning(
                            "Prepared features missing required preprocessor columns; "
                            "neutral fallback applied.",
                            extra={
                                "missing_count": len(missing_model_columns),
                                "missing_sample": missing_model_columns[:10],
                            },
                        )
                    else:
                        valid_features = prepared_features.loc[:, model_required_columns]
                        row_status.loc[valid_features.index] = "pending"
                else:
                    valid_features = prepared_features
                    row_status.loc[valid_features.index] = "pending"
            except Exception as exc:
                if args.strict:
                    raise
                logger.warning(
                    "Feature engineering failed during prediction; neutral fallback applied.",
                    extra={"error": str(exc)},
                )
                failed_rows += len(valid_df)
                row_status.loc[valid_df.index] = "failed"

            if not valid_features.empty:
                model = load_model_with_retry(model_path)
                try:
                    transformed = preprocessor.transform(valid_features)
                    proba = model.predict_proba(transformed)[:, 1]
                    proba_all.loc[valid_features.index] = proba
                    row_status.loc[valid_features.index] = "ok"
                except Exception:
                    if args.strict:
                        raise
                    failed_rows += len(valid_features)
                    row_status.loc[valid_features.index] = "failed"
                    logger.exception("Prediction failed; rows marked as failed.")

        threshold_source = "cli"
        threshold = args.threshold
        if threshold is None:
            threshold = load_threshold(models_dir / cfg.artifacts.final_results_file, model_id=model_id)
            threshold_source = "results_file"
        if threshold is None:
            threshold = 0.5
            threshold_source = "default"

        retained_value = float(
            cfg.business.retained_value or (cfg.business.clv * cfg.business.success_rate)
        )
        contact_cost = float(cfg.business.contact_cost)
        # Internal 'pending' markers must not leak to final outputs.
        row_status = row_status.replace("pending", "failed")

        output = pd.DataFrame({
            "churn_probability": proba_all,
            "prediction_status": row_status,
        })
        # Only compute decisions for successfully predicted rows
        output["churn_prediction"] = pd.NA
        output["decision"] = pd.NA
        output["expected_value"] = pd.NA

        ok_mask = output["prediction_status"] == "ok"
        if ok_mask.any():
            output.loc[ok_mask, "churn_prediction"] = (
                output.loc[ok_mask, "churn_probability"].ge(threshold).astype(int)
            )
            output.loc[ok_mask, "decision"] = output.loc[ok_mask, "churn_prediction"].map(
                {1: "contact", 0: "no_contact"}
            )
            output.loc[ok_mask, "expected_value"] = (
                output.loc[ok_mask, "churn_probability"] * retained_value - contact_cost
            )
        output["model_id"] = model_id
        if customer_ids is not None:
            output.insert(0, "customerID", customer_ids)

        output_issues = validate_prediction_outputs(
            output,
            threshold=threshold,
            strict=args.strict,
        )
        if output_issues:
            logger.warning(
                "Prediction output validation issues detected",
                extra={"issues": output_issues},
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        logger.info(
            "Predictions written",
            extra={
                "output_path": str(args.output),
                "rows": int(len(output)),
                "failed_rows": int(failed_rows),
                "threshold": float(threshold),
                "threshold_source": threshold_source,
                "model_id": model_id,
            },
        )
    except DataValidationError:
        logger.exception("Prediction aborted due to input validation errors.")
        raise
    except Exception:
        logger.exception("Predict stage failed")
        raise
    finally:
        latency_ms = (time.perf_counter() - batch_started) * 1000
        if cfg.monitoring.enabled:
            tracker = ProductionMetricsTracker(resolve_path(root, cfg.monitoring.metrics_file))
            tracker.update_prediction_metrics(
                batch_size=input_rows,
                failed_rows=int(failed_rows),
                latency_ms=float(latency_ms),
            )


if __name__ == "__main__":
    main()
