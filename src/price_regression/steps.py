"""One function per CLI step. Each reads top to bottom: compute with Spark, then print."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession

from . import doctor, experiments
from .data import TRUE_FORMULA, make_cars, true_coefficients
from .evaluate import (
    Metrics, baseline_metrics, evaluate_predictions, improvement, train_summary_metrics,
)
from .i18n import t
from .pipeline import TRAIN_FRACTION, body_labels, fit_model, named_coefficients, split_train_test
from .predict import UnknownBodyError, load_model, predict_price, save_model
from .report import fmt, table
from .residuals import residuals_by_body, residuals_by_km
from .session import default_tmp_dir, get_spark

NOISE_SD = TRUE_FORMULA["noise_sd"]
EXIT_OK, EXIT_ERROR, EXIT_ENV, EXIT_INPUT = 0, 1, 2, 3


@dataclass
class Options:
    lang: str = "en"
    rows: int = 5000
    seed: int = 42
    nonlinear: bool = False
    out: str = "artifacts"


@dataclass
class Run:
    """Holds the Spark session and the objects the steps share (data, split, model)."""

    opts: Options
    spark: SparkSession
    summary: dict = field(default_factory=dict)
    _df: DataFrame | None = None
    _split: tuple[DataFrame, DataFrame] | None = None
    _model: PipelineModel | None = None

    def say(self, key: str, **values: object) -> None:
        print(t(self.opts.lang, key, **values))

    def title(self, step: str) -> None:
        print()
        self.say("step_title", step=step)

    @property
    def unit(self) -> str:
        return t(self.opts.lang, "unit")

    @property
    def df(self) -> DataFrame:
        if self._df is None:
            self._df = make_cars(self.spark, self.opts.rows, self.opts.seed, self.opts.nonlinear)
        return self._df

    @property
    def split(self) -> tuple[DataFrame, DataFrame]:
        if self._split is None:
            self._split = split_train_test(self.df, self.opts.seed)
        return self._split

    @property
    def model(self) -> PipelineModel:
        if self._model is None:
            self._model = fit_model(self.split[0], solver="normal")
        return self._model


# --------------------------------------------------------------------------- doctor
def check_java(lang: str) -> int | None:
    """Return None when Java 17+ is usable, otherwise print the install guide and return 2."""
    info = doctor.find_java()
    if info is None:
        print(t(lang, "java_missing"))
        return EXIT_ENV
    doctor.apply_java_home(info)
    return None


def step_doctor(lang: str) -> int:
    print(t(lang, "doctor_title"))
    info = doctor.find_java()
    if info is None:
        print(t(lang, "java_missing"))
        return EXIT_ENV
    doctor.apply_java_home(info)
    print("[ok] " + t(lang, "doctor_java_ok", version=info.version_text, path=info.executable))
    print("[ok] " + t(lang, "doctor_python", version=doctor.python_version()))
    print("[ok] " + t(lang, "doctor_pyspark", version=doctor.pyspark_version()))
    tmp = default_tmp_dir()
    if not doctor.tmp_writable(tmp):
        print("[!!] " + t(lang, "doctor_tmp_bad", path=tmp))
        return EXIT_ENV
    print("[ok] " + t(lang, "doctor_tmp_ok", path=tmp))
    print(t(lang, "doctor_all_good"))
    return EXIT_OK


# --------------------------------------------------------------------------- explore
def step_explore(run: Run) -> None:
    run.title("explore")
    run.say("mock_notice")
    run.say("explore_preview", unit=run.unit)
    rows = run.df.orderBy("id").limit(5).collect()
    print(table(["id", "age", "km", "body", "price"],
                [(r["id"], f"{r['age']:.0f}", f"{r['km']:,.0f}", r["body"], fmt(r["price"])) for r in rows],
                right_from=4))
    run.say("explore_rows", rows=run.df.count())
    run.say("explore_formula_title")
    run.say("explore_formula")
    run.say("explore_bonus")
    if run.opts.nonlinear:
        run.say("explore_nonlinear")
    run.say("explore_goal")


# --------------------------------------------------------------------------- train
def step_train(run: Run) -> None:
    run.title("train")
    train, test = run.split
    n_train, n_test = train.count(), test.count()
    pct = round(100 * TRAIN_FRACTION)
    run.say("train_split", seed=run.opts.seed, train=n_train, test=n_test, pct=pct, pct_test=100 - pct)
    started = time.time()
    model = run.model
    stages = " -> ".join(type(s).__name__ for s in model.stages)
    run.say("train_pipeline", stages=stages)
    labels = body_labels(model)
    run.say("train_reference", labels="[" + ", ".join(labels) + "]", ref=labels[-1])
    run.say("train_time", seconds=fmt(time.time() - started))
    run.summary.update(train_rows=n_train, test_rows=n_test)


# --------------------------------------------------------------------------- evaluate
def step_evaluate(run: Run) -> None:
    run.title("evaluate")
    train, test = run.split
    run.summary.update(train_rows=train.count(), test_rows=test.count())
    model = run.model
    test_m = evaluate_predictions(model.transform(test))
    base_m = baseline_metrics(train, test)
    train_m = train_summary_metrics(model)
    gain = improvement(test_m.rmse, base_m.rmse)

    lang = run.opts.lang
    run.say("eval_baseline_title")
    print(table(
        [t(lang, "eval_col_metric"), t(lang, "eval_col_baseline"), t(lang, "eval_col_model")],
        [("RMSE", fmt(base_m.rmse), fmt(test_m.rmse)),
         ("MAE", fmt(base_m.mae), fmt(test_m.mae)),
         ("R²", fmt(base_m.r2, 4), fmt(test_m.r2, 4))],
    ))
    print()
    run.say("eval_train_title")
    print(table(
        [t(lang, "eval_col_metric"), t(lang, "eval_col_train"), t(lang, "eval_col_test")],
        [("RMSE", fmt(train_m.rmse), fmt(test_m.rmse)),
         ("R²", fmt(train_m.r2, 4), fmt(test_m.r2, 4))],
    ))
    print()
    run.say("eval_so_what")
    run.say("eval_mae", mae=fmt(test_m.mae, 0), unit=run.unit)
    run.say("eval_rmse", rmse=fmt(test_m.rmse, 0), unit=run.unit)
    run.say("eval_improve", base=fmt(base_m.rmse, 0), pct=fmt(100 * gain, 0))
    run.say("eval_r2", r2=fmt(test_m.r2, 4), pct=fmt(100 * test_m.r2, 1))
    if not run.opts.nonlinear and 0.8 * NOISE_SD <= test_m.rmse <= 1.2 * NOISE_SD:
        run.say("eval_noise")
    run.say("eval_summary_note")
    run.summary.update(
        test=test_m.as_dict(), train=train_m.as_dict(), baseline=base_m.as_dict(), improvement=gain,
    )


# --------------------------------------------------------------------------- coefficients
def step_coefficients(run: Run) -> None:
    run.title("coefficients")
    lang = run.opts.lang
    learned = named_coefficients(run.model)
    truth = true_coefficients()
    names = {"intercept": t(lang, "coef_intercept")}
    rows = []
    for key in ("intercept", "age", "km", "body=suv", "body=sedan"):
        digits = 4 if key == "km" else 1
        rows.append((names.get(key, key), fmt(learned[key], digits), fmt(truth[key], digits),
                     fmt(learned[key] - truth[key], digits)))
    run.say("coef_title")
    print(table([t(lang, "coef_col_name"), t(lang, "coef_col_learned"), t(lang, "coef_col_true"),
                 t(lang, "coef_col_diff")], rows))
    print()
    run.say("coef_read_age", value=fmt(abs(learned["age"]), 0), unit=run.unit)
    run.say("coef_read_km", value=fmt(abs(learned["km"]) * 10_000, 0), unit=run.unit)
    run.say("coef_read_body", suv=fmt(learned["body=suv"], 0), sedan=fmt(learned["body=sedan"], 0), unit=run.unit)
    run.say("coef_scaler_note")
    if run.opts.nonlinear:
        run.say("coef_nonlinear_note")
    run.summary["coefficients"] = learned


# --------------------------------------------------------------------------- residuals
def step_residuals(run: Run) -> None:
    run.title("residuals")
    lang = run.opts.lang
    pred = run.model.transform(run.split[1])
    km_rows = residuals_by_km(pred)
    body_rows = residuals_by_body(pred)
    run.say("resid_title")
    print(table(
        [t(lang, "resid_col_group"), t(lang, "resid_col_n"), t(lang, "resid_col_mean")],
        [(g, n, fmt(m, 0)) for g, n, m in km_rows + body_rows],
    ))
    print()
    group, _, worst = max(km_rows + body_rows, key=lambda r: abs(r[2]))
    if abs(worst) < 40:
        run.say("resid_ok")
    else:
        run.say("resid_bias", group=group, value=fmt(worst, 0), unit=run.unit)
    run.summary["residuals_by_km"] = {g: m for g, _, m in km_rows}


# --------------------------------------------------------------------------- overfit
def step_overfit(run: Run) -> None:
    run.title("overfit")
    lang = run.opts.lang
    result = experiments.overfit_experiment(run.spark, run.opts.seed)
    run.say("overfit_title", train=experiments.OVERFIT_TRAIN_ROWS, noise=result.n_noise,
            valid=experiments.OVERFIT_VALID_ROWS)
    rows = []
    for r in result.rows:
        label = t(lang, "overfit_none") if r.name == "none" else t(lang, f"overfit_{r.name}", reg=fmt(r.reg_param, 0))
        rows.append((label, fmt(r.train_rmse, 0), fmt(r.valid_rmse, 0), f"{r.zero_noise}/{result.n_noise}"))
    print(table([t(lang, "overfit_col_model"), t(lang, "overfit_col_train"), t(lang, "overfit_col_valid"),
                 t(lang, "overfit_col_zero")], rows))
    print()
    none, l1 = result.by_name("none"), result.by_name("l1")
    run.say("overfit_gap", train=fmt(none.train_rmse, 0), valid=fmt(none.valid_rmse, 0),
            rows=experiments.OVERFIT_TRAIN_ROWS)
    run.say("overfit_l1_result", zero=l1.zero_noise, noise=result.n_noise, valid=fmt(l1.valid_rmse, 0))
    run.summary["overfit"] = {r.name: {"train_rmse": r.train_rmse, "valid_rmse": r.valid_rmse,
                                       "zero_noise": r.zero_noise} for r in result.rows}


# --------------------------------------------------------------------------- tune
def step_tune(run: Run, folds: int = 3) -> None:
    run.title("tune")
    lang = run.opts.lang
    result = experiments.tune_with_cv(run.split[0], folds=folds, seed=run.opts.seed)
    run.say("tune_title", folds=result.folds)
    rows = [(fmt(p["regParam"], 1), fmt(p["elasticNetParam"], 0), fmt(m, 2), "*" if i == result.best_index else "")
            for i, (p, m) in enumerate(zip(result.params, result.avg_rmse))]
    print(table([t(lang, "tune_col_reg"), t(lang, "tune_col_elastic"), t(lang, "tune_col_rmse"),
                 t(lang, "tune_col_best")], rows, right_from=0))
    print()
    best = result.best_params
    run.say("tune_best", reg=best["regParam"], elastic=best["elasticNetParam"],
            rmse=fmt(result.avg_rmse[result.best_index], 2))
    spread = max(result.avg_rmse) - min(result.avg_rmse)
    if spread < 1.0:
        run.say("tune_flat", spread=fmt(spread, 2))
    run.summary["cv"] = {"avg_rmse": result.avg_rmse, "best": best}


# --------------------------------------------------------------------------- trap
def step_trap(run: Run) -> None:
    run.title("trap")
    result = experiments.dummy_trap(run.spark, run.split[0])
    run.say("trap_title")

    def short(coefs: dict[str, float]) -> str:
        return ", ".join(f"{k}={fmt(v, 0)}" for k, v in coefs.items() if k != "age" and k != "km")

    run.say("trap_keep_coefs", coefs=short(result.keep_coefficients))
    run.say("trap_error_coefs", coefs=short(result.error_coefficients))
    if result.keep_pvalues_error is not None:
        run.say("trap_keep_nop")
    else:
        run.say("trap_keep_se", se=f"{result.keep_max_std_error:.1e}")
    run.say("trap_error_p", count=result.error_pvalue_count, coefs=result.error_coefficient_count)
    run.say("trap_why")


# --------------------------------------------------------------------------- predict
def ensure_model(run: Run) -> PipelineModel:
    """Load the saved model, or train and save one if none exists yet."""
    model = load_model(run.opts.out)
    if model is None:
        from .predict import model_path
        run.say("predict_no_model", path=model_path(run.opts.out), seed=run.opts.seed, rows=run.opts.rows)
        model = run.model
        save_model(model, run.opts.out)
    return model


def step_predict(run: Run, age: float, km: float, body: str) -> int:
    run.title("predict")
    model = ensure_model(run)
    try:
        result = predict_price(run.spark, model, age, km, body)
    except UnknownBodyError as exc:
        run.say("predict_unknown_body", body=exc.body, known=", ".join(exc.known))
        return EXIT_INPUT
    run.say("predict_result", price=fmt(result.price, 1), unit=run.unit, age=fmt(age, 0), km=f"{km:,.0f}", body=body)
    for key in result.warnings:
        run.say(key)
    return EXIT_OK


# --------------------------------------------------------------------------- all
def step_all(run: Run) -> None:
    started = time.time()
    step_explore(run)
    step_train(run)
    step_evaluate(run)
    step_coefficients(run)
    step_residuals(run)
    step_overfit(run)
    step_tune(run)
    step_trap(run)
    out = Path(run.opts.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = save_model(run.model, out)
    run.summary.update(
        rows=run.opts.rows, seed=run.opts.seed, nonlinear=run.opts.nonlinear, lang=run.opts.lang,
        elapsed_seconds=round(time.time() - started, 1),
    )
    run_json = out / "run.json"
    run_json.write_text(json.dumps(run.summary, indent=2, ensure_ascii=False))
    print()
    run.say("all_saved", path=model_dir, run=run_json)
    run.say("all_done", seconds=fmt(time.time() - started))


STEPS = ("doctor", "explore", "train", "evaluate", "coefficients", "residuals", "overfit", "tune", "trap",
         "predict", "all")
