"""English / Korean messages in one place. Both languages must have exactly the same keys."""
from __future__ import annotations

import os
from typing import Mapping

LANGS = ("en", "ko")
ENV_VAR = "PRICE_REGRESSION_LANG"

EN: dict[str, str] = {
    # --- cli / help
    "prog_description": "Step-by-step PySpark MLlib linear regression: predict used-car prices and read RMSE / MAE / R2.",
    "help_step": "step to run (default: all)",
    "help_lang": "output language: en or ko (default: PRICE_REGRESSION_LANG, then LC_ALL/LANG, else en)",
    "help_rows": "number of synthetic cars (default 5000)",
    "help_seed": "random seed (default 42)",
    "help_nonlinear": "add extra depreciation above 150,000 km (a pattern a straight line cannot follow)",
    "help_out": "folder for the saved model and run.json (default artifacts)",
    "help_age": "car age in years (predict)",
    "help_km": "mileage in km (predict)",
    "help_body": "body type: hatch, sedan or suv (predict)",
    "err_rows": "--rows must be at least 50 (got {value}). Try --rows 1000.",
    "err_age_km_required": "predict needs --age, --km and --body, e.g. --age 3 --km 40000 --body suv.",
    "err_not_finite": "{option} must be a finite number (got {value}). Try {option} {example}.",
    "err_generic": "Unexpected error: {error}",
    "starting_spark": "Starting a local Spark session (the first start takes a few seconds)...",
    "mock_notice": "Practice data: synthetic cars generated from a known formula, not real market prices.",
    "unit": "x10k KRW",
    "step_title": "== {step} ==",
    # --- doctor
    "doctor_title": "Environment check",
    "doctor_java_ok": "Java {version} found at {path}",
    "doctor_python": "Python {version}",
    "doctor_pyspark": "PySpark {version}",
    "doctor_tmp_ok": "Temp folder is writable: {path}",
    "doctor_tmp_bad": "Temp folder is NOT writable: {path}. Set SPARK_LOCAL_DIRS to a folder you can write to.",
    "doctor_all_good": "Everything looks good. Next: price-regression all",
    "java_missing": (
        "Java 17 or newer was not found, and Spark needs it.\n"
        "Install it, then run `price-regression doctor` again:\n"
        "  macOS:   brew install openjdk@21\n"
        "  Ubuntu:  apt install openjdk-21-jre-headless (as an administrator)\n"
        "  Windows: winget install EclipseAdoptium.Temurin.21.JDK\n"
        "If Java is installed in an unusual place, set JAVA_HOME to its folder."
    ),
    # --- explore
    "explore_preview": "First 5 rows (price in {unit}):",
    "explore_rows": "{rows} rows, split later into 80% training / 20% test.",
    "explore_formula_title": "Hidden formula that generated the prices (the model never sees it):",
    "explore_formula": "price = 3000 - 150*age - 0.02*km + body bonus + noise(sd 120)",
    "explore_bonus": "body bonus: hatch -400, sedan 0, suv +800",
    "explore_nonlinear": "plus --nonlinear: an extra -0.04 per km above 150,000 km (a bend a straight line cannot follow)",
    "explore_goal": "Goal: learn that formula from the rows alone.",
    # --- train
    "train_split": "Split with seed {seed}: {train} training rows, {test} test rows ({pct}% / {pct_test}%).",
    "train_pipeline": "Pipeline stages: {stages}",
    "train_reference": "Body order {labels}; the last one ({ref}) is the reference category.",
    "train_time": "Fitted in {seconds} s.",
    # --- evaluate
    "eval_baseline_title": "Test-set scores (data the model never trained on):",
    "eval_col_metric": "Metric",
    "eval_col_baseline": "Baseline (training mean)",
    "eval_col_model": "Linear regression",
    "eval_train_title": "Training summary vs test (Spark's model.summary is computed on TRAINING data):",
    "eval_col_train": "Train (summary)",
    "eval_col_test": "Test",
    "eval_so_what": "So what?",
    "eval_mae": "On average the model is off by about {mae} {unit} per car (MAE).",
    "eval_rmse": "RMSE is {rmse} {unit}: like MAE but big misses count more.",
    "eval_improve": "Always guessing the average price would miss by {base} (RMSE), so the model cut the error by {pct}%.",
    "eval_r2": "R2 = {r2}: the model explains {pct}% of why prices differ (1 = perfect, 0 = no better than the average).",
    "eval_noise": "The data has random noise of 120, so an RMSE near 120 means it cannot get better: it learned everything learnable.",
    "eval_summary_note": "Use the Test column to judge the model; the training numbers only show how well it fit data it already saw.",
    # --- coefficients
    "coef_title": "Learned coefficients vs the true formula (reference body type: hatch):",
    "coef_col_name": "Term",
    "coef_col_learned": "Learned",
    "coef_col_true": "True",
    "coef_col_diff": "Difference",
    "coef_intercept": "intercept (a hatch, age 0, 0 km)",
    "coef_read_age": "Reading it: each extra year of age lowers the price by about {value} {unit}.",
    "coef_read_km": "Each extra 10,000 km lowers it by about {value} {unit}.",
    "coef_read_body": "A suv costs about {suv} {unit} more than a hatch, a sedan about {sedan} {unit} more (same age and mileage).",
    "coef_scaler_note": "Note: with StandardScaler the coefficients mean 'per 1 standard deviation', not original units, so they cannot be compared with the formula.",
    "coef_nonlinear_note": "This data has the extra --nonlinear bend, so the learned line is only an average fit and will not match the table exactly.",
    # --- residuals
    "resid_title": "Mean residual (actual - predicted) by mileage band and body type:",
    "resid_col_group": "Group",
    "resid_col_n": "Cars",
    "resid_col_mean": "Mean residual",
    "resid_ok": "Every group is close to 0: the straight line has no blind spot on this data.",
    "resid_bias": "Biggest miss: {group} ({value} {unit}). Positive = the line predicts too low, negative = too high. The data bends there and a straight line cannot follow.",
    # --- overfit
    "overfit_title": "Overfitting: {train} training cars + {noise} useless noise columns, tested on {valid} new cars",
    "overfit_col_model": "Model",
    "overfit_col_train": "Train RMSE",
    "overfit_col_valid": "Test RMSE",
    "overfit_col_zero": "Noise coefs = 0",
    "overfit_none": "no regularization",
    "overfit_l2": "L2 (regParam {reg})",
    "overfit_l1": "L1 (regParam {reg})",
    "overfit_gap": "Without regularization the training error is {train} but the test error is {valid}: it memorized the {rows} cars (overfitting).",
    "overfit_l1_result": "L1 pushed {zero} of {noise} noise coefficients to exactly 0 and cut the test error to {valid}.",
    # --- tune
    "tune_title": "Cross-validation ({folds} folds): average RMSE for each setting",
    "tune_col_reg": "regParam",
    "tune_col_elastic": "elasticNetParam",
    "tune_col_rmse": "Avg RMSE",
    "tune_col_best": "Best",
    "tune_best": "Best setting: regParam={reg}, elasticNetParam={elastic} (RMSE {rmse}).",
    "tune_flat": "The settings differ by only {spread} RMSE: on clean data with plenty of rows, regularization hardly matters.",
    # --- trap
    "trap_title": "Dummy-variable trap: handleInvalid='keep' + dropLast",
    "trap_keep_coefs": "keep  : coefficients {coefs}",
    "trap_error_coefs": "error : coefficients {coefs}",
    "trap_keep_nop": "keep  : p-values cannot be computed (the system is singular).",
    "trap_keep_se": "keep  : p-values exist but the standard errors are absurd (up to {se}), so they mean nothing.",
    "trap_error_p": "error : {count} p-values for {coefs} coefficients + intercept, as expected.",
    "trap_why": "Why: 'keep' adds a hidden 'unknown' column and dropLast removes that one, so the real body columns always add up to 1 - the same as the intercept. The model cannot tell them apart, so coefficients are arbitrary. Use 'error' when you want to interpret coefficients.",
    # --- all / run
    "all_saved": "Saved model to {path} and summary to {run}.",
    "all_done": "Done in {seconds} s.",
    # --- predict
    "predict_no_model": "No saved model in {path}; training one first (seed {seed}, {rows} rows)...",
    "predict_result": "Predicted price: {price} {unit}  (age {age}, {km} km, {body})",
    "predict_unknown_body": "Unknown body type '{body}': the model never saw it while training. Choose one of: {known}.",
    "warn_age_range": "Warning: age is outside the range seen in training (0-15 years); do not trust this number.",
    "warn_km_range": "Warning: mileage is outside the range seen in training (0-200,000 km); do not trust this number.",
    "warn_negative_price": "Warning: the predicted price is negative. A straight-line model is just a formula; it does not know prices cannot go below 0.",
}

KO: dict[str, str] = {
    "prog_description": "PySpark MLlib 선형회귀를 단계별로 따라 하며 중고차 가격을 예측하고 RMSE·MAE·R2 를 읽어 봅니다.",
    "help_step": "실행할 단계 (기본값: all)",
    "help_lang": "출력 언어: en 또는 ko (기본값: PRICE_REGRESSION_LANG, 다음 LC_ALL/LANG, 그 외 en)",
    "help_rows": "만들 가상 자동차 수 (기본 5000)",
    "help_seed": "난수 시드 (기본 42)",
    "help_nonlinear": "15만 km 넘는 구간에 추가 감가를 넣습니다 (직선이 따라가지 못하는 패턴)",
    "help_out": "저장 모델과 run.json 을 둘 폴더 (기본 artifacts)",
    "help_age": "차 연식(년) (predict)",
    "help_km": "주행거리(km) (predict)",
    "help_body": "차종: hatch, sedan, suv (predict)",
    "err_rows": "--rows 는 50 이상이어야 합니다 (입력값 {value}). --rows 1000 처럼 써 보세요.",
    "err_age_km_required": "predict 에는 --age, --km, --body 가 필요합니다. 예: --age 3 --km 40000 --body suv",
    "err_not_finite": "{option} 는 유한한 숫자여야 합니다 (입력값 {value}). {option} {example} 처럼 써 보세요.",
    "err_generic": "예상하지 못한 오류: {error}",
    "starting_spark": "로컬 Spark 세션을 시작합니다 (처음에는 몇 초 걸립니다)...",
    "mock_notice": "연습용 데이터입니다: 정답 공식으로 만든 가상 자동차이며 실제 시장 가격이 아닙니다.",
    "unit": "만 원",
    "step_title": "== {step} ==",
    "doctor_title": "환경 점검",
    "doctor_java_ok": "Java {version} 확인: {path}",
    "doctor_python": "Python {version}",
    "doctor_pyspark": "PySpark {version}",
    "doctor_tmp_ok": "임시 폴더에 쓸 수 있습니다: {path}",
    "doctor_tmp_bad": "임시 폴더에 쓸 수 없습니다: {path}. SPARK_LOCAL_DIRS 를 쓸 수 있는 폴더로 지정하세요.",
    "doctor_all_good": "모두 정상입니다. 다음: price-regression all",
    "java_missing": (
        "Java 17 이상을 찾지 못했습니다. Spark 가 Java 를 필요로 합니다.\n"
        "아래처럼 설치한 뒤 `price-regression doctor` 를 다시 실행하세요.\n"
        "  macOS:   brew install openjdk@21\n"
        "  Ubuntu:  apt install openjdk-21-jre-headless (관리자 권한으로)\n"
        "  Windows: winget install EclipseAdoptium.Temurin.21.JDK\n"
        "Java 를 특이한 위치에 설치했다면 JAVA_HOME 을 그 폴더로 지정하세요."
    ),
    "explore_preview": "처음 5행 (가격 단위: {unit}):",
    "explore_rows": "총 {rows}행. 나중에 학습 80% / 검증 20% 로 나눕니다.",
    "explore_formula_title": "가격을 만든 숨은 공식 (모델은 이 공식을 보지 못합니다):",
    "explore_formula": "가격 = 3000 - 150*연식 - 0.02*주행거리 + 차종 보정 + 잡음(표준편차 120)",
    "explore_bonus": "차종 보정: hatch -400, sedan 0, suv +800",
    "explore_nonlinear": "--nonlinear: 15만 km 초과분마다 km 당 -0.04 추가 (직선이 따라가지 못하는 꺾임)",
    "explore_goal": "목표: 행들만 보고 이 공식을 스스로 알아내기.",
    "train_split": "시드 {seed} 로 분리: 학습 {train}행, 검증 {test}행 ({pct}% / {pct_test}%).",
    "train_pipeline": "Pipeline 단계: {stages}",
    "train_reference": "차종 순서 {labels}. 마지막({ref})이 기준 범주입니다.",
    "train_time": "학습 시간 {seconds}초.",
    "eval_baseline_title": "검증 데이터 점수 (모델이 학습에 쓰지 않은 데이터):",
    "eval_col_metric": "지표",
    "eval_col_baseline": "기준 모델(학습 평균)",
    "eval_col_model": "선형회귀",
    "eval_train_title": "학습 요약 vs 검증 (Spark 의 model.summary 는 '학습 데이터' 기준입니다):",
    "eval_col_train": "학습(summary)",
    "eval_col_test": "검증",
    "eval_so_what": "그래서 무슨 뜻?",
    "eval_mae": "이 모델은 한 대당 평균 약 {mae}{unit} 틀립니다 (MAE).",
    "eval_rmse": "RMSE 는 {rmse}{unit} 입니다. MAE 와 비슷하지만 크게 틀린 것에 더 큰 벌점을 줍니다.",
    "eval_improve": "항상 평균 가격만 찍으면 {base} 정도 틀립니다(RMSE). 모델은 그 오차를 {pct}% 줄였습니다.",
    "eval_r2": "R2 = {r2}: 가격이 서로 다른 이유의 {pct}% 를 설명합니다 (1 이면 완벽, 0 이면 평균과 같음).",
    "eval_noise": "데이터에 잡음(120)을 넣었으므로 RMSE 가 120 근처면 더 줄일 수 없는 한계까지 배운 것입니다.",
    "eval_summary_note": "모델 평가는 '검증' 열로 하세요. 학습 숫자는 이미 본 데이터를 얼마나 잘 맞췄는지만 보여 줍니다.",
    "coef_title": "모델이 찾은 계수 vs 정답 공식 (기준 차종: hatch):",
    "coef_col_name": "항목",
    "coef_col_learned": "학습 결과",
    "coef_col_true": "정답",
    "coef_col_diff": "차이",
    "coef_intercept": "절편 (연식 0, 0 km 인 hatch)",
    "coef_read_age": "읽는 법: 1년 오래될 때마다 가격이 약 {value}{unit} 내려갑니다.",
    "coef_read_km": "1만 km 더 탈 때마다 약 {value}{unit} 내려갑니다.",
    "coef_read_body": "연식·주행거리가 같으면 suv 는 hatch 보다 약 {suv}{unit}, sedan 은 약 {sedan}{unit} 더 비쌉니다.",
    "coef_scaler_note": "참고: StandardScaler 를 쓰면 계수는 '표준편차 1 단위당' 효과라 원래 단위가 아니므로 공식과 바로 비교할 수 없습니다.",
    "coef_nonlinear_note": "이 데이터에는 --nonlinear 꺾임이 들어 있어 직선은 평균적으로만 맞고, 표와 정확히 같지 않습니다.",
    "resid_title": "주행거리 구간·차종별 평균 잔차 (잔차 = 실제 - 예측):",
    "resid_col_group": "구간",
    "resid_col_n": "대수",
    "resid_col_mean": "평균 잔차",
    "resid_ok": "모든 구간이 0 에 가깝습니다: 이 데이터에서는 직선이 놓치는 구간이 없습니다.",
    "resid_bias": "가장 크게 틀린 곳: {group} ({value}{unit}). 양수면 직선이 너무 싸게, 음수면 너무 비싸게 봅니다. 그 구간에서 데이터가 꺾이는데 직선은 따라가지 못합니다.",
    "overfit_title": "과적합: 학습 {train}대 + 쓸모없는 잡음 열 {noise}개, 새 차 {valid}대로 시험",
    "overfit_col_model": "모델",
    "overfit_col_train": "학습 RMSE",
    "overfit_col_valid": "검증 RMSE",
    "overfit_col_zero": "0 이 된 잡음 계수",
    "overfit_none": "정규화 없음",
    "overfit_l2": "L2 (regParam {reg})",
    "overfit_l1": "L1 (regParam {reg})",
    "overfit_gap": "정규화가 없으면 학습 오차는 {train}인데 검증 오차는 {valid}입니다: {rows}대를 통째로 외워 버렸습니다(과적합).",
    "overfit_l1_result": "L1 은 잡음 계수 {noise}개 중 {zero}개를 정확히 0 으로 만들고 검증 오차를 {valid}로 줄였습니다.",
    "tune_title": "교차검증 ({folds}겹): 설정별 평균 RMSE",
    "tune_col_reg": "regParam",
    "tune_col_elastic": "elasticNetParam",
    "tune_col_rmse": "평균 RMSE",
    "tune_col_best": "선택",
    "tune_best": "가장 좋은 설정: regParam={reg}, elasticNetParam={elastic} (RMSE {rmse}).",
    "tune_flat": "설정끼리 RMSE 차이가 {spread} 뿐입니다: 깨끗하고 행이 충분한 데이터에서는 정규화가 거의 영향을 주지 않습니다.",
    "trap_title": "더미 변수 함정: handleInvalid='keep' + dropLast",
    "trap_keep_coefs": "keep  : 계수 {coefs}",
    "trap_error_coefs": "error : 계수 {coefs}",
    "trap_keep_nop": "keep  : p값을 계산할 수 없습니다 (식이 특이 행렬이 됩니다).",
    "trap_keep_se": "keep  : p값은 나왔지만 표준오차가 말도 안 되게 큽니다(최대 {se}). 의미가 없는 숫자입니다.",
    "trap_error_p": "error : 계수 {coefs}개 + 절편에 대해 p값 {count}개, 정상입니다.",
    "trap_why": "이유: 'keep' 은 숨은 '모르는 범주' 열을 만들고 dropLast 가 그 열을 버립니다. 그래서 진짜 차종 열의 합이 항상 1 이 되어 절편과 똑같아지고, 모델이 둘을 구분하지 못해 계수가 제멋대로 나뉩니다. 계수를 해석하려면 'error' 를 쓰세요.",
    "all_saved": "모델을 {path} 에, 요약을 {run} 에 저장했습니다.",
    "all_done": "총 {seconds}초 걸렸습니다.",
    "predict_no_model": "{path} 에 저장된 모델이 없어 먼저 학습합니다 (시드 {seed}, {rows}행)...",
    "predict_result": "예측 가격: {price}{unit}  (연식 {age}년, {km} km, {body})",
    "predict_unknown_body": "모르는 차종 '{body}': 학습할 때 본 적이 없습니다. 다음 중에서 고르세요: {known}.",
    "warn_age_range": "경고: 연식이 학습 때 본 범위(0~15년) 밖입니다. 이 숫자는 믿지 마세요.",
    "warn_km_range": "경고: 주행거리가 학습 때 본 범위(0~200,000 km) 밖입니다. 이 숫자는 믿지 마세요.",
    "warn_negative_price": "경고: 예측 가격이 음수입니다. 직선 모델은 공식일 뿐이라 가격이 0 아래로 갈 수 없다는 걸 모릅니다.",
}

MESSAGES: Mapping[str, Mapping[str, str]] = {"en": EN, "ko": KO}


def detect_lang(cli_value: str | None = None, env: Mapping[str, str] | None = None) -> str:
    """``--lang`` > ``PRICE_REGRESSION_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``; not Korean -> English."""
    env = os.environ if env is None else env
    if cli_value in LANGS:
        return cli_value
    explicit = (env.get(ENV_VAR) or "").strip().lower()
    if explicit in LANGS:
        return explicit
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = (env.get(var) or "").strip()
        if value:  # the first variable that is set decides, like the C library does
            return "ko" if value.lower().startswith("ko") else "en"
    return "en"


def t(lang: str, key: str, **values: object) -> str:
    """Look up a message and fill ``{placeholders}``."""
    return MESSAGES[lang][key].format(**values)
