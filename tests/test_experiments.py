from price_regression import experiments


def test_overfit_gap_and_lasso_zeroes_noise(spark):
    """AC-5: 15 cars + 10 noise columns -> test error far above training error; L1 zeroes noise."""
    result = experiments.overfit_experiment(spark, seed=42)
    none = result.by_name("none")
    assert none.valid_rmse >= 3 * none.train_rmse
    assert none.zero_noise == 0
    l1 = result.by_name("l1")
    assert l1.zero_noise >= 5
    assert l1.valid_rmse < none.valid_rmse  # regularization helped on unseen cars


def test_cross_validator_grid(split):
    """AC-5: 3 regParam x 2 elasticNetParam = 6 cells, each with an average RMSE and one winner."""
    result = experiments.tune_with_cv(split[0], folds=2, seed=42)
    assert len(result.params) == len(result.avg_rmse) == 6
    assert result.best_index == result.avg_rmse.index(min(result.avg_rmse))
    assert set(result.best_params) == {"regParam", "elasticNetParam"}
    assert {p["regParam"] for p in result.params} == {0.0, 0.1, 1.0}


def test_dummy_trap_keep_vs_error(spark, split):
    """AC-6: 'keep' + dropLast cannot give honest p-values; 'error' gives one per coefficient + intercept."""
    result = experiments.dummy_trap(spark, split[0])
    assert result.keep_pvalues_error is not None or result.keep_max_std_error > 1e6
    assert result.error_pvalue_count == result.error_coefficient_count + 1
    assert set(result.keep_coefficients) >= {"body=suv", "body=sedan", "body=hatch"}  # no reference left
    assert "body=hatch" not in result.error_coefficients
