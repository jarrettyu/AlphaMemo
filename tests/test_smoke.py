from sspm.runner import RunConfig, run_search


def test_alphamemo_smoke():
    payload = run_search(
        RunConfig(strategy="alphamemo", budget=12, batch_size=4, seed=7, n_days=180, n_assets=40),
        verbose=False,
    )
    assert payload["summary"]["budget"] == 12
    assert payload["summary"]["n_ok"] > 0
