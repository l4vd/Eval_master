def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: builds a tiny random model with transformers (seconds); skipped without it"
    )
