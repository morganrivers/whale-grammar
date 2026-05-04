"""Pytest configuration: register custom marks."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end tests that require ELKI + JRE; "
        "first run takes ~10 min, subsequent runs use the cache.",
    )
