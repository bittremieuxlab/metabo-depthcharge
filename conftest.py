import sys


def pytest_addoption(parser):
    parser.addoption(
        "--spectrawl-path",
        default=None,
        help="Absolute path to the spectrawl repo root (required for characterization tests)",
    )
    parser.addoption(
        "--metabo-src-path",
        default=None,
        help="Absolute path to metabo-depthcharge/src (required for characterization tests)",
    )


def pytest_configure(config):
    try:
        spectrawl_path = config.getoption("--spectrawl-path")
        metabo_src_path = config.getoption("--metabo-src-path")
    except ValueError:
        return
    if spectrawl_path:
        sys.path.insert(0, spectrawl_path)
    if metabo_src_path:
        sys.path.insert(0, metabo_src_path)


def pytest_ignore_collect(collection_path, config):
    if "characterization" in str(collection_path):
        if not config.getoption("--spectrawl-path", default=None):
            return True
    return None
