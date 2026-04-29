"""pytest configuration for characterization tests.

Registers two command-line options so that paths to the reference repos are
passed explicitly at run time:

    pytest tests/characterization/ \\
        --spectrawl-path /home/you/spectrawl \\
        --metabo-src-path /home/you/metabo-depthcharge/src
"""

import sys


def pytest_addoption(parser):
    parser.addoption(
        "--spectrawl-path",
        required=True,
        help="Absolute path to the spectrawl repo root",
    )
    parser.addoption(
        "--metabo-src-path",
        required=True,
        help="Absolute path to metabo-depthcharge/src",
    )


def pytest_configure(config):
    """Insert paths before collection so top-level imports in test modules work."""
    try:
        spectrawl_path = config.getoption("--spectrawl-path")
        metabo_src_path = config.getoption("--metabo-src-path")
    except ValueError:
        return
    sys.path.insert(0, spectrawl_path)
    sys.path.insert(0, metabo_src_path)
