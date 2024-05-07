"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import logging
from pathlib import Path

import cloudpickle as pickle

log = logging.getLogger(__name__)


def process_session(path: Path) -> None:
    log.critical(f"Executing {path}")
    # Load the path pickle. It should be a tuple of (run_fn, runner_kwargs, config)
    with path.open("rb") as file:
        loaded = pickle.load(file)

    if not isinstance(loaded, tuple):
        raise TypeError(f"Expected a tuple, got {type(loaded)}")

    if not len(loaded) == 3:
        raise ValueError(f"Expected a tuple of length 3, got {len(loaded)}")

    run_fn, runner_kwargs, config = loaded
    assert callable(run_fn), f"Expected a callable, got {type(run_fn)}"
    assert isinstance(
        runner_kwargs, dict
    ), f"Expected a dict, got {type(runner_kwargs)}"

    # Execute the run_fn
    from jmp.lightning.runner import Runner

    runner = Runner(run_fn, **runner_kwargs)
    _ = runner([config])
    log.critical(f"Executed {path}")


def main():
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "paths", nargs="+", type=Path, help="Paths to the sessions to run"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.paths:
        raise ValueError("No paths provided")

    log.critical(f"Executing {args.paths=}")

    for path in args.paths:
        process_session(path)

    log.critical("All sessions executed")


if __name__ == "__main__":
    main()
