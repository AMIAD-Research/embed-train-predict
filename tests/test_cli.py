"""CPU-only tests of the command line itself: dispatch and Hydra config lookup.

The phases they dispatch to are covered by test_embed.py, test_train.py and
test_predict.py.
"""

from pathlib import Path

import pytest
from conftest import CONFIG_DIR, run_cli


def test_help_lists_the_three_phases():
    r = run_cli("--help")
    assert r.returncode == 0, r.stderr
    for phase in ("embed", "train", "predict"):
        assert phase in r.stdout


def test_no_phase_is_an_error():
    r = run_cli()
    assert r.returncode != 0


def test_train_and_predict_take_the_same_flags():
    # both phases read the same run dir, corpus and split, so the flags that
    # name them are declared once and must stay identical between the two
    from etp.cli.main import build_parser

    def parsed(phase):
        argv = [phase, "--run-dir", "r", "--corpus-dir", "c", "--corpus-split", "s"]
        args = build_parser().parse_args(argv)
        return {k: v for k, v in vars(args).items() if k != "phase"}

    assert parsed("train") == parsed("predict")
    assert parsed("train") == {
        "run_dir": Path("r"),
        "corpus_dir": Path("c"),
        "corpus_split": "s",
        "label_column": "language_l1",  # the clc-fce default
        "layer": "last",
    }


def test_run_dir_corpus_dir_and_split_are_all_required():
    from etp.cli.main import build_parser

    for missing in ("--run-dir", "--corpus-dir", "--corpus-split"):
        argv = ["train", "--run-dir", "r", "--corpus-dir", "c", "--corpus-split", "s"]
        i = argv.index(missing)
        del argv[i : i + 2]
        with pytest.raises(SystemExit):
            build_parser().parse_args(argv)


def test_embed_passes_the_rest_of_the_line_to_hydra():
    # phase 1 has no flags of its own: everything after `etp embed` is a Hydra
    # override, dotted key and all, and must survive argparse untouched
    from etp.cli.main import build_parser

    overrides = ["model_name_or_path=/x", "run_directory=/y", "trainer.devices=4"]
    args, extras = build_parser().parse_known_args(["embed", *overrides])

    assert args.phase == "embed"
    assert extras == overrides


def test_the_embed_config_ships_inside_the_package():
    # it used to be composed from conf/ in the current working directory, which
    # tied `etp embed` to being launched from the repository root. It now lives
    # next to the code that reads it, so the command runs from anywhere.
    assert (CONFIG_DIR / "config.yaml").exists()
    for group in ("corpus", "prompt", "strategy"):
        assert (CONFIG_DIR / group).is_dir()


def test_embed_composes_from_any_working_directory(tmp_path):
    # launched from an empty directory, the run must get past composition and
    # fail on the corpus it cannot find there, not on the config it cannot find
    r = run_cli(
        "embed",
        "model_name_or_path=/no/such/model",
        f"run_directory={tmp_path / 'run'}",
        "trainer.accelerator=cpu",
        cwd=tmp_path,
    )
    assert r.returncode != 0
    combined = r.stdout + r.stderr

    # the clc-fce corpus group composed and resolved: what is missing is its data
    assert "corpus_list" in combined and "FileNotFoundError" in combined
    # and not the config itself
    assert "Cannot find primary config" not in combined
    assert "Primary config directory not found" not in combined


def test_the_job_log_is_left_to_rank_zero(monkeypatch):
    # the ranks of a distributed run share one job log. Rank 0 writes its INFO
    # lines to it, the others are held at WARNING, or an embed run on 16 ranks
    # repeats every line sixteen times, the model repr EmbedModule.setup() logs
    # included
    import logging

    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    from etp.cli.main import _setup_logging

    # named here rather than imported from etp.cli.main: iterating the very
    # tuple under test would pass whatever that tuple happens to say. Lightning
    # gives each of these a level and a handler of its own when it is imported,
    # which the root level never reaches, so each has to be turned down by name
    lightning_loggers = ("lightning", "lightning.fabric", "lightning.pytorch")

    names = ("", *lightning_loggers)
    levels = {name: logging.getLogger(name).level for name in names}
    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        for rank, expected in ((0, logging.INFO), (3, logging.WARNING)):
            monkeypatch.setattr(rank_zero_only, "rank", rank)
            root.handlers.clear()  # basicConfig is a no-op on a configured root
            _setup_logging()

            assert root.level == expected
            for name in lightning_loggers:
                assert logging.getLogger(name).level == expected
            assert logging.getLogger("etp.cli.predict").getEffectiveLevel() == expected
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        root.handlers[:] = handlers
