import argparse
import logging
from pathlib import Path

from lightning.pytorch.utilities.rank_zero import rank_zero_only

from etp.cli.embed import run_embed
from etp.cli.predict import run_predict
from etp.cli.train import run_train


def _setup_logging() -> None:
    level = logging.INFO if rank_zero_only.rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("lightning", "lightning.fabric", "lightning.pytorch"):
        logging.getLogger(name).setLevel(level)


def _add_common_args(parser):
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--corpus-split", required=True, type=str)
    parser.add_argument("--label-column", default="language_l1", type=str)
    parser.add_argument("--layer", default="last", type=str)


def _add_embed(subparsers):
    p = subparsers.add_parser("embed")
    return p


def _add_train(subparsers):
    p = subparsers.add_parser("train")
    _add_common_args(p)
    return p


def _add_predict(subparsers):
    p = subparsers.add_parser("predict")
    _add_common_args(p)
    return p


def build_parser():
    parser = argparse.ArgumentParser(prog="etp")
    subparsers = parser.add_subparsers(dest="phase", required=True)
    _add_embed(subparsers)
    _add_train(subparsers)
    _add_predict(subparsers)
    return parser


def cli():
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    _setup_logging()

    match args.phase:
        case "embed":
            run_embed(overrides)
        case "predict":
            run_predict(
                args.corpus_dir,
                args.run_dir,
                args.corpus_split,
                args.label_column,
                args.layer,
            )
        case "train":
            run_train(
                args.corpus_dir,
                args.run_dir,
                args.corpus_split,
                args.label_column,
                args.layer,
            )
