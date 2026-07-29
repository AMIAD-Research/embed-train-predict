# embed-train-predict: Task-agnostic text classification from frozen language model representations

Official repository for the paper "Using large language models to provide effective embeddings for native language identification" (Auger et al., 2026).

`etp` implements the text embedder experiments from the paper. Each phase is a subcommand, and maps onto one of its sections:

| Phase | Paper | Writes |
|---|---|---|
| `embed` | Section 3, single forward pass and mean pooling over tokens, one vector per layer | `<run-dir>/embeddings.h5` |
| `train` | Section 3, LDA head fitted on training split, one classifier for a model, a corpus and a layer | `<run-dir>/models/lda_<layer>.pkl` |
| `predict` | Section 6, accuracy scores, confidence intervals and confusion matrices | `<run-dir>/scores/lda_<layer>.csv`, `_cm.txt` |

A run produces the following:
- `embeddings.h5` holds the whole corpus, all splits together: `/ids` of shape `(N,)`, and one `/embeddings/layer_<i>` dataset per layer, of shape `(N, D)`, float32.
- `models/lda_<layer>.pkl` is the fitted scikit-learn `LinearDiscriminantAnalysis`, one file per trained layer, read back by the predict phase.
- `scores/lda_<layer>.csv` gives `id`, `reference`, `hypothesis` and one probability column per class.
- `scores/lda_<layer>_cm.txt` gives the confusion matrix as row-normalized percentages, each row summing to exactly 100.
- Accuracy with its Wilson 95% CI half-width and the per-class report are logged, one per scored layer.

The CLC-FCE split shipped [here](scripts/clc-fce-split) is the one used for the paper. A run reproduces the published scores, within the confidence interval the paper reports. [docs/pipeline.md](docs/pipeline.md) describes the phases function by function.

## Installation

```bash
uv sync                                 # .venv from uv.lock (Python 3.14)
source .venv/bin/activate
```

Network is needed at install time. `uv sync` installs the project, so `etp` is on the venv `PATH`.

## Quickstart

Native language identification ships as the worked example. Models that do not fit on one GPU or node are sharded with [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) ZeRO-3.

### Corpus

The worked example uses the public [CLC-FCE dataset](https://ilexir.co.uk/datasets/index.html) (Yannakoudakis et al., 2011). One command downloads it and builds the corpus under `data/clc-fce`:

```bash
python scripts/prepare_clc_fce_data.py  # --zip <archive> on an offline machine
```

The split is published in [scripts/clc-fce-split/](scripts/clc-fce-split) by CLC-FCE sortkey and derive from the CLC-FCE dataset and stay under its license, non-commercial research and educational use only. No text, annotation or metadata of the dataset is redistributed here. If you use this corpus, please cite the paper:

```bibtex
@inproceedings{yannakoudakis-etal-2011-new,
author = {Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben},
booktitle = {The 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
title = {{A New Dataset and Method for Automatically Grading ESOL Texts}},
year = {2011}
}
```

### Option 1: Single GPU

Everything a run produces lives under one `RUN_DIR` folder. A folder whose `embeddings.h5` already holds rows is refused, so that two runs never mix. Naming that folder with `${SLURM_JOB_ID}` keeps them apart.

```bash
RUN_DIR=out/${SLURM_JOB_ID}_clc-fce_Meta-Llama-3-8B-Instruct

etp embed \
    model_name_or_path=/path/hf/Meta-Llama-3-8B-Instruct \
    run_directory=${RUN_DIR}

etp train \
    --run-dir ${RUN_DIR} \
    --corpus-dir data/clc-fce \
    --corpus-split list/clc-fce-train.csv

etp predict \
    --run-dir ${RUN_DIR} \
    --corpus-dir data/clc-fce \
    --corpus-split list/clc-fce-test.csv
```

### Option 2: Multi-GPU

The DeepSpeed package stays out of `uv.lock` because its build needs torch and `nvcc` already in place.

#### DeepSpeed installation

`uv.lock` pins the CUDA 13 wheels of torch, so the toolkit loaded here has to come from the `cuda/13.x` family.

```bash
module avail cuda                       # what the machine offers
module load cuda/13.x                   # load a CUDA 13.x
nvcc --version                          # confirm a CUDA 13.x before building

uv pip install deepspeed --no-build-isolation
```

#### Sharded run

For a sharded run, CUDA has to be loaded first.

```bash
module load cuda/13.x

etp embed \
    strategy=deepspeed \
    model_name_or_path=/path/hf/Llama-3.1-405B \
    run_directory=${RUN_DIR} \
    trainer.devices=4 \
    trainer.num_nodes=4
```

## Config

### The embed phase

Everything after `etp embed` is passed to Hydra as an override. Each group is a directory in [etp/cli/config/](etp/cli/config), each option a file inside it.

| Group | Options | Effect |
|---|---|---|
| `corpus` | `clc-fce` (default) | the tables to load, plus `corpus_name` |
| `prompt` | `none` (default), `instruction` | instruction added to the model input |
| `strategy` | `ddp` (default), `deepspeed` | replication, or ZeRO-3 sharding |

| Key | Default | Effect |
|---|---|---|
| `model_name_or_path` | required | Hugging Face model directory or hub id |
| `run_directory` | required | the run folder, read back by phases 2 and 3 as `--run-dir` |
| `corpus_directory` | `<cwd>/data` | holds the corpus folders |
| `corpus_name` | `clc-fce` | corpus directory name |
| `dataset.limit` | `null` | first N rows only |
| `datamodule.test_dataloader_cfg.batch_size` | `1` | texts per forward pass |
| `datamodule.test_dataloader_cfg.num_workers` | `8` | dataloader workers |
| `trainer.devices` / `trainer.num_nodes` | `1` / `1` | GPUs per node, nodes |

### The train and predict phases

| Flag | Default | Effect |
|---|---|---|
| `--run-dir` | required | the folder `etp embed` wrote |
| `--corpus-dir` | required | the corpus, e.g. `data/clc-fce` |
| `--corpus-split` | required | split file, relative to `--corpus-dir` |
| `--label-column` | `language_l1` | annotation column used as class label |
| `--layer` | `last` | a layer index, `last`, or `all` for every layer in the file |

## Run on a new task

Data has to be formatted with lists of split ids (columns `id`, `annotation_file`) pointing to an `annotation_file`, which needs a unique `id`, a text and a label column. One list csv per split, laid out like the worked example. Add an `etp/cli/config/corpus/mytask.yaml` file, copied from [clc-fce.yaml](etp/cli/config/corpus/clc-fce.yaml) and select it with `corpus=mytask`.

## Development and AI assistance

This library was developed with the help of [Claude Code](https://claude.com/claude-code). The development choices are the author's: the three-phase design, the config and corpus layout, the modeling decisions and the reading of the results. The assistant was used on top of those choices, to optimize the compute-intensive parts of the pipeline, to review code and to catch errors. Every change it proposed was read, tested and then accepted or rejected by the author, who remains responsible for the code as it stands.

## Citing

If you use this code, please cite the paper:

```bibtex
@inproceedings{auger-etal-2026-using,
    title = "Using large language models to provide effective embeddings for native language identification",
    author = "Auger, Roxane and Boeffard, Olivier and Bonastre, Jean-Fran{\c{c}}ois",
    booktitle = "Findings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    year = "2026",
    publisher = "Association for Computational Linguistics"
}
```
