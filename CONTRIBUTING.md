# Contributing

Issues and pull requests are welcome. A change is ready when the suite and the lint below both pass.

## Tests

CPU-only, no network, no GPU. The tokenizer and the model are replaced by the stubs of [tests/stubs.py](tests/stubs.py), every phase runs on a synthetic corpus written to a temporary directory, and the three phases are also exercised end to end through the `etp` command line itself.

```bash
pytest                                  # the suite, about a minute
ruff check . && ruff format --check .   # the lint
```

The same two commands run in continuous integration, on every push and pull request, from [.github/workflows/tests.yml](.github/workflows/tests.yml).

## Layout

```text
docs/pipeline.md       the three phases, function by function (Mermaid)
etp/
  cli/
    main.py            the `etp` command: parses, dispatches
    embed.py           phase 1: composes the config, runs the Lightning trainer
    train.py           phase 2: fits an LDA per layer, pickles it
    predict.py         phase 3: applies the LDA, scores, reports
    config/            the embed Hydra config: corpus/, prompt/, strategy/
  utils.py             corpus tables, embeddings, label alignment (phases 2 and 3)
  embedder/            phase 1 internals: dataset, collate, model, callbacks, writers, zero3
scripts/               corpus preparation and clc-fce split
tests/                 the CPU-only suite
```
