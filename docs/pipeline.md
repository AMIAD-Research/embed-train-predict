# Pipeline

Three phases, each a subcommand of `etp`: phase 1 (`embed`) writes one `embeddings.h5` for the whole corpus, phase 2 (`train`) fits a linear classifier on it and phase 3 (`predict`) evaluates that classifier. The run folder is the unit that matters, set with `run_directory=` in phase 1 and with `--run-dir` in phases 2 and 3, both pointing at the same folder.

Phase 1 is built on PyTorch Lightning, phases 2 and 3 are plain Python. The diagrams show the functions called in execution order, each labeled with the file it lives in.

🟩 entry point 🟦 pipeline step 🟧 loop ⬜ one-time block 🟪 disk

## 1. embed

Lightning is organized around two objects: the `DataModule` says how the data is loaded and batched, the `EmbedModule` what is computed on each batch. The `PostProcessStep` callback covers everything that happens around the batches: it makes sure the output file is free before any weight is loaded, it collects the embeddings on rank 0 after each batch, and it closes the file when the loop ends. The strategy decides how the model is spread over the hardware, `ddp` replicating it and `deepspeed` sharding it with ZeRO-3.

```mermaid
flowchart TD
    main["<b>etp embed</b><br/>Composes the packaged Hydra config, instantiates everything, runs trainer.test()<br/><br/>etp/cli/embed.py"]

    subgraph setup["one-time setup"]
        direction LR
        sd["<b>DataModule.setup()</b><br/>Instantiates the TableDataset and checks the corpus columns do not collide with the keys the collate function writes<br/><br/>etp/embedder/dataset.py"]
        st["<b>PostProcessStep.setup()</b><br/>Stops the run if embeddings.h5 already holds rows<br/><br/>etp/embedder/callbacks.py"]
        sm["<b>EmbedModule.setup()</b><br/>Loads the model onto the rank's device, ZeRO-3 sharding it under deepspeed<br/><br/>etp/embedder/model.py"]
        sd --> st --> sm
    end

    subgraph loop["once per batch"]
        direction LR
        gi["<b>TableDataset.__getitem__()</b><br/>Converts one corpus row into a dict<br/><br/>etp/embedder/dataset.py"]
        co["<b>text_collate()</b><br/>Wraps each text in the prompt, tokenizes and left-pads into ids and mask<br/><br/>etp/embedder/collate.py"]
        ts["<b>EmbedModule.test_step()</b><br/>Forward pass, one mask-weighted mean vector per hidden state (layer)<br/><br/>etp/embedder/model.py"]
        cb["<b>PostProcessStep.on_test_batch_end()</b><br/>Gathers ids and embeddings on rank 0, drops the padded duplicates, appends them to the output file<br/><br/>etp/embedder/callbacks.py"]
        gi --> co --> ts --> cb
    end

    onend["<b>on_test_end()</b><br/>Closes the file and releases the weights before Lightning copies them back to host RAM<br/><br/>etp/embedder/model.py<br/>etp/embedder/callbacks.py"]
    h5["<b>embeddings</b><br/><br/>&lt;run-dir&gt;/embeddings.h5"]

    main --> setup --> loop --> onend --> h5

    classDef step fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef script fill:#d1fae5,stroke:#059669,color:#111827
    classDef disk fill:#ede9fe,stroke:#7c3aed,color:#111827
    class main script
    class sd,sm,st,gi,co,ts,cb,onend step
    class h5 disk
    style loop fill:#fffbeb,stroke:#d97706,color:#111827
    style setup fill:#f8fafc,stroke:#94a3b8,color:#111827
```

## 2. train

One linear discriminant analysis per layer, fitted on the train split. `--layer` takes a layer index for one layer, `last` for the deepest layer stored, `all` for every layer in the file.

```mermaid
flowchart TD
    main["<b>etp train</b><br/>Reads the train split, resolves --layer against what the embeddings file holds<br/><br/>etp/cli/train.py"]

    subgraph fit["for each layer"]
        direction LR
        le["<b>load_embeddings()</b><br/>One layer as a DataFrame indexed by id<br/><br/>etp/utils.py"]
        ae["<b>align_embeddings()</b><br/>Matches X and y on id, in split order<br/><br/>etp/utils.py"]
        fl["<b>LinearDiscriminantAnalysis().fit()</b><br/>Fits the classifier<br/><br/>etp/cli/train.py"]
        sv["<b>pickle.dump()</b><br/>Saves the fitted classifier under the layer's name<br/><br/>etp/cli/train.py"]
        le --> ae --> fl --> sv
    end

    pkl["<b>fitted classifier</b><br/>&lt;run-dir&gt;/models/lda_&lt;layer&gt;.pkl"]

    main --> fit --> pkl

    classDef step fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef script fill:#d1fae5,stroke:#059669,color:#111827
    classDef disk fill:#ede9fe,stroke:#7c3aed,color:#111827
    class main script
    class le,ae,fl,sv step
    class pkl disk
    style fit fill:#fffbeb,stroke:#d97706,color:#111827
```

## 3. predict

Loads one classifier per layer and evaluates it on the test split. Nothing is fitted here, `etp predict` loads only what `etp train` wrote under `<run-dir>/models/`.

```mermaid
flowchart TD
    main["<b>etp predict</b><br/>Reads the test split, resolves --layer against what the embeddings file holds<br/><br/>etp/cli/predict.py"]

    subgraph score["for each layer"]
        direction LR
        lm["<b>load_model()</b><br/>Unpickles the classifier<br/><br/>etp/cli/predict.py"]
        le["<b>load_embeddings()</b><br/>One layer as a DataFrame indexed by id<br/><br/>etp/utils.py"]
        pas["<b>predict_and_score()</b><br/>Aligns, predicts and gets the class probabilities, score_layer() writes the result table to the csv<br/><br/>etp/cli/predict.py"]
        scm["<b>save_confusion_matrix()</b><br/>Row-normalized percentages summing to 100, out-of-set predictions in their own column<br/><br/>etp/cli/predict.py"]
        pe["<b>log_evaluation()</b><br/>Logs accuracy with its Wilson 95% CI half-width, per-class precision, recall and f1<br/><br/>etp/cli/predict.py"]
        lm --> le --> pas --> scm --> pe
    end

    cm["<b>confusion matrix</b><br/>&lt;run-dir&gt;/scores/lda_&lt;layer&gt;_cm.txt"]
    scores["<b>scores</b><br/>&lt;run-dir&gt;/scores/lda_&lt;layer&gt;.csv"]

    main --> score
    score --> cm
    score --> scores

    classDef step fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef script fill:#d1fae5,stroke:#059669,color:#111827
    classDef disk fill:#ede9fe,stroke:#7c3aed,color:#111827
    class main script
    class lm,le,pas,scm,pe step
    class cm,scores disk
    style score fill:#fffbeb,stroke:#d97706,color:#111827
```
