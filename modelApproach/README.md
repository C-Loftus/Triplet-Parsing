# Explanation of this directory
This directory contains the implementation of a spacy model for relation extraction. This code can be ran directly in the folder or from one up with the `api.py` file. 
Before training the model, make sure you have placed the spacy-formatted annotations in the `assets` directory. 
Then run `spacy project run all` to perform training and validation.

To test this model in this directory from the command line without training further run
```
python .\getTriplets.py .\training\model-best data\test.spacy --stdout
```

## NECESSARY DATA
Before you train you need spacy formatted data. In order to run the model you also need data in a json format where you **specify the entities and relationships**. This is since the model is doing classification and needs to know what classifiers to predict. See [here](https://github.com/explosion/spaCy/discussions/9567) for more information.

# Project Structure Explanation
## project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

###  Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `data` | Parse the gold-standard annotations from the Prodigy annotations. |
| `train_cpu` | Train the REL model on the CPU and evaluate on the dev corpus. |
| `train_gpu` | Train the REL model with a Transformer on a GPU and evaluate on the dev corpus. |
| `evaluate` | Apply the best model to new, unseen text, and measure accuracy at different thresholds. |
| `clean` | Remove intermediate files to start data preparation and training from a clean slate. |

To switch to using an existing transformer model you should go into `configs/rel_trf.cfg` and change any variables for your appropriate transformer model. Then change any references in the top level `project.yml` file to use `rel_trf.cfg` instead of `rel_tok2vec.cfg`. This will make it so the project is run with the transformer config.

### Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `data` &rarr; `train_cpu` &rarr; `evaluate` |
| `all_gpu` | `data` &rarr; `train_gpu` &rarr; `evaluate` |

### Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/annotations.jsonl`](assets/annotations.jsonl) | Local | Gold-standard REL annotations created with Prodigy |

#### To create more assets
First read `https://spacy.io/api/data-formats#json-input` 
Then run `python -m spacy convert .\nameOfYourData.json .;` to get your json file into the spacy format for training.

See `exampleData/` for an example of data that can be converted. This data was gotten off Github. 