<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Healthsea

This is the spaCy project for the Healthsea pipeline. It holds all training data and workflows for training a Named Entity Recognition model and a custom Text Classification model (Clausecat). You can read more in this [blog post](explosion.ai)

## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `requirements` | Install dependencies and requirements |
| `parse_ner` | Load the annotations to Prodigy and use data-to-spacy to split the data into a training and development set |
| `analyze_ner` | Analyze the NER annotation dataset |
| `parse_clausecat` | Parse the textcat annotations into clausecat format and split into training and development set |
| `train_ner` | Train an NER model |
| `evaluate_ner` | Evaluate the trained NER model |
| `train_clausecat` | Train the custom Clausecat component |
| `evaluate_clausecat` | Evaluate a custom Clausecat component |
| `evaluate` | Evaluate the healthsea pipeline |
| `reset` | Reset the project to its original state and delete all training process |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `install` | `requirements` |
| `process_ner` | `parse_ner` &rarr; `analyze_ner` &rarr; `train_ner` &rarr; `evaluate_ner` |
| `process_clausecat` | `parse_clausecat` &rarr; `train_clausecat` &rarr; `evaluate_clausecat` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/ner/annotation.jsonl`](assets/ner/annotation.jsonl) | Local | NER annotations exported from Prodigy with 5000 examples and 2 labels |
| [`assets/ner/train.spacy`](assets/ner/train.spacy) | Local | Training dataset for NER |
| [`assets/ner/dev.spacy`](assets/ner/dev.spacy) | Local | Development dataset for NER |
| [`assets/clausecat/annotation.jsonl`](assets/clausecat/annotation.jsonl) | Local | Annotations exported from Prodigy with 5000 examples and 4 classes (textcat.recipe) |
| [`assets/clausecat/train.spacy`](assets/clausecat/train.spacy) | Local | Training dataset for Clausecat (Text Classification) |
| [`assets/clausecat/dev.spacy`](assets/clausecat/dev.spacy) | Local | Development dataset for Clausecat (Text Classification) |
| [`assets/end_to_end_evaluation.json`](assets/end_to_end_evaluation.json) | Local | End-to-end evaluation dataset |
| [`assets/pretrained_weights_ner.bin`](assets/pretrained_weights_ner.bin) | Local | Pretrained weights trained on IHerb reviews for initializing NER tok2vec |
| [`assets/pretrained_weights_clausecat.bin`](assets/pretrained_weights_clausecat.bin) | Local | Pretrained weights trained on IHerb reviews for initializing Clausecat tok2vec |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

### 💾 Data
The data for training and evaluation is provided in the assets folder. The ```parse_ner``` command requires [prodigy](https://prodi.gy/) but isn't necessary for training since the data is already present.

### 🤖 Training with GPU

To train with the gpu it's required to have ```cuda``` installed that is compatible with your current ```torch``` version. More information about installation can be found in the [spaCy installation docs](https://spacy.io/usage#quickstart)

You can change the ```gpu_id``` variable inside the ```project.yml``` to your current gpu device (normally 0, cpu is -1).