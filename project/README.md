<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: Healthsea

This project trains a Named Entity Recognition (NER) model, a Text Classification model, and assembles them together with custom components into a finished end-to-end pipeline.

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
| `parse_textcat` | Parse the textcat annotation file manually into a training and development set |
| `train_ner` | Train an NER model |
| `evaluate_ner` | Evaluate a trained NER model |
| `train_textcat` | Train a Text Classification model |
| `evaluate_textcat` | Evaluate a trained Text Classification model |
| `assemble_healthsea` | Assemble trained components into the healthsea pipeline |
| `evaluate_healthsea` | Evaluate the finished healthsea pipeline |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `install` | `requirements` |
| `parse` | `parse_ner` &rarr; `parse_textcat` |
| `train` | `train_ner` &rarr; `train_textcat` |
| `evaluate` | `evaluate_ner` &rarr; `evaluate_textcat` |
| `assemble` | `assemble_healthsea` &rarr; `evaluate_healthsea` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| [`assets/ner/ner_annotation.jsonl`](assets/ner/ner_annotation.jsonl) | Local | NER annotations exported from Prodigy with 5000 examples and 2 labels |
| [`assets/ner/train.spacy`](assets/ner/train.spacy) | Local | Training dataset for NER |
| [`assets/ner/dev.spacy`](assets/ner/dev.spacy) | Local | Development dataset for NER |
| [`assets/textcat/textcat_annotation.jsonl`](assets/textcat/textcat_annotation.jsonl) | Local | Text Classification annotations exported from Prodigy with 5000 examples and 4 classes |
| [`assets/textcat/train.spacy`](assets/textcat/train.spacy) | Local | Training dataset for Text Classification |
| [`assets/textcat/dev.spacy`](assets/textcat/dev.spacy) | Local | Development dataset for Text Classification |
| [`assets/end_to_end_evaluation.json`](assets/end_to_end_evaluation.json) | Local | Examples for end-to-end evaluation of the pipeline |
| [`assets/pretrained_weights.bin`](assets/pretrained_weights.bin) | Local | Pretrained weights trained with IHerb reviews for initializing tok2vec components |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->

### 💾 Data
The data for training and evaluation (25% split) is provided in the assets folder. The ```parse_ner``` command requires [prodigy](https://prodi.gy/) to work but it's not necessary for training.

### 🤖 Training with GPU

To train with the gpu it's required to have ```cuda``` installed which is compatible with your current ```torch``` version. More information about installation can be found in the [spaCy installation docs](https://spacy.io/usage#quickstart) and in the [requirements file](https://github.com/thomashacker/healthsea/blob/main/project/requirements.txt).

You can change the ```gpu_id``` variable inside the ```project.yml``` to your current gpu device (normally 0, no gpu is -1).

### ⚙️Tok2Vec & Transformer

Two configs are provided for every trainable pipeline, one which uses ```tok2vec``` as embedding layer and one which uses ```transformers```. To change the config, you can change the ```config``` variable to either ```config_tok2vec``` or ```config_trf```