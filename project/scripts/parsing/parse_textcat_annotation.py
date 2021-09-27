from spacy.tokens import DocBin
from spacy.lang.en import English

from wasabi import Printer
from wasabi import table

import json
from pathlib import Path
import typer

msg = Printer()


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
):
    """Parse the textcat annotations into a training and development set."""

    examples = []
    label_dict = {}
    nlp = English()

    # Load dataset
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)

            labels = example["accept"]
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = []

            examples.append(example)

    # Group information
    for example in examples:
        if example["answer"] == "accept":

            doc = nlp(example["text"])
            labels = example["accept"]

            cats_dict = {}
            for label in label_dict.keys():
                cats_dict[label] = 0.0

            for label in labels:
                cats_dict[label] = 1.0

            doc.cats = cats_dict
            label_dict[label].append(doc)

    # Split
    train = []
    dev = []
    table_data = []
    checksum = 0

    for label in label_dict:
        split = int(len(label_dict[label]) * eval_split)
        train += label_dict[label][split:]
        dev += label_dict[label][:split]
        checksum += len(label_dict[label])
        table_data.append(
            (
                label,
                len(label_dict[label]),
                len(label_dict[label][split:]),
                len(label_dict[label][:split]),
            )
        )

    msg.divider("TEXTCAT dataset summary")
    msg.info(f"Evaluation split: {eval_split}")
    table_data.append(("All", checksum, len(train), len(dev)))
    header = ("Label", "Total", "Training", "Development")
    print(table(table_data, header=header, divider=True))

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)
    msg.good(f"Parsing complete")


if __name__ == "__main__":
    typer.run(main)
