import spacy
import typer
from pathlib import Path
from wasabi import Printer, table
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from typing import Union, Iterable, Iterator
from spacy.scorer import PRFScore
import operator

msg = Printer()

from clausecat import clausecat_component, clausecat_model, clausecat_reader

# To do


def main(model_path: Path, eval_path: Path):
    nlp = spacy.load(model_path)
    reader = clausecat_reader.Clausecat_corpus(eval_path)
    examples = reader(nlp)

    clausecat = nlp.get_pipe("clausecat")
    threshold = clausecat.threshold

    scorer = {
        "POSITIVE": PRFScore(),
        "NEGATIVE": PRFScore(),
        "NEUTRAL": PRFScore(),
        "ANAMNESIS": PRFScore(),
    }

    for i, example in enumerate(examples):
        prediction = example.predicted
        reference = example.reference

        try:
            prediction = clausecat(prediction)
        except Exception as error:
            print(f"ERROR at {i} {error}")

        for pred_clause, ref_clause in zip(prediction._.clauses, reference._.clauses):
            prediction_cats = pred_clause["cats"]
            reference_cats = ref_clause["cats"]
            prediction_class = max(prediction_cats.items(), key=operator.itemgetter(1))[
                0
            ]

            for label in prediction_cats:
                if label != prediction_class:
                    prediction = 0
                else:
                    prediction = 1

                if prediction == 0 and reference_cats[label] != 0:
                    scorer[label].fn += 1

                elif prediction == 1 and reference_cats[label] != 1:
                    scorer[label].fp += 1

                elif prediction == 1 and reference_cats[label] == 1:
                    scorer[label].tp += 1

    textcat_data = []
    avg_fscore = 0
    avg_recall = 0
    avg_precision = 0

    for label in scorer:
        textcat_data.append(
            (
                label,
                round(scorer[label].fscore, 2),
                round(scorer[label].recall, 2),
                round(scorer[label].precision, 2),
            )
        )
        avg_fscore += scorer[label].fscore
        avg_recall += scorer[label].recall
        avg_precision += scorer[label].precision

    textcat_data.append(
        (
            "AVERAGE",
            round(avg_fscore / len(scorer), 2),
            round(avg_recall / len(scorer), 2),
            round(avg_precision / len(scorer), 2),
        )
    )

    header = ("Label", "F-Score", "Recall", "Precision")

    print(table(textcat_data, header=header, divider=True))


if __name__ == "__main__":
    typer.run(main)
