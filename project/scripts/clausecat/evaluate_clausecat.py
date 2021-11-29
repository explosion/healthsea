import spacy
from spacy.scorer import PRFScore
import typer
from pathlib import Path
from wasabi import Printer, table
import operator
import benepar

import clausecat_component
import clausecat_model
import clausecat_reader
import clause_segmentation
import clause_aggregation

msg = Printer()


def main(model_path: Path, eval_path: Path):
    """This script is used to evaluate the clausecat component"""

    nlp = spacy.load(model_path)
    reader = clausecat_reader.ClausecatCorpus(eval_path)
    examples = reader(nlp)

    clausecat = nlp.get_pipe("clausecat")

    scorer = {
        "POSITIVE": PRFScore(),
        "NEGATIVE": PRFScore(),
        "NEUTRAL": PRFScore(),
        "ANAMNESIS": PRFScore(),
    }

    for i, example in enumerate(examples):
        prediction = example.predicted
        reference = example.reference

        # Prediction
        prediction = clausecat(prediction)

        # Iterate through prediction and references
        for pred_clause, ref_clause in zip(prediction._.clauses, reference._.clauses):
            prediction_cats = pred_clause["cats"]
            reference_cats = ref_clause["cats"]
            prediction_class = max(prediction_cats.items(), key=operator.itemgetter(1))[
                0
            ]

            # Add to matrix
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

    # Printing
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
