import spacy
import typer
from pathlib import Path
from wasabi import Printer, table
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from typing import Union, Iterable, Iterator
from spacy.scorer import PRFScore

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

    for example in examples:
        prediction = example.predicted
        reference = example.reference

        prediction = clausecat(prediction)

        for pred_clause, ref_clause in zip(prediction._.clauses, reference._.clauses):
            prediction_cats = pred_clause["cats"]
            reference_cats = ref_clause["cats"]


if __name__ == "__main__":
    typer.run(main)
