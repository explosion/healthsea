from spacy.tokens import Doc, Span
from thinc.api import Model
from thinc.api import chain
from thinc.types import Floats2d, Ragged
from typing import List, Callable, Tuple, Optional
from spacy import registry


@registry.architectures("spacy.clausecat_model.v1")
def build_text_classifier_lowdata(
    blinder: Model[List[Doc], List[Doc]], textcat: Model[List[Doc], Floats2d]
) -> Model[List[Doc], Floats2d]:

    with Model.define_operators({">>": chain}):
        model = blinder >> textcat
    return model


@registry.layers("spacy.blinder.v1")
def build_blinder() -> Model[List[Doc], List[Doc]]:

    return Model(
        "blinder",
        forward=forward,
    )


def forward(
    model: Model[List[Doc], List[Doc]], docs: List[Doc], is_train: bool
) -> List[Doc]:

    clauses = []
    for doc in docs:
        for clause in doc._.clauses:
            words = []
            clause_slice = doc[clause["split_indices"][0] : clause["split_indices"][1]]

            if clause["has_ent"]:
                for i, token in enumerate(clause_slice):
                    if i + 1 == clause["ent_indices"][0]:
                        words.append(clause["blinder"])
                    elif i + 1 not in range(
                        clause["ent_indices"][0], clause["ent_indices"][1]
                    ):
                        words.append(token.text)
                clauses.append(Doc(doc.vocab, words=words))

            else:
                for token in clause_slice:
                    words.append(token.text)
                clauses.append(Doc(doc.vocab, words=words))

    def backprop():
        return

    return clauses, backprop
