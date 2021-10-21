from spacy.tokens import Doc
from thinc.api import Model
from thinc.api import chain
from thinc.types import Floats2d
from typing import List
from spacy import registry


@registry.architectures("healthsea.clausecat_model")
def build_clausecat(
    blinder: Model[List[Doc], List[Doc]], textcat: Model[List[Doc], Floats2d]
) -> Model[List[Doc], Floats2d]:
    """
    Build a wrapper model that chains a blinder layer to a textcat model
    """
    with Model.define_operators({">>": chain}):
        model = blinder >> textcat
    return model


@registry.layers("healthsea.blinder")
def build_blinder() -> Model[List[Doc], List[Doc]]:
    """
    Build a blinder layer that uses the custom attribute ._.clauses to split docs into clauses and blinds entities
    """
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

    def backprop(dY):
        return

    return clauses, backprop
