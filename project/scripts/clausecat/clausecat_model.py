from spacy.tokens import Doc, Span
from thinc.api import Model
from thinc.types import Floats2d, Ragged
from typing import List, Callable, Tuple, Optional
from spacy import registry


@registry.architectures("spacy.clausecat.v1")
def clausecat(
    textcat: Model[List[Doc], Floats2d], get_clauses: Callable[[Doc], List[Span]]
) -> Model[List[Doc], List[Ragged]]:
    """Wrap a text classification model so that it is applied over individual
    clauses, rather than, whole documents.
    """
    return Model(
        "clausecat",
        forward,
        init=init,
        attrs={"get_clauses": get_clauses},
        layers=[textcat],
        refs={"textcat": textcat},
    )


def forward(
    model: Model[List[Doc], Ragged], docs: List[Doc], is_train: bool
) -> Tuple[Ragged, Callable]:
    get_clauses = model.attrs["get_clauses"]
    textcat = model.get_ref("textcat")
    clauses = []
    lengths = []

    for doc in docs:
        doc_clauses = get_clauses(doc)
        clauses.extend([span[0] for span in doc_clauses])
        lengths.append(len(doc_clauses))

    clause_scores, backprop_textcat = textcat(clauses, is_train=is_train)
    output = Ragged(clause_scores, textcat.ops.asarray1i(lengths))

    def backprop(d_output: Ragged) -> List[Doc]:
        # We don't have to get a gradient with respect to the docs themselves,
        # we just need to backprop through the textcat layer.
        _ = backprop_textcat(d_output)
        return docs

    return output, backprop


def init(
    model: Model[List[Doc], Ragged],
    X: Optional[List[Doc]] = None,
    Y: Optional[Ragged] = None,
):
    textcat = model.get_ref("textcat")
    if X is None and Y is None:
        textcat.initialize()
    elif Y is None:
        textcat.initialize(X=X)
    elif len(X) == 0:
        textcat.initialize(X=X, Y=Y)
    else:
        textcat.initialize(X=X[:1], Y=Y)
    return
