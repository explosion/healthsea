import pytest
import spacy
from ..custom_code import *
from wasabi import Printer

msg = Printer()

# fmt: off
@pytest.mark.parametrize(
    "model, text",
    [
        ("training/clausecat/config_tok2vec/model-best/","This helped my joint pain"),
        ("training/clausecat/config_tok2vec/model-best/","This helped my joint pain but this has also caused rashes"),
        ("training/clausecat/config_tok2vec/model-best/","This helped my joint pain but not hip pain"),
    ],
)
# fmt: on
def test_segmentation(model, text):
    nlp = spacy.load(model)
    doc = nlp(text)

    print_text = f"""
    {text}
        >> {doc.ents}
        >> {doc._.clauses}
        >> {doc._.effects}
        \n
    """

    print(print_text)
