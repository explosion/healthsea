import pytest
from spacy.lang.en import English
from ..clausecat_reader import Clausecat_corpus
from spacy.training.example import Example

# fmt: off
@pytest.mark.parametrize(
    "path",
    [
        ("assets/textcat/dev.spacy"),
        ("assets/textcat/train.spacy"),
    ],
)
# fmt: on
def test_reader(path):
    reader = Clausecat_corpus(path)
    nlp = English()
    examples = list(reader(nlp))

    wrong = set([type(eg) for eg in examples if not isinstance(eg, Example)])

    assert not wrong
    assert examples[0].reference.has_extension("clauses")
    assert type(examples[0].reference._.clauses[0]["blinder"]) == str

    for example in examples[:6]:
        print(example.predicted.text, len(example.predicted))
        print(f">> {example.predicted._.clauses} \n")
