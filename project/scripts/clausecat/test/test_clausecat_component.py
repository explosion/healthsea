import pytest
from spacy.training import initialize
from spacy.lang.en import English
from ..clausecat_component import *
from ..clausecat_reader import Clausecat_corpus
from ..clausecat_model import *


# fmt: off
@pytest.mark.parametrize(
    "train_path, dev_path",
    [
        ("assets/textcat/train.spacy","../../assets/textcat/dev.spacy"),
    ],
)
# fmt: on
def test_reader(train_path, dev_path):
    nlp = English()

    train_reader = Clausecat_corpus(train_path)
    dev_reader = Clausecat_corpus(dev_path)

    clausecat = nlp.add_pipe("healthsea.clausecat")

    # Testing initialize(), add_label(), _examples_to_truth()
    clausecat.initialize(lambda: islice(train_reader(nlp), 100))
