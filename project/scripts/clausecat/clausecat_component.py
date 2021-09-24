from itertools import islice
from typing import Iterable, Tuple, Optional, Dict, List, Callable, Any
from thinc.api import get_array_module, Model, Optimizer, set_dropout_rate, Config
from thinc.types import Floats2d
from spacy.training.example import Example
import numpy

from spacy.pipeline import TrainablePipe
from spacy.language import Language
from spacy.training import Example
from spacy.errors import Errors
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.vocab import Vocab

single_label_default_config = """
[model]
@architectures = "spacy.TextCatEnsemble.v2"

[model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 64
rows = [2000, 2000, 1000, 1000, 1000, 1000]
attrs = ["ORTH", "LOWER", "PREFIX", "SUFFIX", "SHAPE", "ID"]
include_static_vectors = false

[model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = ${model.tok2vec.embed.width}
window_size = 1
maxout_pieces = 3
depth = 2

[model.linear_model]
@architectures = "spacy.TextCatBOW.v2"
exclusive_classes = true
ngram_size = 1
no_output_layer = false
"""
DEFAULT_SINGLE_TEXTCAT_MODEL = Config().from_str(single_label_default_config)["model"]


@Language.factory(
    "clausecat.v1",
    assigns=["doc.cats"],
    requires=["doc._.clauses"],
    default_config={"threshold": 0.5, "model": DEFAULT_SINGLE_TEXTCAT_MODEL},
    default_score_weights={
        "cats_score": 1.0,
        "cats_score_desc": None,
        "cats_micro_p": None,
        "cats_micro_r": None,
        "cats_micro_f": None,
        "cats_macro_p": None,
        "cats_macro_r": None,
        "cats_macro_f": None,
        "cats_macro_auc": None,
        "cats_f_per_type": None,
        "cats_macro_auc_per_type": None,
    },
)
def make_clausecat(
    nlp: Language, name: str, model: Model[List[Doc], List[Floats2d]], threshold: float
) -> "Clausecat":

    return Clausecat(nlp.vocab, model, name, threshold=threshold)


class Clausecat(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "clausecat",
        *,
        threshold: float,
    ) -> None:
        self.vocab = vocab
        self.model = model
        self.name = name
        cfg = {"labels": [], "threshold": threshold, "positive_label": None}
        self.cfg = dict(cfg)

    @property
    def labels(self) -> Tuple[str]:
        return tuple(self.cfg["labels"])

    @property
    def label_data(self) -> List[str]:
        return self.labels

    def predict(self, clause_containers: Iterable[Doc]):
        """Apply the pipeline's model to a batch of docs, without modifying them.

        docs (Iterable[Doc]): The documents to predict.
        RETURNS: The models prediction for each document.

        DOCS: https://spacy.io/api/textcategorizer#predict
        """
        docs = []
        for clause_container in clause_containers:
            for clause in clause_container._.clauses:
                docs.append(clause[0])

        if not any(len(doc) for doc in docs):
            # Handle cases where there are no tokens in any docs.
            tensors = [doc.tensor for doc in docs]
            xp = get_array_module(tensors)
            scores = xp.zeros((len(docs), len(self.labels)))
            return scores
        scores = self.model.predict(docs)
        scores = self.model.ops.asarray(scores)
        return scores

    def set_annotations(self, clause_containers: Iterable[Doc], scores) -> None:
        """Modify a batch of Doc objects, using pre-computed scores.

        docs (Iterable[Doc]): The documents to modify.
        scores: The scores to set, produced by TextCategorizer.predict.

        DOCS: https://spacy.io/api/textcategorizer#set_annotations
        """
        docs = []
        for clause_container in clause_containers:
            for clause in clause_container._.clauses:
                docs.append(clause[0])

        for i, doc in enumerate(docs):
            for j, label in enumerate(self.labels):
                doc.cats[label] = float(scores[i, j])

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
        labels: Optional[Iterable[str]] = None,
        positive_label: Optional[str] = None,
    ) -> None:

        if labels is None:
            for example in get_examples():
                for clause in example.y._.clauses:
                    for cat in clause[0].cats:
                        self.add_label(cat)
        else:
            for label in labels:
                self.add_label(label)

        if len(self.labels) < 2:
            raise ValueError(Errors.E867)
        if positive_label is not None:
            if positive_label not in self.labels:
                err = Errors.E920.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
            if len(self.labels) != 2:
                err = Errors.E919.format(pos_label=positive_label, labels=self.labels)
                raise ValueError(err)
        self.cfg["positive_label"] = positive_label

        subbatch = list(islice(get_examples(), 10))
        doc_sample = []
        for doc in subbatch:
            for clause in doc.y._.clauses:
                doc_sample.append(clause[0])

        label_sample, _ = self._examples_to_truth(doc_sample)
        # self._require_labels()
        assert len(doc_sample) > 0, Errors.E923.format(name=self.name)
        assert len(label_sample) > 0, Errors.E923.format(name=self.name)
        self.model.initialize(X=doc_sample, Y=label_sample)

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe.

        label (str): The label to add.
        RETURNS (int): 0 if label is already present, otherwise 1.

        DOCS: https://spacy.io/api/textcategorizer#add_label
        """
        if not isinstance(label, str):
            raise ValueError(Errors.E187)
        if label in self.labels:
            return 0
        self.cfg["labels"].append(label)
        if self.model and "resize_output" in self.model.attrs:
            self.model = self.model.attrs["resize_output"](
                self.model, len(self.cfg["labels"])
            )
        self.vocab.strings.add(label)
        return 1

    def _examples_to_truth(
        self, examples: List[Example]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        truths = numpy.zeros((len(examples), len(self.labels)), dtype="f")
        not_missing = numpy.ones((len(examples), len(self.labels)), dtype="f")
        for i, eg in enumerate(examples):
            for j, label in enumerate(self.labels):
                if label in eg.cats:
                    truths[i, j] = eg.cats[label]
                else:
                    not_missing[i, j] = 0.0
        truths = self.model.ops.asarray(truths)
        return truths, not_missing

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss.

        examples (Iterable[Example]): A batch of Example objects.
        drop (float): The dropout rate.
        sgd (thinc.api.Optimizer): The optimizer.
        losses (Dict[str, float]): Optional record of the loss during training.
            Updated using the component name as the key.
        RETURNS (Dict[str, float]): The updated losses dictionary.

        DOCS: https://spacy.io/api/textcategorizer#update
        """
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        predicted_examples = []
        gold_examples = []
        for example in examples:
            for clause in example.predicted._.clauses:
                predicted_examples.append(clause[0])
            for clause in example.reference._.clauses:
                gold_examples.append(clause[0])

        scores, bp_scores = self.model.begin_update(predicted_examples)
        loss, d_scores = self.get_loss(gold_examples, scores)
        bp_scores(d_scores)
        if sgd is not None:
            self.finish_update(sgd)
        losses[self.name] += loss
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores.

        examples (Iterable[Examples]): The batch of examples.
        scores: Scores representing the model's predictions.
        RETURNS (Tuple[float, float]): The loss and the gradient.

        DOCS: https://spacy.io/api/textcategorizer#get_loss
        """

        truths, not_missing = self._examples_to_truth(examples)
        not_missing = self.model.ops.asarray(not_missing)
        d_scores = (scores - truths) / scores.shape[0]
        d_scores *= not_missing
        mean_square_error = (d_scores ** 2).sum(axis=1).mean()
        return float(mean_square_error), d_scores

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples.

        examples (Iterable[Example]): The examples to score.
        RETURNS (Dict[str, Any]): The scores, produced by Scorer.score_cats.

        DOCS: https://spacy.io/api/textcategorizer#score
        """

        examples_clauses = []
        for example in examples:
            prediction = example.predicted
            reference = example.reference
            for clause_pred, clause_ref in zip(
                prediction._.clauses, reference._.clauses
            ):
                examples_clauses.append(Example(clause_pred, clause_ref))

        kwargs.setdefault("threshold", self.cfg["threshold"])
        kwargs.setdefault("positive_label", self.cfg["positive_label"])
        return Scorer.score_cats(
            examples_clauses,
            "cats",
            labels=self.labels,
            multi_label=False,
            **kwargs,
        )
