from typing import Union
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from spacy.training.example import Example
from spacy.training.corpus import walk_corpus
from typing import Union, Iterable, Iterator
from pathlib import Path
import typer
import copy


from scripts.clausecat import clausecat_component


@util.registry.readers("clausecat.reader.v1")
def create_docbin_reader(path):
    return Clausecat_corpus(path)


class Clausecat_corpus:
    def __init__(self, path: Union[str, Path]):
        self.path = util.ensure_path(path)
        if not Doc.has_extension("clauses"):
            Doc.set_extension("clauses", default=[])

    def __call__(self, nlp) -> Iterator[Example]:

        pred_docs = self.read_docbin(nlp.vocab, walk_corpus(self.path, ".spacy"))
        for reference in pred_docs:

            pred_doc = Doc(
                nlp.vocab,
                words=[t.text for t in reference],
                spaces=[t.whitespace_ for t in reference],
            )
            pred_clause = Doc(
                nlp.vocab,
                words=[t.text for t in reference],
                spaces=[t.whitespace_ for t in reference],
            )
            pred_clause.cats = copy.deepcopy(reference.cats)

            gold_doc = Doc(
                nlp.vocab,
                words=[t.text for t in reference],
                spaces=[t.whitespace_ for t in reference],
            )

            pred_doc._.clauses = [(pred_clause, None, None)]
            gold_doc._.clauses = [(reference, None, None)]

            yield Example(pred_doc, gold_doc)

    def read_docbin(
        self, vocab: Vocab, locs: Iterable[Union[str, Path]]
    ) -> Iterator[Doc]:
        """Yield training examples as example dicts"""
        i = 0
        for loc in locs:
            loc = util.ensure_path(loc)
            if loc.parts[-1].endswith(".spacy"):
                doc_bin = DocBin().from_disk(loc)
                docs = doc_bin.get_docs(vocab)
                for doc in docs:
                    if len(doc):
                        yield doc
                        i += 1


def test(path: Path):
    from spacy.lang.en import English

    reader = Clausecat_corpus(path)
    nlp = English()
    reader(nlp)


if __name__ == "__main__":
    typer.run(test)
