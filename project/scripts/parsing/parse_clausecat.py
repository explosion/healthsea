from typing import Union
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from spacy.lang.en import English
from spacy.training.example import Example
from spacy.training.corpus import walk_corpus
from typing import Union, Iterable, Iterator
from pathlib import Path
import copy
import typer
from wasabi import Printer

msg = Printer()


def main(dev_path, output_path):
    Doc.set_extension("clauses", default=[])
    nlp = English()
    docs = read_docbin(nlp.vocab, walk_corpus(dev_path, ".spacy"))

    for doc in docs:
        has_ent = False
        entity_index = 0
        for index, token in enumerate(doc):
            # Because our annotations already have blinded entities we're looking for '<', '>' inside a token (<CONDITION>, <BENEFIT>)
            if ">" in token.text and "<" in token.text:
                entity_index = index
                has_ent = True
                break

        # ._.clause format
        ## split_indices: Tuple[int,int], has_ent: bool, ent_indices: Tuple[int,int], blinder: str, ent_name: str, cats: dict[str,float]
        ## Note that future entity indices have to be reduced by the clause indices, (e.g index 0 of an entity is the first token of the doc slice and not the whole doc)
        clauses = [
            {
                "split_indices": (0, len(doc) - 1),
                "has_ent": has_ent,
                "ent_indices": (entity_index, entity_index),
                "blinder": doc[entity_index].text,
                "ent_name": "Entity",
                "cats": doc.cats,
            }
        ]
        doc._.clauses = clauses

    docbin = DocBin(docs=docs, store_user_data=True)
    docbin.to_disk(output_path)
    msg.good(f"Parsing complete")


def read_docbin(vocab: Vocab, locs: Iterable[Union[str, Path]]) -> Iterator[Doc]:
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


if __name__ == "__main__":
    typer.run(main)
