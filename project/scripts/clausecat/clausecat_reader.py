from typing import Union
from spacy import util
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from spacy.training.example import Example
from spacy.training.corpus import walk_corpus
from typing import Union, Iterable, Iterator
from pathlib import Path
import copy


@util.registry.readers("healthsea.clausecat_reader.v1")
def create_docbin_reader(path):
    return ClausecatCorpus(path)


class ClausecatCorpus:
    def __init__(self, path: Union[str, Path]):
        self.path = util.ensure_path(path)
        if not Doc.has_extension("clauses"):
            Doc.set_extension("clauses", default=[])

    def __call__(self, nlp) -> Iterator[Example]:
        docs = self.read_docbin(nlp.vocab, walk_corpus(self.path, ".spacy"))
        for doc in docs:

            reference_doc = Doc(
                nlp.vocab,
                words=[word.text for word in doc],
                spaces=[bool(word.whitespace_) for word in doc],
            )

            prediction_doc = Doc(
                nlp.vocab,
                words=[word.text for word in doc],
                spaces=[bool(word.whitespace_) for word in doc],
            )

            has_ent = False
            start_entity_index = 0
            end_entity_index = 0
            for token in doc:
                # Because our annotations already have blinded entities we're looking for '<', '>' inside a token (<CONDITION>, <BENEFIT>)
                if "<" == token.text:
                    for tokenx2 in doc[token.i :]:
                        if tokenx2.text == ">":
                            start_entity_index = token.i
                            end_entity_index = tokenx2.i + 1
                            has_ent = True
                            break
                    break

            # Changing prefix & suffix of the blinder to make sure tokenizer tokenizes the blinder as one token
            blinder = str(doc[start_entity_index:end_entity_index].text)
            if "<CONDITION>" in blinder:
                blinder = "_CONDITION_"
            elif "<BENEFIT>" in blinder:
                blinder = "_BENEFIT_"
            # ._.clause format
            ## split_indices: Tuple[int,int], has_ent: bool, ent_indices: Tuple[int,int], blinder: str, ent_name: str, cats: dict[str,float]
            clauses = [
                {
                    "split_indices": (0, len(doc)),
                    "has_ent": has_ent,
                    "ent_indices": (start_entity_index, end_entity_index),
                    "blinder": blinder,
                    "ent_name": "Entity",
                    "cats": doc.cats,
                }
            ]
            reference_doc._.clauses = clauses
            prediction_doc._.clauses = copy.deepcopy(clauses)

            yield Example(prediction_doc, reference_doc)

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
