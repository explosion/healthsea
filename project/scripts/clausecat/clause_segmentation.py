import spacy, benepar
from spacy.tokens import Doc
from typing import Tuple, List
from spacy.language import Language
import warnings

# fmt: off
warning_text =  "<class 'torch_struct.distributions.TreeCRF'> does not define `arg_constraints`. " + 'Please set `arg_constraints = {}` or initialize the distribution ' + 'with `validate_args=False` to turn off validation.'
warnings.filterwarnings("ignore", warning_text)
# fmt: on


@Language.factory("healthsea.segmentation.v1")
def make_segmentation(nlp: Language, name: str):
    return ClauseSegmentation(nlp, name)


class ClauseSegmentation:
    """Use the benepar tree to split sentences and blind entities."""

    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc):
        """Extract clauses and save the split indices in a custom attribute"""
        clauses = []
        split_indices = self.benepar_split(doc)

        for index_pair in split_indices:
            current_span = doc[index_pair[0] : index_pair[1]]
            if len(current_span.ents) != 0:
                for entity in current_span.ents:
                    clauses.append(
                        {
                            "split_indices": index_pair,
                            "has_ent": True,
                            "ent_indices": (entity.start, entity.end),
                            "blinder": f"_{entity.label_}_",
                            "ent_name": entity.text,
                            "cats": {},
                        }
                    )
            else:
                clauses.append(
                    {
                        "split_indices": index_pair,
                        "has_ent": False,
                        "ent_indices": (0, 0),
                        "blinder": None,
                        "ent_name": None,
                        "cats": {},
                    }
                )

        # Check if ._.clauses exists and only overwrite when len() == 0
        if not doc.has_extension("clauses") or (
            doc.has_extension("clauses") and len(doc._.clauses) == 0
        ):
            doc.set_extension("clauses", default={}, force=True)
            doc._.clauses = clauses

        return doc

    def benepar_split(self, doc: Doc) -> List[Tuple]:
        """Split a doc into individual clauses
        doc (Doc): Input doc containing one or more sentences
        RETURNS (List[Tuple]): List of extracted clauses, defined by their start-end offsets
        """
        split_indices = []
        for sentence in doc.sents:
            can_split = False
            for constituent in sentence._.constituents:
                # Include every clause with the label "S" (Sentence) and with the root sentence as parent
                if "S" in constituent._.labels and constituent._.parent == sentence:
                    split_indices.append((constituent.start, constituent.end))
                    can_split = True

            # If no clause found just append the whole sentence
            if not can_split:
                split_indices.append((sentence.start, sentence.end))

        return split_indices
