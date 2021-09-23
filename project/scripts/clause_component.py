from spacy.tokens import Doc
from spacy.language import Language
from typing import Tuple, List

# Fix import
import scripts.clause_segmentation

# import clause_segmentation

import operator


@Language.factory("clause_component")
def create_clause_component(nlp: Language, name: str):
    return Clausecomponent(nlp)


class Clausecomponent:
    def __init__(self, nlp: Language):

        if nlp.has_pipe("transformer_textcat"):
            self.embedding = nlp.get_pipe("transformer_textcat")

        elif nlp.has_pipe("tok2vec_textcat"):
            self.embedding = nlp.get_pipe("tok2vec_textcat")

        self.textcat = nlp.get_pipe("textcat")

    def __call__(self, doc: Doc):

        # Clause Segmentation
        classified_clauses = self.apply_clause_segmentation(
            doc, self.textcat, self.embedding
        )

        # Summarize health effects
        health_effects = self.summarize_health_effects(classified_clauses)

        # Merge multiple classifications to one
        self.merge_health_effects(doc, health_effects)

        return doc

    def apply_clause_segmentation(self, doc: Doc, textcat, embedding) -> List[Tuple]:
        """Apply clause segmentation on a doc"""

        clauses = clause_segmentation.clause_segmentation(doc)
        classified_clauses = []
        for clause in clauses:
            classified_doc = textcat(embedding(clause[0]))
            classification = classified_doc.cats

            classified_clauses.append((clause[0], clause[1], classification, clause[2]))

        doc.set_extension("clauses", default=[], force=True)
        doc._.clauses = classified_clauses

        return classified_clauses

    def summarize_health_effects(self, classified_clauses: List[Tuple]) -> List[Tuple]:
        """Collect classifications per entity"""

        # Patient Information
        patient_information = []
        health_effects = {}

        for clause in classified_clauses:
            classification = max(clause[2].items(), key=operator.itemgetter(1))[0]
            entity = clause[1]

            # Patient Information
            if classification == "ANAMNESIS" and entity != None:
                patient_information.append((entity, []))
            else:
                for patient_health in patient_information:
                    patient_health[1].append(classification)

            # Health effects
            if entity != None:
                if entity not in health_effects:
                    health_effects[entity] = {
                        "classification": [],
                        "label": clause[3],
                    }
                health_effects[entity]["classification"].append(classification)

        # Apply Patient Information on health effects
        for patient_health in patient_information:
            entity = patient_health[0]
            score = 0
            for classification in patient_health[1]:
                if classification == "POSITIVE":
                    score += 1
                elif classification == "NEGATIVE":
                    score -= 1

            if score > 0:
                end_classification = "POSITIVE"
            elif score < 0:
                end_classification = "NEGATIVE"
            else:
                end_classification = "NEUTRAL"

            health_effects[entity]["classification"].append(end_classification)

        return health_effects

    def merge_health_effects(self, doc: Doc, health_effects: List[Tuple]):
        """Merge multiple classifcations per entity into one"""

        unique_health_effects = {}

        for entity in health_effects:
            score = 0
            for classification in health_effects[entity]["classification"]:
                if classification == "POSITIVE":
                    score += 1
                elif classification == "NEGATIVE":
                    score -= 1

            if score > 0:
                end_classification = "POSITIVE"
            elif score < 0:
                end_classification = "NEGATIVE"
            else:
                end_classification = "NEUTRAL"

            unique_health_effects[entity] = {}
            unique_health_effects[entity]["classification"] = end_classification
            unique_health_effects[entity]["label"] = health_effects[entity]["label"]

        doc.set_extension("health_effects", default={}, force=True)
        doc._.health_effects = unique_health_effects
