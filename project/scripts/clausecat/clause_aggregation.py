from spacy.tokens import Doc
from spacy.language import Language
import operator


@Language.factory("healthsea.aggregation")
def create_clause_aggregation(nlp: Language, name: str):
    return Clause_aggregation(nlp)


class Clause_aggregation:
    def __init__(self, nlp: Language):
        self.nlp = nlp

    def __call__(self, doc: Doc):

        patient_information = []
        health_effects = {}

        for clause in doc._.clauses:
            classification = max(clause["cats"].items(), key=operator.itemgetter(1))[0]
            entity = clause["ent_name"].replace(" ", "_").strip()

            # Collect patient information
            if classification == "ANAMNESIS" and entity != None:
                patient_information.append((entity, []))
            elif entity == None:
                for patient_health in patient_information:
                    patient_health[1].append(classification)

            # Collect health effects
            if entity != None:
                if entity not in health_effects:
                    health_effects[entity] = {
                        "effects": [],
                        "effect": "NEUTRAL",
                        "label": clause["blinder"],
                        "text": clause["ent_name"],
                    }
                health_effects[entity]["effects"].append(classification)

        # Add patient information to list of health effects
        for patient_health in patient_information:
            entity = patient_health[0]
            score = 0
            for classification in patient_health[1]:
                health_effects[entity]["effects"].append(classification)

        # Aggregate health effects
        for entity in health_effects:
            score = 0
            for classification in health_effects[entity]["effects"]:
                if classification == "POSITIVE":
                    score += 1
                elif classification == "NEGATIVE":
                    score -= 1

            if score > 0:
                aggregated_classification = "POSITIVE"
            elif score < 0:
                aggregated_classification = "NEGATIVE"
            else:
                aggregated_classification = "NEUTRAL"

            health_effects[entity]["effect"] = aggregated_classification

        doc.set_extension("effects", default={}, force=True)
        doc._.effects = health_effects

        return doc
