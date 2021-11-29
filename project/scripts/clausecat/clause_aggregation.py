from spacy.tokens import Doc
from spacy.language import Language
import operator


@Language.factory("healthsea.aggregation.v1")
def create_clause_aggregation(nlp: Language, name: str):
    return ClauseAggregation(nlp, name)


class ClauseAggregation:
    """Aggregate the predicted effects from the clausecat and apply the patient information logic"""

    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp
        self.name = name

    def __call__(self, doc: Doc):
        patient_information = []
        health_effects = {}

        for clause in doc._.clauses:
            classification = max(clause["cats"].items(), key=operator.itemgetter(1))[0]

            if not clause["has_ent"]:
                if len(patient_information) > 0:
                    patient_information[-1][1].append(classification)
                continue

            entity = str(clause["ent_name"]).replace(" ", "_").strip().lower()

            # Collect patient information
            if classification == "ANAMNESIS" and entity is not None:
                patient_information.append((entity, []))

            # Collect health effects
            if entity is not None:
                if entity not in health_effects:
                    health_effects[entity] = {
                        "effects": [],
                        "effect": "NEUTRAL",
                        "label": str(clause["blinder"])
                        .replace("_", "")
                        .replace("_", ""),
                        "text": clause["ent_name"],
                    }
                health_effects[entity]["effects"].append(classification)

        # Add patient information to list of health effects
        for patient_health in patient_information:
            entity = patient_health[0]
            health_effects[entity]["effects"] += patient_health[1]

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

        doc.set_extension("health_effects", default={}, force=True)
        doc._.health_effects = health_effects

        return doc
