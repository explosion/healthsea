import spacy
from spacy.scorer import PRFScore
import typer
import json
from pathlib import Path
from wasabi import Printer
from wasabi import table

from clausecat import clausecat_component
from clausecat import clausecat_model
from clausecat import clausecat_reader
from clausecat import clause_segmentation
from clausecat import clause_aggregation

msg = Printer()


def main(model_path: Path, eval_path: Path):
    """This script is used to evaluate the healthsea pipeline with manually created examples."""

    nlp = spacy.load(model_path)

    with open(eval_path, "r", encoding="utf8") as jsonfile:
        evaluation = json.load(jsonfile)

    scorer = {
        "CONDITION": PRFScore(),
        "BENEFIT": PRFScore(),
        "POSITIVE": PRFScore(),
        "NEGATIVE": PRFScore(),
        "NEUTRAL": PRFScore(),
    }
    data = {
        "CONDITION": [],
        "BENEFIT": [],
        "POSITIVE": 0,
        "NEGATIVE": 0,
        "NEUTRAL": 0,
    }
    uncorrect_ner = []
    uncorrect_textcat = []

    for eval in evaluation:
        # Count evaluation examples per entity and classification
        for entity in eval["health_effects"]:
            if entity not in data[eval["health_effects"][entity]["label"]]:
                data[eval["health_effects"][entity]["label"]].append(entity)
            data[eval["health_effects"][entity]["classification"]] += 1

        # Compare prediction with expected results
        doc = nlp(eval["text"])
        health_effects_gold = eval["health_effects"]
        health_effects_pred = doc._.health_effects

        entities_gold = list(health_effects_gold.keys())
        entities_pred = list(health_effects_pred.keys())

        # Compare NER results
        middle_join = [value for value in entities_gold if value in entities_pred]
        left_join = [value for value in entities_gold if value not in entities_pred]
        right_join = [value for value in entities_pred if value not in entities_gold]

        for entity in middle_join:
            if (
                health_effects_pred[entity]["label"]
                == health_effects_gold[entity]["label"]
            ):
                scorer[health_effects_gold[entity]["label"]].tp += 1
            else:
                scorer[health_effects_gold[entity]["label"]].fn += 1
                scorer[health_effects_pred[entity]["label"]].fp += 1
                uncorrect_ner.append(
                    f"({eval['number']}) {entity} (Prediction: {health_effects_pred[entity]['label']}) (Expected: {health_effects_gold[entity]['label']})"
                )

        for entity in left_join:
            scorer[health_effects_gold[entity]["label"]].fn += 1
            uncorrect_ner.append(f"({eval['number']}) False negative {entity}")
            data[eval["health_effects"][entity]["classification"]] -= 1

        for entity in right_join:
            scorer[health_effects_pred[entity]["label"]].fp += 1
            uncorrect_ner.append(f"({eval['number']}) False positive {entity}")

        # Compare TEXTCAT results
        for entity_index in middle_join:
            if (
                health_effects_pred[entity_index]["effect"]
                == health_effects_gold[entity_index]["classification"]
            ):
                scorer[health_effects_gold[entity_index]["classification"]].tp += 1
            else:
                uncorrect_textcat.append(
                    f"({eval['number']}) {entity_index} (Prediction: {health_effects_pred[entity_index]['effect']}) (Expected: {health_effects_gold[entity_index]['classification']}) ({eval['text']})"
                )
                scorer[health_effects_pred[entity_index]["effect"]].fp += 1
                scorer[health_effects_gold[entity_index]["classification"]].fn += 1

    # Printing results
    msg.divider(f"Total reviews: {len(evaluation)}")
    msg.divider(f"Evaluating Named Entity Recognition")
    msg.info(
        f"Total examples ({len(data['BENEFIT'])+len(data['CONDITION'])}), CONDITION ({len(data['CONDITION'])}), BENEFIT ({len(data['BENEFIT'])})"
    )

    ner_data = [
        (
            "CONDITION",
            round(scorer["CONDITION"].fscore, 2),
            round(scorer["CONDITION"].recall, 2),
            round(scorer["CONDITION"].precision, 2),
        ),
        (
            "BENEFIT",
            round(scorer["BENEFIT"].fscore, 2),
            round(scorer["BENEFIT"].recall, 2),
            round(scorer["BENEFIT"].precision, 2),
        ),
        (
            "AVERAGE",
            round((scorer["BENEFIT"].fscore + scorer["CONDITION"].fscore) / 2, 2),
            round((scorer["BENEFIT"].recall + scorer["CONDITION"].recall) / 2, 2),
            round((scorer["BENEFIT"].precision + scorer["CONDITION"].precision) / 2, 2),
        ),
    ]
    header = ("Label", "F-Score", "Recall", "Precision")
    widths = (10, 10, 10, 10)
    aligns = ("l", "c", "c", "c")
    print(table(ner_data, header=header, divider=True, widths=widths, aligns=aligns))

    ner_error_percentage = round(
        (len(uncorrect_ner) / (len(data["BENEFIT"]) + len(data["CONDITION"]))) * 100, 2
    )

    msg.fail(
        f"{len(uncorrect_ner)}/{len(data['BENEFIT'])+len(data['CONDITION'])} false predictions ({ner_error_percentage}%)"
    )
    for error_ner in uncorrect_ner:
        print("  >> " + error_ner)

    msg.divider(f"Evaluating Text Classification")
    msg.info(
        f"Total examples ({data['POSITIVE']+data['NEGATIVE']+data['NEUTRAL']}), POSITIVE ({data['POSITIVE']}), NEGATIVE ({data['NEGATIVE']}), NEUTRAL ({data['NEUTRAL']})"
    )

    textcat_data = [
        (
            "POSITIVE",
            round(scorer["POSITIVE"].fscore, 2),
            round(scorer["POSITIVE"].recall, 2),
            round(scorer["POSITIVE"].precision, 2),
        ),
        (
            "NEGATIVE",
            round(scorer["NEGATIVE"].fscore, 2),
            round(scorer["NEGATIVE"].recall, 2),
            round(scorer["NEGATIVE"].precision, 2),
        ),
        (
            "NEUTRAL",
            round(scorer["NEUTRAL"].fscore, 2),
            round(scorer["NEUTRAL"].recall, 2),
            round(scorer["NEUTRAL"].precision, 2),
        ),
        (
            "AVERAGE",
            round(
                (
                    scorer["NEUTRAL"].fscore
                    + scorer["NEGATIVE"].fscore
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
            round(
                (
                    scorer["NEUTRAL"].recall
                    + scorer["NEGATIVE"].recall
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
            round(
                (
                    scorer["NEUTRAL"].precision
                    + scorer["NEGATIVE"].precision
                    + scorer["POSITIVE"].precision
                )
                / 3,
                2,
            ),
        ),
    ]
    header = ("Label", "F-Score", "Recall", "Precision")
    widths = (10, 10, 10, 10)
    aligns = ("l", "c", "c", "c")
    print(
        table(textcat_data, header=header, divider=True, widths=widths, aligns=aligns)
    )

    clausecat_error_percentage = round(
        (
            len(uncorrect_textcat)
            / (data["POSITIVE"] + data["NEGATIVE"] + data["NEUTRAL"])
        )
        * 100,
        2,
    )
    msg.fail(
        f"{len(uncorrect_textcat)}/{data['POSITIVE']+data['NEGATIVE']+data['NEUTRAL']} false predictions ({clausecat_error_percentage}%)"
    )
    for error_textcat in uncorrect_textcat:
        print("  >> " + error_textcat)


if __name__ == "__main__":
    typer.run(main)
