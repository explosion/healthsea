import json
import typer
from wasabi import Printer
from wasabi import table
from pathlib import Path

msg = Printer()


def main(
    annotation_path: Path,
):
    """Create a small summary of the NER annotation"""

    # Load in the dataset
    dataset = []
    with annotation_path.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            dataset.append(json.loads(line))

    conditions = {}
    benefits = {}
    entry_with_no_entity = 0

    for example in dataset:
        if example["answer"] == "accept":

            # Count examples with no entity
            if "spans" not in example or len(example["spans"]) == 0:
                entry_with_no_entity += 1

            else:
                for span in example["spans"]:
                    token_start = span["token_start"]
                    token_end = span["token_end"]
                    entity = ""

                    # Collect entites with multiple tokens
                    for token in example["tokens"]:
                        if token["id"] in range(token_start, token_end + 1):
                            entity += token["text"]

                    entity = entity.replace(" ", "_").strip().lower()

                    if span["label"] == "CONDITION":
                        if entity not in conditions:
                            conditions[entity] = 0
                        conditions[entity] += 1

                    else:
                        if entity not in benefits:
                            benefits[entity] = 0
                        benefits[entity] += 1

    total_condition = sum(conditions.values())
    total_benefits = sum(benefits.values())

    sorted_conditions = sorted(conditions.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    sorted_benefits = sorted(benefits.items(), key=lambda x: x[1], reverse=True)[:10]

    # Printing
    msg.divider("NER dataset summary")
    msg.info(
        f"{len(dataset)} examples with {entry_with_no_entity} without entities ({round((entry_with_no_entity/len(dataset))*100,2)}%)"
    )

    table_data = [
        (
            "All entities",
            total_condition + total_benefits,
            len(conditions) + len(benefits),
        ),
        ("CONDITION", total_condition, len(conditions)),
        ("BENEFIT", total_benefits, len(benefits)),
    ]
    header = ("Label", "Total", "Unique")
    aligns = ("c", "c", "c")
    print(table(table_data, header=header, divider=True, aligns=aligns))

    # Print top entities per group
    table_data_condition = []
    table_data_benefit = []
    header_condition = ("Condition", "Total")
    header_benefit = ("Benefit", "Total")
    aligns = ("c", "c")

    for top_benefit, top_condition in zip(sorted_benefits, sorted_conditions):
        table_data_condition.append((top_condition[0], top_condition[1]))
        table_data_benefit.append((top_benefit[0], top_benefit[1]))

    print(
        table(
            table_data_condition, header=header_condition, divider=True, aligns=aligns
        )
    )
    print(table(table_data_benefit, header=header_benefit, divider=True, aligns=aligns))


if __name__ == "__main__":
    typer.run(main)
