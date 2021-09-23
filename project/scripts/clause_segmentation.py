import spacy, benepar, typer
from spacy.tokens import Span, Doc, Token
from typing import Tuple, List
from wasabi import Printer


@spacy.registry.misc("clause_segmentation")
def return_clause_segmentation():
    return clause_segmentation


def benepar_split(sentence: Span) -> List[Span]:
    """Split a Span into individual clauses
    sentence (Span): Sentence of a Doc object
    RETURNS (List[Span]): List of extracted clauses
    """
    split_sentences = []
    for constituent in sentence._.constituents:
        # Include every clause with the label "S" (Sentence) and with the root sentence as parent
        if "S" in constituent._.labels and constituent._.parent == sentence:
            split_sentences.append(constituent)

    # If no clause found just append the whole sentence
    if len(split_sentences) == 0:
        split_sentences.append(sentence)
    return split_sentences


def add_placeholder(clauses: List[Span]) -> List[Tuple[Doc, List[Span]]]:
    """Replace entities with label placeholder
    clauses (List[Span]): List of extracted clauses
    RETURNS (List[Tuple[Doc, List[Span]]]): List of tuples with placeholder spans
    """
    clause_tuples = []
    for clause in clauses:
        if len(clause.ents) > 0:
            for index in range(len(clause.ents)):
                start = clause.ents[index].start
                end = clause.ents[index].end

                words = []
                replaced = False

                for word in clause:
                    if word.i >= start and word.i < end and not replaced:
                        words.append(f"<{clause.ents[index].label_}>")
                        replaced = True
                    elif not (word.i >= start and word.i < end):
                        words.append(word.text)

                doc = Doc(clause.doc.vocab, words=words)
                entity = str(clause.ents[index]).lower().strip()
                entity = entity.replace(" ", "_")

                clause_tuples.append((doc, entity, clause.ents[index].label_))
        else:
            words = [word.text for word in clause]
            doc = Doc(clause.doc.vocab, words=words)
            clause_tuples.append((doc, None, None))

    return clause_tuples


def clause_segmentation(doc: Doc) -> List[Tuple[Doc, Span]]:
    """Extract clauses and add placeholder for entities
    doc (Doc): Document
    RETURNS (List[Tuple[Doc, List[Span]]]): List of tuples with placeholder spans
    """
    clauses = []
    for sent in doc.sents:
        extracted_clauses = benepar_split(sent)
        placeholder_clauses = add_placeholder(extracted_clauses)
        clauses.extend(placeholder_clauses)
    return clauses


def test_clause_segmentation():

    msg = Printer()

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
    example_text = "My name is Edward and I am writing code in Germany"
    doc = nlp(example_text)
    clauses = clause_segmentation(doc)

    msg.divider("Clause Segmentation")
    print(f"Text: {doc.text}")
    print(f"Entities {doc.ents} \n")
    for clause in clauses:
        print(f">> Span: {clause[0]}")
        print(f">> Entity: {clause[1]}")
        print(f">> Label: {clause[2]} \n")


if __name__ == "__main__":
    typer.run(test_clause_segmentation)
