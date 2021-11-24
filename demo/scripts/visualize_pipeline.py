import streamlit as st
import spacy
from spacy_streamlit import visualize_ner
from support_functions import HealthseaPipe
import operator

healthsea_pipe = HealthseaPipe()

color_code = {
    "POSITIVE": ("#3C9E58", "#1B7735"),
    "NEGATIVE": ("#FF166A", "#C0094B"),
    "NEUTRAL": ("#7E7E7E", "#4E4747"),
    "ANAMNESIS": ("#E49A55", "#AD6B2D"),
}

example_reviews = [
    "This is great for joint pain.",
    "This help joint pain but causes rashes",
    "I'm diagnosed with gastritis. This product helped!",
    "Made my insomnia worse",
    "Didn't help my energy levels",
]

# Functions
def kpi(n, text):
    html = f"""
    <div class='kpi'>
        <h1>{n}</h1>
        <span>{text}</span>
    </div>
    """
    return html


def central_text(text):
    html = f"""<h2 class='central_text'>{text}</h2>"""
    return html


def format_clause(text, meta, pred):
    html = f"""
    <div>
        <div class="clause" style="background-color:{color_code[pred][0]} ; box-shadow: 0px 5px {color_code[pred][1]}; border-color:{color_code[pred][1]};">
            <div class="clause_text">{text}</div>
        </div>
        <div class="clause_meta">
            <div>{meta}</div>
        </div>
    </div>"""
    return html


def format_effect(text, pred):
    html = f"""
    <div>
        <div class="clause" style="background-color:{color_code[pred][0]} ; box-shadow: 0px 5px {color_code[pred][1]}; border-color:{color_code[pred][1]};">
            <div class="clause_text">{text}</div>
        </div>
    </div>"""
    return html


# Header
with open("scripts/style.css") as f:
    st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

st.title("Welcome to Healthsea 🪐")

intro, jellyfish = st.columns(2)
jellyfish.markdown("\n")

####
data_load_state = intro.subheader("Loading model...")

# Load model
nlp = spacy.load("en_healthsea")

data_load_state.subheader("Create easier access to health✨")
####

jellyfish.image("data/img/Jellymation.gif")
intro.markdown(
    "Healthsea is a pipeline for analyzing user reviews to supplement products with the goal of extracting stated effects on health."
)
intro.markdown(
    """With this app, you're able to explore the results of the pipeline by writing example reviews. 
    You'll get insights into the functionality of the Named Entity Recognition, Clause Segmentation, Blinding and Text Classification. 
    """
)
intro.markdown(
    """If you want to learn more about the pipeline and it's architecture, you can read more in our [blog post](explosion.ai).
    """
)

st.markdown("""---""")

# Pipeline
st.markdown(central_text("⚙️ Pipeline"), unsafe_allow_html=True)

check = st.checkbox("Use predefined examples")

if not check:
    text = st.text_input(label="Write a review", value="This is great for joint pain!")
else:
    text = st.selectbox("Predefined example reviews", example_reviews)
doc = nlp(text)

# NER
visualize_ner(
    doc,
    labels=nlp.get_pipe("ner").labels,
    show_table=False,
    title="✨ Named Entity Recognition",
    colors={"CONDITION": "#FF4B76", "BENEFIT": "#629B68"},
)

st.markdown("""---""")

# Segmentation, Blinding, Classification
st.markdown("## 🔮 Segmentation, Blinding, Classification")

clauses = healthsea_pipe.get_clauses(doc)
for doc_clause, clause in zip(clauses, doc._.clauses):
    classification = max(clause["cats"].items(), key=operator.itemgetter(1))[0]
    percentage = round(float(clause["cats"][classification]) * 100, 2)
    meta = f"{clause['ent_name']} ({classification} {percentage}%)"

    st.markdown(
        format_clause(doc_clause.text, meta, classification), unsafe_allow_html=True
    )
    st.markdown("\n")

st.markdown("""---""")

# Aggregation
st.markdown("## 🔗 Aggregation")

for effect in doc._.health_effects:
    st.markdown(
        format_effect(
            f"{doc._.health_effects[effect]['effect']} effect on {effect}",
            doc._.health_effects[effect]["effect"],
        ),
        unsafe_allow_html=True,
    )
    st.markdown("\n")

st.markdown("""---""")
# Indepth
st.markdown("## 🔧 Pipeline attributes")
clauses_col, effect_col = st.columns(2)

clauses_col.markdown("### doc._.clauses")
for clause in doc._.clauses:
    clauses_col.json(clause)
effect_col.markdown("### doc._.health_effects")
effect_col.json(doc._.health_effects)
