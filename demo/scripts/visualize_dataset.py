import streamlit as st
import typer
from pathlib import Path
import json
from support_functions import HealthseaSearch

# Configuration
health_aspect_path = Path("data/health_aspects.json")
product_path = Path("data/products.json")
condition_path = Path("data/condition_vectors.json")
benefit_path = Path("data/benefit_vectors.json")


# Load data
@st.cache(allow_output_mutation=True)
def load_data(
    _health_aspect_path: Path,
    _product_path: Path,
    _condition_path: Path,
    _benefit_path: Path,
):
    with open(_health_aspect_path) as reader:
        health_aspects = json.load(reader)
    with open(_product_path) as reader:
        products = json.load(reader)
    with open(_condition_path) as reader:
        conditions = json.load(reader)
    with open(_benefit_path) as reader:
        benefits = json.load(reader)
    return health_aspects, products, conditions, benefits


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


# Header
with open("scripts/style.css") as f:
    st.markdown("<style>" + f.read() + "</style>", unsafe_allow_html=True)

st.title("Welcome to Healthsea 🪐")

intro, jellyfish = st.columns(2)
jellyfish.markdown("\n")

data_load_state = intro.subheader("Loading data...")
health_aspects, products, conditions, benefits = load_data(
    health_aspect_path, product_path, condition_path, benefit_path
)
search_engine = HealthseaSearch(health_aspects, products, conditions, benefits)
data_load_state.subheader("Create easier access to health✨")

jellyfish.image("data/img/Jellymation.gif")
intro.markdown(
    "Healthsea is a pipeline for analyzing user reviews to supplement products with the goal of extracting stated effects on health."
)
intro.markdown(
    """With this app, you're able to explore the results of healthsea that analyzed up to 1 million reviews. 
    You can search for any health aspect, whether it is an disease (e.g. joint pain) or a desired health effect such as (e.g. energy),
    and the app will return a list of products and substances with the highest score.
    """
)
intro.markdown(
    """If you want to learn more about the pipeline and it's architecture, you can read more in our [blog post](explosion.ai).
    """
)

st.markdown("""---""")

# KPI

st.markdown(central_text("🎀 Dataset"), unsafe_allow_html=True)

kpi_products, kpi_reviews, kpi_condition, kpi_benefit = st.columns(4)

kpi_products.markdown(kpi(len(products), "Products"), unsafe_allow_html=True)
kpi_reviews.markdown(kpi(933.240, "Reviews"), unsafe_allow_html=True)
kpi_condition.markdown(kpi(len(conditions), "Conditions"), unsafe_allow_html=True)
kpi_benefit.markdown(kpi(len(benefits), "Benefits"), unsafe_allow_html=True)

st.markdown("""---""")

# Search
search = st.text_input(label="Search for an health aspect", value="joint pain")
n = st.slider("Show top n results", min_value=10, max_value=1000, value=25)

st.markdown("""---""")
st.markdown(central_text("🧃 Products"), unsafe_allow_html=True)

# DataFrame
st.write(search_engine.get_products_df(search, n))

# KPI & Alias
aspect_alias = search_engine.get_aspect(search)["alias"]

if len(aspect_alias) > 0:
    kpi_mentions, kpi_product_mentions, kpi_alias = st.columns(3)
    kpi_mentions.markdown(
        kpi(search_engine.get_aspect_meta(search)["frequency"], "Mentions"),
        unsafe_allow_html=True,
    )
    kpi_product_mentions.markdown(
        kpi(len(search_engine.get_aspect(search)["products"]), "Products"),
        unsafe_allow_html=True,
    )
    kpi_alias.markdown(
        kpi(len(aspect_alias), "Similar health aspects"),
        unsafe_allow_html=True,
    )

    vectors = []
    main_aspect = search_engine.get_aspect_meta(search)
    vectors.append((main_aspect["name"], main_aspect["vector"]))
    for aspect in aspect_alias:
        current_aspect = search_engine.get_aspect_meta(aspect)
        vectors.append((current_aspect["name"], current_aspect["vector"]))
    st.markdown("\n")
    st.write(search_engine.tsne_plot(vectors))

else:
    kpi_mentions, kpi_product_mentions = st.columns(2)
    kpi_mentions.markdown(
        kpi(search_engine.get_aspect_meta(search)["frequency"], "Mentions"),
        unsafe_allow_html=True,
    )
    kpi_product_mentions.markdown(
        kpi(len(search_engine.get_aspect(search)["products"]), "Products"),
        unsafe_allow_html=True,
    )

st.markdown("""---""")

# Substances
st.markdown(central_text("🍯 Substances"), unsafe_allow_html=True)

# DataFrame
st.write(search_engine.get_substances_df(search, n))
kpi_tmp, kpi_substances = st.columns(2)
kpi_substances.markdown(
    kpi(len(search_engine.get_aspect(search)["substance"]), "Substances"),
    unsafe_allow_html=True,
)
