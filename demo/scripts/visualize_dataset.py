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

data_load_state = st.subheader("Loading data...")
health_aspects, products, conditions, benefits = load_data(
    health_aspect_path, product_path, condition_path, benefit_path
)
search_engine = HealthseaSearch(health_aspects, products, conditions, benefits)
data_load_state.subheader("Discover the results of the pipeline ✨")

intro, jellyfish = st.columns(2)

jellyfish.image("data/img/jellyfish.png", use_column_width="auto")
intro.markdown(
    "With healthsea we analyzed up to 1 million reviews. You can use the app to explore the results and get product and substance recommendations based on your choosen health aspect."
)
intro.markdown("""Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.""")

st.markdown("""---""")

# KPI

st.markdown(central_text("Dataset stats"), unsafe_allow_html=True)

tmp1,kpi_products,tmp3,kpi_reviews,tmp5 = st.columns(5)
kpi_aspects, kpi_condition, kpi_benefit = st.columns(3)

kpi_products.markdown(kpi(len(products), "Products"), unsafe_allow_html=True)
kpi_reviews.markdown(kpi(933.240, "Reviews"), unsafe_allow_html=True)

kpi_aspects.markdown(
    kpi(len(conditions) + len(benefits), "Health aspects"), unsafe_allow_html=True
)
kpi_condition.markdown(kpi(len(conditions), "Conditions"), unsafe_allow_html=True)
kpi_benefit.markdown(kpi(len(benefits), "Benefits"), unsafe_allow_html=True)

st.markdown("""---""")

# Search
search = st.text_input(label="Search for an health aspect", value="joint pain")
n = st.slider("Show top n results", min_value=1, max_value=100, value=10)

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
    kpi_alias.write("Including similar aspects")
    kpi_alias.json(aspect_alias)
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
