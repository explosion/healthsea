# Welcome to Healthsea ✨
Create better access to health with machine learning and natural language processing. The journey of developing healthsea, an end-to-end [spaCy](https://github.com/explosion/spaCy) v3 pipeline for analyzing user reviews to complementary medicine and extract their potential effects on health.

![](img/healthsea_anim.gif)

## 💉 Creating better access to health
The goal of our project is to analyze user-written reviews of supplements in relation to their effect on health conditions. Based on this analysis, we aim to provide product recommendations.
For many people, supplements are a great addition for maintaining health and due to its rising popularity, consumers have an increasing access to a variety of products.

However, it's likely that most of the products are redundant or produced in a "quantity over quality" fashion to maximize profit. This creates a white noise of products that make it hard to find suitable supplements. Additionally, manufacturers are not allowed to claim health benefits for their products, which increases the manual work for customers.
To estimate possible health effects, it's required to read and evaluate product reviews manually. 

**With Healthsea, we aim to automize the analysis and provide information in an easily digestible way.** ✨


## 📖 Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| 🪐 **spaCy project**      | Learn more about the spaCy project               |
| ✨ **Demo**                | Learn more about the streamlit app              |
| ⭐️ **Features**           | Which features does the healthsea project include               |

## 🪐 spaCy project
The ```project``` folder contains a [spaCy project](https://spacy.io/usage/projects) that holds all the training data and trains the pipeline.

Use ```spacy project run``` inside the project folder to get an overview of all commands and assets. For more detailed documentation, visit the [project folders readme](https://github.com/thomashacker/healthsea/tree/main/project). 

## ✨ Demo
The ```demo``` folder contains a [streamlit app](https://streamlit.io/) which visualizes the analyzed dataset and shows the pipeline in production. The app shows product/substance recommendations for specific health conditions.

For more detailed documentation, visit the [demo folders readme](https://github.com/thomashacker/healthsea/tree/main/demo).

## ⭐️ Features
This project includes following features:
- Training a [Named Entity Recognition model](https://spacy.io/usage/linguistic-features#named-entities) 
- Training a custom Text Classificaton model
- Building a [custom spaCy component](https://spacy.io/usage/processing-pipelines#custom-components)
- Assembling spaCy components into one [pipeline](https://spacy.io/usage/processing-pipelines#pipelines)
- Visualizing an analyzed dataset and providing product/substance recommendations

