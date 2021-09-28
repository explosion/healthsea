# Welcome to Healthsea ✨
Create better access to health with machine learning and natural language processing. 
A [spaCy](https://github.com/explosion/spaCy) end-to-end pipeline for analyzing user reviews of supplementary products and their effects on health.

## 💉 Creating better access to health
The goal of healthsea is to analyze user reviews of supplements and extract their effects on health. We want to create product recommendations for specific health conditions based on what customers wrote in their reviews.

The usage of complementary medicine is an excellent addition when it comes to maintaining health. Due to its rising popularity, consumers have access to a wide variety of products. However, it's likely that most of them are redundant and produced in a "quantity over quality" fashion to maximize profit. This white noise of products makes it hard for customers to find valuable supplements. Additionally, supplement manufacturers are not allowed to claim health benefits for their products if not scientifically approved, increasing manual work for customers.

Healthsea aims to reduce manual work by automizing this process and thus create better access to health. 🤗 

## 📖 Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| 🪐 **spaCy project**      | Learn more about the spaCy project               |
| ✨ **Demo**                | Learn more about the streamlit app              |
| ⭐️ **Features**           | Which features does the healthsea project include               |

## 🪐 spaCy project
The ```project``` folder contains a [spaCy project](https://spacy.io/usage/projects) which builds and trains the healthsea pipeline with spaCy v3.

Use ```spacy project run``` inside the project folder to get an overview of all possible commands and assets. For more detailed documentation, visit the [project folders readme](https://github.com/thomashacker/healthsea/tree/main/project). The ```project.yml``` file contains all commands, scripts and variables, which can be also changed.

## ✨ Demo
The ```demo``` folder contains a [streamlit app](https://streamlit.io/) which visualizes an analyzed dataset and gives product recommendations for specific health conditions.

For more detailed documentation, visit the [demo folders readme](https://github.com/thomashacker/healthsea/tree/main/demo).

## ⭐️ Features
This project includes following features:
- Training a [Named Entity Recognition model](https://spacy.io/usage/linguistic-features#named-entities) 
- Training a custom Text Classificaton model
- Building a [custom spaCy component](https://spacy.io/usage/processing-pipelines#custom-components)
- Assembling spaCy components into one [pipeline](https://spacy.io/usage/processing-pipelines#pipelines)
- Visualizing an analyzed dataset and providing product/substance recommendations

