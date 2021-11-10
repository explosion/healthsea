# Welcome to Healthsea ✨
Create better access to health with machine learning and natural language processing. This is the spaCy project of healthsea, a pipeline for analyzing user reviews to complementary medicine and extracting effects on health. 

If you're interested to learn more, here's the [blog post]()!

![](img/healthsea_anim.gif)

## 💉 Creating better access to health
The goal of Healthsea is to analyze user-written reviews of supplements in relation to their effect on health. Based on this analysis, we aim to provide product recommendations.
For many people, supplements are a addition for maintaining and achieving health goals. Due to its rising popularity consumers have an increasing access to a variety of products.

However, it's likely that most of the products are redundant or produced in a "quantity over quality" fashion to maximize profit. This creates a white noise of products that make it hard to find suitable supplements. A
To estimate possible health effects, it's required to read and evaluate product reviews manually. 

**With Healthsea, we aim to automize the analysis and provide information in an easily digestible way.** ✨


## 📖 Documentation

| Documentation              |                                                                |
| -------------------------- | -------------------------------------------------------------- |
| ⚙️ **Pipeline**      | Learn more about the pipeline               |
| 🪐 **spaCy project**      | Learn more about the spaCy project               |
| ⭐️ **Features**           | Learn more about the features of the pipeline              |
| ✨ **Demo**                | Learn more about the streamlit app              |

## ⚙️ Pipeline
The Healthsea pipeline uses Named Entity Recognition to detect two types of entities ```Condition``` and ```Benefit```.

 ```Conditions``` can be defined as health aspects that can be improved by decreasing them, they include diseases, symptoms and vague descriptions of health problems (e.g. pain in back). ```Benefits``` on the other hand, are desired states of health (muscle recovery, glowing skin) improved by an increase.

![](img/ner_guide.PNG)

We use Text Classification for predicting health effects and apply a Clause Segmentation algorithm based on the [benepar parser](). The segmentation splits sentences and uses a blinder to mask entities for an improve in generalization. The model predicts four exclusive classes: ```Positive, Negative, Neutral, Anamnesis```.

![](img/clausecat_guide.PNG)

## 🪐 spaCy project
The ```project``` folder contains a [spaCy project](https://spacy.io/usage/projects) that holds all the data and training workflows.

Use ```spacy project run``` inside the project folder to get an overview of all commands and assets. For more detailed documentation, visit the [project folders readme](https://github.com/thomashacker/healthsea/tree/main/project). 

## ⭐️ Features
The spaCy project includes following features:
- Training a [Named Entity Recognition model](https://spacy.io/usage/linguistic-features#named-entities) 
- Building [custom spaCy components](https://spacy.io/usage/processing-pipelines#custom-components) and utilizing pipelines from the [spaCy universe](https://spacy.io/universe)
- Bulding and training a custom Text Classificaton model (Clausecat)

## ✨ Demo
> Note: You need to have use git lfs fetch to get receive the data (700mb) 

The ```demo``` folder contains two [streamlit apps](https://streamlit.io/) which visualize an analyzed dataset and show the pipeline.

For more detailed documentation, visit the [demo folders readme](https://github.com/thomashacker/healthsea/tree/main/demo).


