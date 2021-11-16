import pandas as pd
import difflib
from spacy.tokens import Doc

import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import numpy as np


class HealthseaSearch:
    def __init__(self, _health_aspects, _products, _conditions, _benefits):
        self.health_aspects = _health_aspects
        self.products = _products
        self.conditions = _conditions
        self.benefits = _benefits

    def __call__(self, query):
        return query

    # Load product meta
    def get_products(self, _aspect, n):
        product_list = []
        product_ids = {}
        _n = n
        _aspect = _aspect.replace(" ", "_")
        if _aspect in self.health_aspects:
            aspect = self.health_aspects[_aspect]
        else:
            _aspect = difflib.get_close_matches("_aspect", self.health_aspects.keys())[
                0
            ]
            aspect = self.health_aspects[_aspect]

        product_scoring = aspect["products"]
        if n != 0:
            if n > len(product_scoring):
                n = len(product_scoring)
            product_scoring = aspect["products"][:n]

        for product in product_scoring:
            if product[1] not in product_ids:
                product_list.append((product[0], self.products[product[1]], _aspect))
                product_ids[product[1]] = 1

        for alias in aspect["alias"]:
            n = _n
            _product_scoring = self.health_aspects[alias]["products"]
            if n != 0:
                if n > len(_product_scoring):
                    n = len(_product_scoring)
                _product_scoring = self.health_aspects[alias]["products"][:n]

            for product in _product_scoring:
                if product[1] not in product_ids:
                    product_list.append((product[0], self.products[product[1]], alias))
                    product_ids[product[1]] = 1

        n = _n
        if len(product_list) > n and n != 0:
            product_list = product_list[:n]
        product_list = sorted(product_list, key=lambda tup: tup[0], reverse=True)

        return product_list

    # Load product meta and return as DataFrame
    def get_products_df(self, _aspect, n):
        product_list = self.get_products(_aspect, n)
        product_data = {
            "product": [],
            "score": [],
            "health_aspect": [],
            "rating": [],
            "reviews": [],
        }
        for product in product_list:
            product_data["score"].append(product[0])
            product_data["product"].append(product[1]["name"])
            product_data["health_aspect"].append(product[2])
            product_data["rating"].append(product[1]["rating"])
            product_data["reviews"].append(product[1]["review_count"])

        datatypes = {
            "product": str,
            "score": int,
            "health_aspect": str,
            "rating": str,
            "reviews": int,
        }

        df = pd.DataFrame(data=product_data)
        df = df.astype(datatypes)

        return df

    # Get health aspect
    def get_aspect(self, _aspect):
        _aspect = _aspect.replace(" ", "_")
        if _aspect in self.health_aspects:
            return self.health_aspects[_aspect]
        else:
            _aspect = difflib.get_close_matches("_aspect", self.health_aspects.keys())[
                0
            ]
            return self.health_aspects[_aspect]

    # Get health aspect meta
    def get_aspect_meta(self, _aspect):
        _aspect = _aspect.replace(" ", "_")
        if _aspect in self.conditions:
            return self.conditions[_aspect]
        elif _aspect in self.benefits:
            return self.benefits[_aspect]
        else:
            _aspect = difflib.get_close_matches("_aspect", self.conditions.keys())[0]
            return self.conditions[_aspect]

    # Plotting vectors (2D/3D)
    def tsne_plot(self, dataset):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []

        for i in dataset:
            tokens.append(np.array(i[1]))
            labels.append(i[0])

        if len(dataset) > 2:
            tsne_model = TSNE(
                perplexity=40, n_components=3, init="pca", n_iter=2500, random_state=23
            )

            new_values = tsne_model.fit_transform(tokens)

            x = []
            y = []
            z = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
                z.append(value[2])

            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                text=labels,
                textposition="top right",
                mode="lines+markers+text",
                marker={
                    "size": 10,
                    "opacity": 0.8,
                },
            )

            # Configure the layout.
            layout = go.Layout(
                margin={"l": 0, "r": 0, "b": 0, "t": 0}, font={"color": "#DF55E2"}
            )

            data = [trace]

            return go.Figure(data=data, layout=layout)

        else:
            tsne_model = TSNE(
                perplexity=40, n_components=2, init="pca", n_iter=2500, random_state=23
            )

            new_values = tsne_model.fit_transform(tokens)

            x = []
            y = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])

            trace = go.Scatter(
                x=x,
                y=y,
                text=labels,
                textposition="top right",
                mode="lines+markers+text",
                marker={
                    "size": 10,
                    "opacity": 0.8,
                },
            )

            # Configure the layout.
            layout = go.Layout(
                margin={"l": 0, "r": 0, "b": 0, "t": 0}, font={"color": "#DF55E2"}
            )

            data = [trace]

            return go.Figure(data=data, layout=layout)

    # Load substance meta
    def get_substances(self, _aspect, n):
        substance_list = []
        substance_ids = {}
        exclude = ["sodium", "sugar", "sugar_alcohol"]
        _n = n
        _aspect = _aspect.replace(" ", "_")
        if _aspect in self.health_aspects:
            aspect = self.health_aspects[_aspect]
        else:
            _aspect = difflib.get_close_matches("_aspect", self.health_aspects.keys())[
                0
            ]
            aspect = self.health_aspects[_aspect]

        substance_scoring = aspect["substance"]
        if n != 0:
            if n > len(substance_scoring):
                n = len(substance_scoring)
            substance_scoring = aspect["substance"][:n]

        for substance in substance_scoring:
            if substance[1] in exclude:
                continue
            if substance[1] not in substance_ids:
                substance_list.append((substance[0], substance[1], _aspect))
                substance_ids[substance[1]] = 1

        for alias in aspect["alias"]:
            n = _n
            _substance_scoring = self.health_aspects[alias]["substance"]
            if n != 0:
                if n > len(_substance_scoring):
                    n = len(_substance_scoring)
                _substance_scoring = self.health_aspects[alias]["substance"][:n]

            for substance in _substance_scoring:
                if substance[1] in exclude:
                    continue
                if substance[1] not in substance_ids:
                    substance_list.append((substance[0], substance[1], alias))
                    substance_ids[substance[1]] = 1

        n = _n
        if len(substance_list) > n and n != 0:
            substance_list = substance_list[:n]
        substance_list = sorted(substance_list, key=lambda tup: tup[0], reverse=True)

        return substance_list

    # Load substance meta and return as DataFrame
    def get_substances_df(self, _aspect, n):
        substance_list = self.get_substances(_aspect, n)
        substance_data = {"substance": [], "score": [], "health_aspect": []}
        for substance in substance_list:
            substance_data["score"].append(substance[0])
            substance_data["substance"].append(substance[1])
            substance_data["health_aspect"].append(substance[2])

        datatypes = {"substance": str, "score": int, "health_aspect": str}

        df = pd.DataFrame(data=substance_data)
        df = df.astype(datatypes)

        return df


class HealthseaPipe:

    # Get Clauses and their predictions
    def get_clauses(self, doc):
        clauses = []
        for clause in doc._.clauses:
            words = []
            spaces = []
            clause_slice = doc[clause["split_indices"][0] : clause["split_indices"][1]]

            if clause["has_ent"]:
                for token in clause_slice:
                    if token.i == clause["ent_indices"][0]:
                        words.append(
                            clause["blinder"].replace(">", "").replace("<", "")
                        )
                        spaces.append(True)
                    elif token.i not in range(
                        clause["ent_indices"][0], clause["ent_indices"][1]
                    ):
                        words.append(token.text)
                        spaces.append(token.whitespace_)
                clauses.append(Doc(doc.vocab, words=words, spaces=spaces))

            else:
                for token in clause_slice:
                    words.append(token.text)
                    spaces.append(token.whitespace_)
                clauses.append(Doc(doc.vocab, words=words, spaces=spaces))

        return clauses
