from altair.vegalite.v4.api import condition
import pandas as pd
import difflib


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
