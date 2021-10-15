try:
    import prodigy
except ModuleNotFoundError:
    raise Exception(
        "Prodigy is not installed. You can skip the parse_ner command, data for training is already provided in the assets folder."
    )
