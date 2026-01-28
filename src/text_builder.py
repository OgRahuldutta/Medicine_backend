def build_corpus(df):
    corpus = []

    for _, row in df.iterrows():
        text = f"""
        Medicine Name: {row['Medicine Name']}
        Composition: {row['Composition']}
        Uses: {row['Uses']}
        Side Effects: {row['Side_effects']}
        Manufacturer: {row['Manufacturer']}
        """
        corpus.append(text.strip())

    return corpus
