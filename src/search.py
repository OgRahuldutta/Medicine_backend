def search(query, model, index, df, top_k=5):
    q_emb = model.encode([query])
    _, indices = index.search(q_emb, top_k)

    results = []
    for i in indices[0]:
        results.append(df.iloc[i].to_dict())

    return results
