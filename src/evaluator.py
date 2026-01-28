import numpy as np

def evaluate_search(model, index, corpus, test_queries, expected_keywords, top_k=5):
    correct = 0

    for query, keyword in zip(test_queries, expected_keywords):
        q_emb = model.encode([query])
        _, results = index.search(q_emb, top_k)

        retrieved_texts = [corpus[i] for i in results[0]]

        if any(keyword.lower() in text.lower() for text in retrieved_texts):
            correct += 1

    accuracy = correct / len(test_queries)
    return accuracy
