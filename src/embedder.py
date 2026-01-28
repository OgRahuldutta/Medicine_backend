from sentence_transformers import SentenceTransformer

def load_model(model_name):
    return SentenceTransformer(model_name)

def generate_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)
