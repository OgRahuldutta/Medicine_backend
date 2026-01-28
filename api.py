import os
import pickle
import faiss
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from src.search import search

# 🔒 Force CPU only (important for free tiers)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = FastAPI()

# 🌐 Allow GitHub Pages / public frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📦 Load lightweight data at startup (OK for free tier)
df = pickle.load(open("medicine_df.pkl", "rb"))
index = faiss.read_index("medicine.index")

# 🧠 Lazy-loaded model (CRITICAL FIX)
model = None

def get_model():
    global model
    if model is None:
        # ✅ Smaller + memory-efficient model
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2"
        )
    return model

# ✅ Health check
@app.get("/")
def health():
    return {"status": "Medical AI backend running"}

# 🔍 Search endpoint
@app.get("/search")
def search_medicine(q: str = Query(..., min_length=2)):
    model_instance = get_model()
    return search(q, model_instance, index, df)
