from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from src.search import search

app = FastAPI()

# CORS (allow GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load AI once
df = pickle.load(open("medicine_df.pkl", "rb"))
index = faiss.read_index("medicine.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def health():
    return {"status": "Medical AI backend running"}

@app.get("/search")
def search_medicine(q: str = Query(...)):
    return search(q, model, index, df)
