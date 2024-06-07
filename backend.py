import argparse
from langchain.vectorstores.chroma import Chroma


from get_embedding_function import get_embedding_function

from fastapi import FastAPI
CHROMA_PATH = "chroma"
app = FastAPI()


@app.get("/query/{query_text}")
async def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB. Peut etre que le k peut être modifié
    results = db.similarity_search_with_score(query_text, k=2)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    score = [_score for doc, _score in results]
    formatted_response = f"Response: {context_text}\nSources: {sources}\nScore: {score}"
    return formatted_response
async def root():
    return {"message": "Vous n'avez pas passez la requete en param"}
