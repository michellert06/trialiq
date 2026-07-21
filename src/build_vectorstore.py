import pandas as pd
from sqlalchemy import create_engine
import chromadb
from sentence_transformers import SentenceTransformer
import os

DATABASE_URL = "postgresql://localhost/trialiq"

def load_trials():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("""
        SELECT nct_id, title, conditions, phase, status, 
               lead_sponsor, sponsor_class, enrollment_count,
               start_date, completion_date
        FROM trials
        WHERE title IS NOT NULL
    """, engine)
    return df

def build_vectorstore():
    print("Loading trials from database...")
    df = load_trials()
    print(f"Loaded {len(df)} trials")

    # Build a rich text representation of each trial
    # This is what gets embedded and searched
    def make_document(row):
        return (
            f"Trial ID: {row['nct_id']}. "
            f"Title: {row['title']}. "
            f"Conditions: {row['conditions']}. "
            f"Phase: {row['phase']}. "
            f"Status: {row['status']}. "
            f"Lead Sponsor: {row['lead_sponsor']}. "
            f"Sponsor Type: {row['sponsor_class']}. "
            f"Enrollment: {row['enrollment_count']}. "
            f"Start Date: {row['start_date']}. "
            f"Completion Date: {row['completion_date']}."
        )

    documents = df.apply(make_document, axis=1).tolist()
    ids = df["nct_id"].tolist()

    # Load embedding model
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed all documents
    print("Embedding trials (this may take a few minutes)...")
    embeddings = embedder.encode(documents, show_progress_bar=True)

    # Store in ChromaDB
    print("Building ChromaDB vectorstore...")
    client = chromadb.PersistentClient(path="data/chroma")
    
    # Delete existing collection if it exists (clean rebuild)
    try:
        client.delete_collection("trials")
    except:
        pass

    collection = client.create_collection("trials")
    
    # Add in batches of 500 (ChromaDB limit per call)
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            ids=ids[i:i+batch_size]
        )
        print(f"  Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

    print(f"\nVectorstore built — {collection.count()} trials indexed")

if __name__ == "__main__":
    build_vectorstore()