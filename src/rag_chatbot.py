import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def load_resources():
    """Load the vectorstore and embedding model."""
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection("trials")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return collection, embedder

def retrieve(question, collection, embedder, n_results=5):
    """Embed the question and find the most similar trials."""
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=n_results
    )
    return results["documents"][0]  # list of matching trial texts

def generate_answer(question, context_docs):
    """Pass retrieved context to Claude and get a grounded answer."""
    context = "\n\n".join([f"Trial {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])

    system_prompt = """You are TrialIQ, an intelligent clinical trial assistant. 
You answer questions about clinical trials using only the provided trial data.
Be concise, accurate, and cite the trial IDs (NCT numbers) when referencing specific trials.
If the context doesn't contain enough information to answer, say so clearly."""

    user_message = f"""Here is relevant clinical trial data:

{context}

Question: {question}

Answer based only on the trial data provided above."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    return response.content[0].text

def chat():
    """Simple terminal chat loop."""
    print("Loading TrialIQ RAG system...")
    collection, embedder = load_resources()
    print(f"Vectorstore loaded — {collection.count()} trials available")
    print("\nTrialIQ Clinical Trial Assistant")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            break
        if not question:
            continue

        print("Searching trials...")
        context_docs = retrieve(question, collection, embedder)

        print("Generating answer...\n")
        answer = generate_answer(question, context_docs)

        print(f"TrialIQ: {answer}\n")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    chat()