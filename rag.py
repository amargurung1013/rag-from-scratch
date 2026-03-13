from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
from groq import Groq


with open("data.txt", "r") as f:
    data = f.read()

print(data)

chunks = data.split(".")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(chunks)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding size: {len(embeddings[0])}")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("planets")

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in Chroma")

question = "Who is the president of the United States?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings = question_embedding,
    n_results=2
)

print("Question:", question)
print("Relevant Chunks:", results['documents'])

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

context = results['documents'][0]
context_text = "\n".join(context)

message = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content":f"""Answer the question using only the context below.

            Context: {context_text}

Question: {question}"""
        }
    ]
)

print("Answer:", message.choices[0].message.content)

