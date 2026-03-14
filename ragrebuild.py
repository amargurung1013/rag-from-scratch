with open("data.txt", "r") as f:
	data = f.read()

print(data)

chunk = data.split(".")
chunk = [chunk.strip() for chunk in chunk if  chunk.strip()]
print(chunk)

import os

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunk)

chunks = chunk

import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("planets")

collection.add(
	documents = chunks,
	embeddings = embeddings.tolist(),
	ids = [str(i) for i in range(len(chunks))]
)

#retrive relevant chunks
question = "What is Earth?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
	query_embeddings = question_embedding,
	n_results = 2
)

from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

context = results['documents'][0]
context_text = "\n".join(context)

message = groq_client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [
        {
            "role": "user",
            "content": f"""Answer the question using only the context below.

            Context: {context_text}     
Question: {question}"""
        }
    ]
)
print(message.choices[0].message.content)