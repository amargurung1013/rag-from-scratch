import requests
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

url = "https://en.wikipedia.org/wiki/The_Eras_Tour"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, 'html.parser')
paragraphs = soup.find_all('p')
data = "\n".join([para.get_text() for para in paragraphs])

chunks = data.split(".")
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection("wikipedia_eras_tour")
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)

question = "When did the Eras Tour start?"
question_embedding = model.encode([question]).tolist()
results = collection.query(
    query_embeddings = question_embedding,
    n_results=5
)
print("Question:", question)
print("Relevant Chunks:", results['documents'])

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