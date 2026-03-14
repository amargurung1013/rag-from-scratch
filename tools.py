import os
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq

load_dotenv(override=True)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """You are a helpful assistant.
If the search requires current or real-time information, respond with exactly:
SEARCH: <search query>
Otherwise, just answer directly."""

question = input("Ask a question: ")

response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

reply = response.choices[0].message.content

if reply.startswith("SEARCH:"):
    tavily_api=os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api)
    result = client.search(question)
    context = result["results"][0]["content"]

    response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": f"""Answer this question using the search results below.

            Question: {question}
            Search Results: {context}"""
        }
    ])

    print("Answer:", response.choices[0].message.content)
else:
    print("Answer:", response.choices[0].message.content)