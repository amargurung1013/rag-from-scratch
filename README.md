# RAG from Scratch 🔍

Built a Retrieval Augmented Generation (RAG) system from scratch using raw Python — no LangChain, no LlamaIndex, just the core concepts.

## What I Built
- A RAG chatbot that answers questions from a custom text file
- A "Chat with any Wikipedia article" tool

## How it works
1. Load a document and split it into chunks
2. Embed each chunk using SentenceTransformers
3. Store embeddings in ChromaDB
4. Take a user question, embed it, find the most similar chunks
5. Send retrieved chunks + question to an LLM for a grounded answer

## Stack
- Python
- ChromaDB — vector database
- SentenceTransformers — embeddings
- Groq (LLaMA 3.3) — LLM

## Key Lessons
- RAG retrieves by meaning, not keywords
- Chunk quality directly affects answer quality
- RAG prevents hallucination by grounding the LLM in real context

## Part of my AI Agents learning journey
Following a structured path: RAG → Tool Calling → Memory → Agent Loops
```

Copy that into a `README.md` file in your project folder, then:
```
git add README.md
git commit -m "add README"
git push
