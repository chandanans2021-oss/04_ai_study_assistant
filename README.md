# 04 AI Study Assistant

Workflow:
PDF Upload
-> Extract Text
-> Generate Embeddings
-> Vector Database
-> Research Agent
-> Analysis Agent
-> Review Agent
-> Final Answer

The assignment suggests Pinecone. This starter uses FAISS locally so it can run without a cloud account. You can later replace FAISS with Pinecone.

Run:
1. `ollama pull llama3.2`
2. `ollama pull nomic-embed-text`
3. `pip install -r requirements.txt`
4. `python app.py`
