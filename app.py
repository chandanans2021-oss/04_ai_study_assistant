import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

llm = ChatOllama(model="llama3.2")
vector_store = None

def extract_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def research_agent(question):
    docs = vector_store.similarity_search(question, k=5)
    return "\n\n".join(d.page_content for d in docs)

def analysis_agent(question, evidence):
    prompt = f"""You are the Analysis Agent.
Analyze the retrieved study material and prepare a clear answer.

Question: {question}

Retrieved material:
{evidence}
"""
    return llm.invoke(prompt).content

def review_agent(question, evidence, draft):
    prompt = f"""You are the Review Agent.
Check whether the draft answer is supported by the retrieved study material.
If unsupported claims exist, correct or remove them.

Question: {question}
Evidence: {evidence}
Draft answer: {draft}

Return a final reliable answer only.
"""
    return llm.invoke(prompt).content

@app.route("/", methods=["GET", "POST"])
def index():
    global vector_store
    result = None

    if request.method == "POST":
        action = request.form.get("action")

        if action == "upload":
            file = request.files["file"]
            path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(path)

            text = extract_pdf(path)
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=900, chunk_overlap=120
            ).split_text(text)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.from_texts(chunks, embeddings)
            result = {"message": "Study material uploaded and indexed successfully."}

        elif action == "ask":
            question = request.form["question"]
            if vector_store is None:
                result = {"error": "Upload study material first."}
            else:
                evidence = research_agent(question)
                draft = analysis_agent(question, evidence)
                final = review_agent(question, evidence, draft)
                result = {
                    "question": question,
                    "research": evidence,
                    "analysis": draft,
                    "final": final
                }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
