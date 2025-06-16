from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

app = Flask(__name__)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("flower_faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

generator = pipeline(
    model="google/flan-t5-base",
    task="text2text-generation",
    max_length=256,
    do_sample=True,
    temperature=0.5
)
llm = HuggingFacePipeline(pipeline=generator)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        query = request.form["question"]
        answer = qa_chain.invoke(query)
        result = {"result": answer['result']}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
