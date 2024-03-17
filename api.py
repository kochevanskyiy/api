from flask import Flask
import os
from urllib.parse import urlparse, parse_qs, unquote
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route("/query", methods=["GET"])

#def hello_world():
#    return 'Hello, World!'
def query():
    query = request.args.get("vector")
    pc = Pinecone(api_key=os.getenv("8f956e88-1a8f-4698-8c6f-3c94b4eb7fff"))
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key="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")

    vector_store = PineconeVectorStore(
        index_name="vectortest", embedding=embeddings_model
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    return jsonify((retriever.get_relevant_documents(query)[0].page_content))



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)