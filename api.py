from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

app = Flask(__name__)

@app.route("/query", methods=["GET"])
def query():
    vector_query = request.args.get("vector")
    if not vector_query:
        return jsonify({"error": "No vector query provided"}), 400

    # Assuming the API key is set correctly in the environment variables
    pc_api_key = os.getenv("PINECONE_API_KEY")
    if not pc_api_key:
        return jsonify({"error": "Pinecone API key not found"}), 500

    # Initialize the embeddings model and vector store
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = PineconeVectorStore(index_name="vectortest", embedding=embeddings_model)

    # Use the vector store for operations
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    documents = retriever.get_relevant_documents(vector_query)
    if documents:
        return jsonify(documents[0].page_content)
    else:
        return jsonify({"error": "No documents found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
