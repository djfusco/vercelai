from flask import Flask, request, jsonify
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

app = Flask(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'default_openai_api_key')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'default_pinecone_api_key')
INDEX_NAME = os.getenv('INDEX_NAME', 'default_index_name')

pinecone = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0.0)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    answer = qa.run(query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
