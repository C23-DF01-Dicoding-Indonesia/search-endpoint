from flask import Flask, request, jsonify
import requests
import pinecone
import json
import os
from transformers import AutoTokenizer
import json
import numpy as np

app = Flask(__name__)
@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    print(query)
    result = get_search_result(query)
    return jsonify(data = result)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENVIRONMENT')

def pinecone_init():

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )


def get_index(index_name = 'semantic-search'):
    return pinecone.Index(index_name)



def get_embedding(
    text: str
) -> list[float]:
    """
    Embed a single text with embedding-endpoint
    # run the docker of embedding-endpoint first
    """
    tokenizer = AutoTokenizer.from_pretrained("carlesoctav/multi-qa-en-id-mMiniLMv2-L6-H384")
    batch = tokenizer(text)
    batch = dict(batch)
    batch = [batch]
    input_data = {"instances": batch}
    r = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(input_data))
    result = json.loads(r.text)["predictions"][0]["last_hidden_state"][0]
    return result




def semantic_search(query: str, index_name: str = "semantic-search", top_k: int = 10) -> dict:
    """
    Semantic search of a query in a Pinecone index.
    """
    index = get_index(index_name)
    xq = get_embedding(query)
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc


def get_search_result(query: str) -> list:
    index_name: str = "INSERT_INDEX_NAME_HERE"
    top_k=20
    """
    Get search result of a query in a Pinecone index.
    """
    xc = semantic_search(query, index_name, top_k)
    result = []
    for i in xc["matches"]:
        data = {
            "id": i["id"],
            "title": i["metadata"]["title"],
            "type": i["metadata"]["type"],
            "content": i["metadata"]["content"],
            "score": i["score"]
        }
        result.append(data)
    return result


if __name__ == "__main__":
    pinecone_init()
    app.run(host='0.0.0.0', port=8080)
