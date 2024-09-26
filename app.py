import os
import torch
import logging
import requests
import numpy as np
import pdfplumber
import json

from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

app = Flask(__name__)
pdf_text = ""
chunk_texts = []
vector_store = None
chunk_embeddings = None

# Initialize the Hugging Face model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Initialize LangChain components
folder_path = "C:\\Users\\user\\Downloads\fiverr_projects\html5-mistral\\uploads"
cached_llm = Ollama(model="llama3")
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200, length_function=len, is_separator_regex=False)
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Updated Document class with metadata
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_path):
    global pdf_text
    pdf_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")

# Function to chunk text
def chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to create vector embeddings for chunks
def create_embeddings(chunks):
    global chunk_embeddings, model, tokenizer
    
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    
    chunk_embeddings = embeddings
    return embeddings

# Function to search for the most relevant chunks
def search_chunks(query, top_k=1):
    global chunk_embeddings, tokenizer, model
    
    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    
    indices = np.argsort(similarities[0])[::-1][:top_k]
    
    return indices

# Function to generate response using the LLM API
def generate_response(messages):
    try:
        api_key = "API_KEY"  
        api_endpoint = "https://chat-ai.academiccloud.de/v1"
        model_name = "meta-llama-3-70b-instruct"
        
        url = f"{api_endpoint}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {api_key}"
        }
        data = {
            'model': model_name,
            'messages': messages
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

# Route to process and upload the PDF
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global chunk_texts, vector_store, chunk_embeddings 
    if 'pdf' in request.files:
        pdf_file = request.files['pdf']
        pdf_filename = pdf_file.filename
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        pdf_path = os.path.join(uploads_dir, pdf_filename)
        pdf_file.save(pdf_path)
        
        extract_text_from_pdf(pdf_path)
        chunk_texts = chunk_text(pdf_text)
        chunk_embeddings = create_embeddings(chunk_texts)
        
        documents = [Document(chunk, metadata={"source": f"Page {i+1}"}) for i, chunk in enumerate(chunk_texts)]
        
        vector_store = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=folder_path)
        vector_store.persist()
        
        # Show PDF content on the UI
        return render_template('index.html', pdf_text=pdf_text), 200
    else:
        return jsonify({"error": "No PDF uploaded."}), 400

# Route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    global chunk_texts, vector_store, chunk_embeddings
    messages = request.json.get('messages')
    user_message = messages[0]['content']

    if vector_store is not None and chunk_embeddings is not None:
        indices = search_chunks(user_message)
        relevant_chunk = chunk_texts[indices[0]]
        
        # Construct the prompt with the relevant chunk and generate response using LLM
        response = generate_response([{"role": "system", "content": f"Context: {relevant_chunk} \n\n {user_message}"}])
        
        if response:
            return jsonify({"content": response['choices'][0]['message']['content']}), 200
        else:
            return jsonify({"error": "Sorry, I couldn't generate a response."}), 500
    else:
        response = generate_response(messages)
        if response:
            return jsonify({"content": response['choices'][0]['message']['content']}), 200
        else:
            return jsonify({"error": "Sorry, I couldn't generate a response."}), 500

# Route to download PDF key values as JSON
@app.route('/download_pdf_json', methods=['POST'])
def download_pdf_json():
    if 'pdf' in request.files:
        pdf_file = request.files['pdf']
        pdf_filename = pdf_file.filename
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        pdf_path = os.path.join(uploads_dir, pdf_filename)
        pdf_file.save(pdf_path)
        
        extract_text_from_pdf(pdf_path)
        
        json_data = {
            "pdf_text": pdf_text,
            "chunk_texts": chunk_texts
        }
        
        json_filename = os.path.join(uploads_dir, f"{os.path.splitext(pdf_filename)[0]}.json")
        with open(json_filename, 'w') as json_file:
            json.dump(json_data, json_file)
        
        return jsonify({"message": "PDF key values downloaded as JSON.", "json_filename": json_filename}), 200
    else:
        return jsonify({"error": "No PDF uploaded."}), 400

# New route to handle AI queries
@app.route("/ai", methods=["POST"])
def aiPost():
    json_content = request.json
    query = json_content.get("query")

    response = cached_llm.invoke(query)

    response_answer = {"answer": response}
    return response_answer

# New route to handle PDF queries
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    json_content = request.json
    query = json_content.get("query")

    if vector_store is None:
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

@app.route('/')
def index():
    return render_template('index.html', pdf_text=pdf_text)

if __name__ == '__main__':
    # Disable auto-reloading by setting debug to False
    app.run(debug=False)
