import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import PyPDF2
from bs4 import BeautifulSoup

# Langchain and FAISS imports for semantic retrieval
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List

# Import cross-encoder for reranking
from sentence_transformers import CrossEncoder

import numpy as np

# ---------- Helper Functions ----------

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file stream."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def scrape_article_from_url(url):
    """Scrape text content from a URL using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Remove script and style elements then join visible text
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return ""

def create_documents(pdf_texts, article_texts):
    """Wrap raw texts in Langchain Document objects with basic metadata."""
    docs = []
    for i, text in enumerate(pdf_texts):
        docs.append(Document(page_content=text, metadata={"source": f"PDF_{i}"}))
    for i, text in enumerate(article_texts):
        docs.append(Document(page_content=text, metadata={"source": f"Article_{i}"}))
    return docs

# ---------- Ollama Embeddings Class ----------

class OllamaEmbeddings(Embeddings):
    """
    Custom embeddings class that calls a local Ollama embeddings endpoint.
    Adjust the endpoint URL and the payload structure as per your local setup.
    """
    def __init__(self, model_name: str = "ollama-embedding-model"):
        self.model_name = model_name
        # Adjust this URL if your Ollama embeddings endpoint differs.
        self.url = "http://localhost:11434/api/embed"  

    def embed_query(self, text: str) -> List[float]:
        payload = {"text": text, "model": self.model_name}
        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("embedding", [])
        except Exception as e:
            st.error(f"Error obtaining query embedding from Ollama: {e}")
            return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            emb = self.embed_query(text)
            if not emb:
                st.warning("Received an empty embedding for some document.")
            embeddings.append(emb)
        return embeddings

# ---------- Build FAISS Index ----------

def build_faiss_index(documents, embedding_model):
    """
    Build a FAISS vector store using the provided documents and an embedding model.
    This uses Langchain's FAISS interface.
    """
    texts = [doc.page_content for doc in documents]
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=[doc.metadata for doc in documents])
    return vectorstore

# ---------- Reranking with a Cross-Encoder ----------

def rerank_with_crossencoder(query, candidate_docs):
    """
    Rerank candidate documents using an open-source cross-encoder.
    This function uses SentenceTransformers' CrossEncoder to score (query, document) pairs.
    """
    if not candidate_docs:
        return []
    
    # Initialize the cross-encoder model.
    # You might consider loading the model only once in a production setup.
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [(query, doc.page_content) for doc in candidate_docs]
    scores = cross_encoder.predict(pairs)
    
    # Zip documents with their scores, sort descending, and extract sorted docs.
    ranked = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
    ranked_docs = [doc for score, doc in ranked]
    return ranked_docs

# ---------- Query Ollama for Generation ----------

def query_ollama(prompt):
    """
    Query the local Ollama model with the given prompt.
    Update the URL and payload as per your model's API.
    """
    ollama_url = "http://localhost:11434/api/generate"  # Adjust as needed
    payload = {"prompt": prompt}
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response returned.")
    except Exception as e:
        st.error(f"Error calling Ollama model: {e}")
        return ""

# ---------- Streamlit Application ----------

st.title("Upgraded Agentic RAG with Ollama Embeddings & Open-Source Reranking")
st.write("""
This app accepts PDFs and an Excel file with article URLs, embeds their text using Ollama embeddings, 
builds a FAISS vector index, and uses an open-source cross-encoder for reranking before generating 
an answer with a locally hosted Ollama model.
""")

# --- File Upload Section ---

# Upload PDF Files
uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
pdf_texts = []
if uploaded_pdfs:
    st.info("Processing PDF files...")
    for pdf_file in uploaded_pdfs:
        bytes_data = pdf_file.read()
        text = extract_text_from_pdf(bytes_data)
        if text:
            pdf_texts.append(text)
    st.success("PDF files processed.")

# Upload Excel File with article URLs
uploaded_excel = st.file_uploader("Upload Excel file containing article URLs", type=["xlsx"])
article_texts = []
if uploaded_excel:
    try:
        df_links = pd.read_excel(uploaded_excel)
        if "URL" not in df_links.columns:
            st.error("Excel file must contain a column named 'URL'.")
        else:
            st.info("Scraping articles from provided URLs...")
            urls = df_links["URL"].dropna().unique().tolist()
            for url in urls:
                text = scrape_article_from_url(url)
                if text:
                    article_texts.append(text)
            st.success("Article scraping completed.")
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")

# --- Build Document Index ---
documents = create_documents(pdf_texts, article_texts)
if documents:
    st.write(f"Total documents loaded: {len(documents)}")
    with st.spinner("Initializing Ollama embedding model and building FAISS index..."):
        embedding_model = OllamaEmbeddings(model_name="ollama-embedding-model")
        vectorstore = build_faiss_index(documents, embedding_model)
else:
    st.warning("No documents loaded. Please upload some PDFs or an Excel file with article URLs.")

# --- Query Section ---

user_query = st.text_input("Enter your query:")
if st.button("Get Answer") and user_query and documents:
    st.info("Performing semantic retrieval using FAISS...")
    # Retrieve top 5 candidate documents via semantic search
    candidate_docs = vectorstore.similarity_search(user_query, k=5)
    
    st.write("### Candidate Documents (Pre-Reranking):")
    for i, doc in enumerate(candidate_docs):
        st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**: {doc.page_content[:200]}...")
    
    # Rerank the candidates using the cross-encoder
    st.info("Reranking candidate documents using the cross-encoder...")
    reranked_docs = rerank_with_crossencoder(user_query, candidate_docs)
    
    st.write("### Reranked Documents:")
    for i, doc in enumerate(reranked_docs):
        st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**: {doc.page_content[:200]}...")
    
    # Combine the reranked document contents as context for the answer
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Compose the prompt for the Ollama generation model
    prompt = f"""You are a helpful assistant. Use the following context to answer the query.

Context:
{context}

Query:
{user_query}

Answer:"""
    
    st.info("Querying the local Ollama model for the final answer...")
    answer = query_ollama(prompt)
    st.write("### Answer")
    st.write(answer)
elif st.button("Get Answer"):
    st.warning("Please ensure you have loaded documents and entered a query.")