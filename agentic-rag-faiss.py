import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import PyPDF2
from bs4 import BeautifulSoup

# Langchain and FAISS imports for semantic retrieval
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

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

def build_faiss_index(documents, embedding_model):
    """
    Build a FAISS vector store from the provided documents using the embedding model.
    This index performs semantic search.
    """
    texts = [doc.page_content for doc in documents]
    # FAISS.from_texts is a convenience method that creates Document objects internally,
    # but we already have them if you need the metadata preserved, so you might call:
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=[doc.metadata for doc in documents])
    return vectorstore

def simple_rerank(query, candidate_docs):
    """
    (Optional) Rerank the candidate documents using a local LLM prompt.
    Here we call the Ollama model to simulate a reranking.
    In a production system, you might integrate a dedicated reranking chain from Langchain or LlamaIndex.
    """
    if not candidate_docs:
        return []
    # Combine candidate texts with metadata
    docs_text = "\n\n".join([f"{i+1}. {doc.page_content[:200]}..." for i, doc in enumerate(candidate_docs)])
    prompt = (f"Given the query: '{query}', here are summaries of candidate documents:\n\n"
              f"{docs_text}\n\n"
              "Reorder the candidate documents by relevance (most relevant first) and return the ordered indices as a comma-separated list. "
              "If uncertain, keep the original order.")
    ranked = query_ollama(prompt)
    try:
        # Parse the comma-separated indices, e.g., "2,1,3"
        order = [int(i.strip()) for i in ranked.split(",") if i.strip().isdigit()]
        # Ensure indices are within candidate_docs range
        ordered_docs = [candidate_docs[i-1] for i in order if 0 < i <= len(candidate_docs)]
        if ordered_docs:
            return ordered_docs
    except Exception as e:
        st.warning("Could not rerank documents, proceeding with candidate order.")
    return candidate_docs

def query_ollama(prompt):
    """
    Query the local Ollama model with the given prompt.
    Ensure you update the endpoint, parameters, and headers as per your setup.
    """
    ollama_url = "http://localhost:11434/api/generate"  # Adjust as needed
    payload = {"prompt": prompt}
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        # Adjust parsing depending on your model's output format
        return result.get("response", "No response returned.")
    except Exception as e:
        st.error(f"Error calling Ollama model: {e}")
        return ""

# ---------- Streamlit Application ----------

st.title("Upgraded Agentic RAG Application with FAISS and Semantic Reranking")
st.write("""
This application accepts PDF files and an Excel file with article URLs, builds a semantic document index using FAISS 
(via Langchain), performs semantic retrieval and optional reranking, and finally uses a locally hosted Ollama model for generation.
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
    # Instantiate a HuggingFace embedding model (change model_name as needed; ensure it's installed locally)
    with st.spinner("Initializing embedding model and building FAISS index..."):
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        vectorstore = build_faiss_index(documents, embedding_model)
else:
    st.warning("No documents loaded. Please upload some PDFs or an Excel file with article URLs.")

# --- Query Section ---

user_query = st.text_input("Enter your query:")
if st.button("Get Answer", key="01") and user_query and documents:
    st.info("Performing semantic retrieval using FAISS...")
    # Retrieve top 5 candidate documents using FAISS semantic similarity search
    candidate_docs = vectorstore.similarity_search(user_query, k=5)
    
    st.write("### Candidate Documents (Pre-Reranking):")
    for i, doc in enumerate(candidate_docs):
        st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**: {doc.page_content[:200]}...")
    
    # Optional reranking step using the Ollama model (this is a simple example)
    st.info("Reranking retrieved documents for improved relevance...")
    reranked_docs = simple_rerank(user_query, candidate_docs)
    
    st.write("### Reranked Documents:")
    for i, doc in enumerate(reranked_docs):
        st.write(f"**Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')})**: {doc.page_content[:200]}...")
    
    # Combine the reranked document contents as context for the answer
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    
    # Compose the prompt for Ollama model
    prompt = f"""You are a helpful assistant. Use the following context to answer the query.

Context:
{context}

Query:
{user_query}

Answer:"""
    
    st.info("Querying the local Ollama model...")
    answer = query_ollama(prompt)
    st.write("### Answer")
    st.write(answer)
elif st.button("Get Answer", key="02"):
    st.warning("Please ensure you have loaded documents and entered a query.")