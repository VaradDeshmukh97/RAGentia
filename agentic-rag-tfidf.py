import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import PyPDF2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------- Helper Functions ----------

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file stream."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def scrape_article_from_url(url):
    """Scrape text content from a URL using BeautifulSoup."""
    try:
        # Fetch content from the URL
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

def build_corpus(pdf_texts, article_texts):
    """Combine all extracted texts into a corpus (list of documents)."""
    corpus = []
    corpus.extend(pdf_texts)
    corpus.extend(article_texts)
    return corpus

def vectorize_corpus(corpus):
    """Vectorize the corpus using TF‑IDF."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def retrieve_relevant_context(query, vectorizer, tfidf_matrix, corpus, top_n=3):
    """Retrieve the top N relevant documents from the corpus for the query."""
    query_vec = vectorizer.transform([query])
    cosine_similarities = (tfidf_matrix * query_vec.T).toarray().ravel()
    if np.count_nonzero(cosine_similarities) == 0:
        return ""
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    retrieved_chunks = [corpus[idx] for idx in top_indices if cosine_similarities[idx] > 0]
    return "\n\n".join(retrieved_chunks)

def query_ollama(prompt):
    """
    Query the local Ollama model with the given prompt.
    Make sure you update the endpoint, parameters, and headers according to your Ollama setup.
    """
    ollama_url = "http://localhost:11434/api/generate"  # update this URL as needed
    payload = {"prompt": prompt}
    try:
        response = requests.post(ollama_url, json=payload, timeout=30)
        response.raise_for_status()
        # Adjust parsing depending on how Ollama returns responses
        result = response.json()
        return result.get("response", "No response returned.")
    except Exception as e:
        st.error(f"Error calling Ollama model: {e}")
        return ""

# ---------- Streamlit Application ----------

st.title("Agentic RAG Application with Ollama Integration")
st.write("""
This app lets you upload PDF files and an Excel file with article links. It extracts text from PDFs, 
scrapes the articles for context, builds a text-retrieval index, and finally uses a locally hosted Ollama 
model for answering your queries.
""")

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

# Upload Excel File containing article URLs
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

# Build the corpus from both PDFs and scraped articles
if pdf_texts or article_texts:
    corpus = build_corpus(pdf_texts, article_texts)
    st.write(f"Total documents loaded for context: {len(corpus)}")
    # Build or update the vectorization index
    vectorizer, tfidf_matrix = vectorize_corpus(corpus)
else:
    st.warning("No documents loaded. Please upload PDF files and/or an Excel file with article URLs.")

# Query Input
user_query = st.text_input("Enter your query:")
if st.button("Get Answer") and user_query and (pdf_texts or article_texts):
    st.info("Retrieving relevant context...")
    context = retrieve_relevant_context(user_query, vectorizer, tfidf_matrix, corpus)
    
    # Prepare prompt for the Ollama model: you can customize the prompt format as needed
    prompt = f"""You are a helpful assistant. Use the following context to answer the query.

Context:
{context}

Query:
{user_query}

Answer:"""
    
    st.info("Querying the Ollama model...")
    # Call the local Ollama model with the prompt (adjust as needed)
    answer = query_ollama(prompt)
    st.write("### Answer")
    st.write(answer)
elif st.button("Get Answer"):
    st.warning("Please enter a query and ensure documents are loaded.")