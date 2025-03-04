import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from config import Config

def extract_text_from_pdfs(pdf_files):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Splits text into smaller chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def load_or_create_embeddings(text_chunks):
    """Loads existing FAISS index or creates new embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=Config.GOOGLE_API_KEY)

    try:
        db = FAISS.load_local(Config.EMBEDDINGS_DIR, embeddings, allow_dangerous_deserialization=True)
        print("🔹 FAISS index loaded from local storage.")
        return db
    except Exception as e:
        print(f"⚠️ No existing FAISS index found, creating new one: {e}")
        db = FAISS.from_texts(text_chunks, embedding=embeddings)
        db.save_local(Config.EMBEDDINGS_DIR)
        print("✅ FAISS index created and saved.")
        return db


def create_qa_chain(temperature=0.3):
    """Creates a QA chain using Gemini API."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002", temperature=temperature, google_api_key=Config.GOOGLE_API_KEY)
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not contained within the text, say "I don't have the relevant information."
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def answer_question(db, question, chain):
    """Answers a question using RAG approach."""
    search_results = db.similarity_search(question)
    return chain({"input_documents": search_results, "question": question}, return_only_outputs=True)["output_text"]
