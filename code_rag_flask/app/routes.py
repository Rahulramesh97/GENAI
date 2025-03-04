import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from app.utils import extract_text_from_pdfs, chunk_text, load_or_create_embeddings, create_qa_chain, answer_question

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def configure_routes(app: Flask):
    
    @app.route("/", methods=["GET"])
    def home():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload_pdfs():
        """Handles PDF uploads, processes text, and creates embeddings."""
        if "pdfs" not in request.files:
            return jsonify({"error": "No files uploaded"}), 400

        pdf_files = request.files.getlist("pdfs")

        # Save files locally (optional)
        for pdf in pdf_files:
            filename = secure_filename(pdf.filename)
            pdf.save(os.path.join(UPLOAD_FOLDER, filename))

        raw_text = extract_text_from_pdfs(pdf_files)
        text_chunks = chunk_text(raw_text)
        db = load_or_create_embeddings(text_chunks)

        return jsonify({"message": "PDFs processed and embeddings created!"})

    @app.route("/ask", methods=["POST"])
    def ask_question():
        """Handles questions and retrieves answers using RAG."""
        data = request.json
        question = data.get("question")
        temperature = float(data.get("temperature", 0.3))

        if not question:
            return jsonify({"error": "No question provided"}), 400

        db = load_or_create_embeddings([])
        qa_chain = create_qa_chain(temperature)
        answer = answer_question(db, question, qa_chain)

        return jsonify({"answer": answer})
