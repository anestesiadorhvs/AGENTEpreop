from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, MarianMTModel, MarianTokenizer
import torch
import PyPDF2
import os

app = Flask(__name__)

# Cargar modelos
tokenizer_qa = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model_qa = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Función para extraer texto de PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Función para traducir texto
def translate_text(text, src_lang="en", tgt_lang="es"):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# Función para obtener respuestas
def get_answer(question, text):
    # Traducir la pregunta al inglés
    translated_question = translate_text(question, src_lang="es", tgt_lang="en")
    
    # Preprocesar y obtener la respuesta en inglés
    inputs = tokenizer_qa(translated_question, text, return_tensors="pt", truncation=True, padding=True)
    outputs = model_qa(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer_qa.convert_tokens_to_string(tokenizer_qa.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    
    # Traducir la respuesta al español
    translated_answer = translate_text(answer, src_lang="en", tgt_lang="es")
    return translated_answer

# Ruta para hacer preguntas
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    pdf_path = data.get("pdf_path")  # Ruta al archivo PDF
    text = extract_text_from_pdf(pdf_path)
    answer = get_answer(question, text)
    return jsonify({"answer": answer})

# Ruta para subir nuevos textos (PDFs)
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file and file.filename.endswith(".pdf"):
        file_path = os.path.join("textos", file.filename)
        file.save(file_path)
        return jsonify({"status": "success", "file_path": file_path})
    return jsonify({"error": "Invalid file type"}), 400

# Ruta principal para servir la interfaz web
@app.route("/")
def home():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    os.makedirs("textos", exist_ok=True)
    app.run(debug=True)