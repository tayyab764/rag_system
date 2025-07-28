from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import shutil
from PyPDF2 import PdfReader

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

app = FastAPI()

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=BASE_DIR)

# ✅ Your Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCPb4ufYveBN3R3r18EJEzMaUxcDRXN6FY"

# Global vectorstore
vectorstore = None

# Load PDF text
def load_pdf_text(file_path):
    pdf = PdfReader(file_path)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Embed and store in FAISS
def embed_and_store_text(text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])
    return FAISS.from_documents(docs, embeddings)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("e.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = load_pdf_text(file_path)
    global vectorstore
    vectorstore = embed_and_store_text(text)
    return {"message": "✅ PDF uploaded and indexed successfully."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vectorstore
    if not vectorstore:
        return {"error": "❌ Please upload a PDF first."}
    docs = vectorstore.similarity_search(question)
    
    # ✅ Use a working model name
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return {"answer": answer}
