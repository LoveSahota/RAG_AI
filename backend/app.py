import os
import shutil

from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import SessionLocal, engine
import models
from pdf_utils import extract_text_from_pdf, chunk_text
from rag_pipeline import retrieve_relevant_chunks, build_rag_prompt, ask_ai

models.Base.metadata.create_all(bind=engine)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    user_id: int
    title: str


class MessageRequest(BaseModel):
    chat_id: int
    message: str


@app.get("/")
def home():
    return {"message": "AnswerAI backend running"}


@app.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == data.email).first()
    if existing:
        return {"error": "User already exists"}

    user = models.User(name=data.name, email=data.email, password=data.password)
    db.add(user)
    db.commit()
    db.refresh(user)

    return {"id": user.id, "name": user.name, "email": user.email}


@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(models.User).filter(models.User.email == data.email).first()

        if not user:
            return {"error": "User not found"}

        if user.password != data.password:
            return {"error": "Invalid password"}

        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        }

    except Exception as e:
        print("LOGIN ERROR:", str(e))  # 👈 VERY IMPORTANT
        return {"error": str(e)}


@app.post("/create_chat")
def create_chat(data: ChatRequest, db: Session = Depends(get_db)):
    chat = models.Chat(user_id=data.user_id, title=data.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return {"id": chat.id, "title": chat.title, "user_id": chat.user_id}


@app.get("/get_chats/{user_id}")
def get_chats(user_id: int, db: Session = Depends(get_db)):
    chats = db.query(models.Chat).filter(models.Chat.user_id == user_id).all()
    return [{"id": c.id, "title": c.title, "user_id": c.user_id} for c in chats]


@app.get("/get_messages/{chat_id}")
def get_messages(chat_id: int, db: Session = Depends(get_db)):
    messages = db.query(models.Message).filter(models.Message.chat_id == chat_id).all()
    return [
        {"id": m.id, "role": m.role, "content": m.content, "chat_id": m.chat_id}
        for m in messages
    ]


@app.post("/upload_pdf")
def upload_pdf(
    chat_id: int = Form(...),
    user_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    saved_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = extract_text_from_pdf(saved_path)

    if not extracted_text.strip():
        return {"error": "No readable text found in PDF."}

    document = models.Document(
        user_id=user_id,
        chat_id=chat_id,
        filename=file.filename
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    chunks = chunk_text(extracted_text, chunk_size=800, overlap=150)

    for i, chunk in enumerate(chunks):
        db_chunk = models.DocumentChunk(
            document_id=document.id,
            chat_id=chat_id,
            chunk_index=i,
            content=chunk
        )
        db.add(db_chunk)

    db.commit()

    return {
        "message": "PDF uploaded and processed successfully.",
        "document_id": document.id,
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


@app.post("/send_message")
def send_message(data: MessageRequest, db: Session = Depends(get_db)):
    user_msg = models.Message(
        chat_id=data.chat_id,
        role="user",
        content=data.message
    )
    db.add(user_msg)
    db.commit()

    history_rows = db.query(models.Message).filter(
        models.Message.chat_id == data.chat_id
    ).all()

    history = [{"role": m.role, "content": m.content} for m in history_rows]

    chunk_rows = db.query(models.DocumentChunk).filter(
    models.DocumentChunk.chat_id == data.chat_id
    ).order_by(models.DocumentChunk.chunk_index).all()

    print("Chat ID:", data.chat_id)
    print("Chunks fetched:", len(chunk_rows))

    chunks = [{"content": c.content} for c in chunk_rows]

    if not chunks:
        ai_response = "No PDF has been uploaded for this chat yet."
    else:
        relevant_chunks = retrieve_relevant_chunks(data.message, chunks, top_k=4)
        useful_chunks = [c for c in relevant_chunks if c.get("score", 0) > 0]

        if not useful_chunks:
            ai_response = "The answer was not found in the uploaded document."
        else:
            prompt = build_rag_prompt(
                question=data.message,
                relevant_chunks=useful_chunks,
                history=history
            )
            ai_response = ask_ai(prompt)

    ai_msg = models.Message(
        chat_id=data.chat_id,
        role="assistant",
        content=ai_response
    )

    db.add(ai_msg)
    db.commit()

    # ✅ AUTO RENAME CHAT (only first question)
    chat = db.query(models.Chat).filter(models.Chat.id == data.chat_id).first()

    if chat and chat.title == "New Chat":
        
        # Get document name
        doc = db.query(models.Document).filter(
            models.Document.chat_id == data.chat_id
        ).first()

        doc_name = doc.filename if doc else "Chat"

        # Take first few words of question
        short_question = " ".join(data.message.split()[:5])

        # Create title
        new_title = f"{doc_name} - {short_question}"

        chat.title = new_title[:50]  # limit length
        db.commit()

    return {"response": ai_response}


from fastapi import Depends
from sqlalchemy.orm import Session
import models
from database import SessionLocal

@app.delete("/delete_chats/{user_id}")
def delete_chats(user_id: int, db: Session = Depends(get_db)):

    # 🔥 Get all chats of user
    chats = db.query(models.Chat).filter(models.Chat.user_id == user_id).all()

    for chat in chats:
        # Delete messages of this chat
        db.query(models.Message).filter(models.Message.chat_id == chat.id).delete()

        # Delete document chunks
        db.query(models.DocumentChunk).filter(models.DocumentChunk.chat_id == chat.id).delete()

        # Delete documents
        db.query(models.Document).filter(models.Document.chat_id == chat.id).delete()

        # Delete chat
        db.delete(chat)

    db.commit()

    return {"status": "All chats deleted successfully"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

from pydantic import BaseModel

class RenameChatRequest(BaseModel):
    chat_id: int
    new_title: str


@app.put("/rename_chat")
def rename_chat(data: RenameChatRequest, db: Session = Depends(get_db)):
    chat = db.query(models.Chat).filter(models.Chat.id == data.chat_id).first()

    if not chat:
        return {"error": "Chat not found"}

    chat.title = data.new_title
    db.commit()

    return {"message": "Chat renamed successfully"}

from fastapi import UploadFile, File, Form
import os

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


from fastapi.staticfiles import StaticFiles

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")