import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from RAG import create_chroma_db, load_chroma_db, CHROMA_DIR

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_index():
    return FileResponse("frontend/index.html")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure Chroma vectorstore exists
if not os.path.exists(CHROMA_DIR):
    print("🔄 Creating Chroma vectorstore from documents...")
    create_chroma_db()

vectorstore = load_chroma_db()

# Request model
class QuestionRequest(BaseModel):
    question: str

# DeepSeek LLM
def get_deepseek_llm():
    return ChatOpenAI(
        openai_api_base="https://api.deepseek.com/v1",
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        model_name="deepseek-chat",
        temperature=0.3,
    )

llm = get_deepseek_llm()

# Persistent chat history
chat_history = ChatMessageHistory()

# Pending question state
pending_question = {
    "type": None,  # "RAG" or "GENERAL"
    "question": None,
    "used_docs": set(),
}   

# Prompt setup
system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful chatbot assistant. You answer questions kindly and accurately. "
    "If you don't have specific information, you can answer from your general knowledge. "
    "Try to end your answers with a helpful question or suggestion about related topics."
)
human_message = HumanMessagePromptTemplate.from_template("{question}")

prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="chat_history"),
    human_message
])

# Helper: call LLM
def ask_llm(question, history):
    formatted_prompt = prompt.format_prompt(
        chat_history=history,
        question=question
    ).to_messages()
    return llm.invoke(formatted_prompt).content

# Helper: check if user input is affirmative in natural language
def is_affirmative(text: str) -> bool:
    """
    Use the LLM to determine if the user's message is an affirmative response
    asking for more information.
    """
    check_prompt = f"""
    Does the following text indicate that the user wants more information or agrees to continue?
    Text: "{text}"
    Answer only "Yes" or "No".
    """
    response = llm.invoke([{"role": "user", "content": check_prompt}]).content.strip().lower()
    return response.startswith("yes")

@app.post("/chat")
async def chat(req: QuestionRequest):
    query = req.question.strip()
    
    # Initialize variables
    answer = None
    source_summary = None

    # Step 0: Check if user is confirming a pending question
    if pending_question["type"] and is_affirmative(query):
        if pending_question["type"] == "GENERAL":
            # 🔧 FIX: use user's follow-up query instead of re-asking old question
            answer = ask_llm(query, chat_history.messages)
            source_summary = None
            # Reset pending question
            pending_question["type"] = None
            pending_question["question"] = None
            
        elif pending_question["type"] == "RAG":
            # ✅ Re-search with the follow-up query, not the original
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)
            relevance_threshold = 0.8
            
            # Instead of excluding entire sources, exclude only *exact chunks* already used
            new_docs = [
                doc for doc, score in docs_and_scores
                if score >= relevance_threshold and doc.page_content not in pending_question["used_docs"]
            ]

            if new_docs:
                context_text = "\n\n".join(doc.page_content for doc in new_docs)
                question_with_context = f"Answer using the following documents:\n{context_text}\n\nQuestion: {query}"
                answer = ask_llm(question_with_context, chat_history.messages)

                # Track used chunks instead of whole files
                for doc in new_docs:
                    pending_question["used_docs"].add(doc.page_content)

                source_names = [doc.metadata.get("source", "unknown").replace("FAQ-docs\\", "") for doc in new_docs]
                source_summary = ", ".join(set(source_names))
            else:
                # If nothing new, fall back to LLM general knowledge
                answer = ask_llm(query, chat_history.messages)
                source_summary = None

            # Reset pending question
            pending_question["type"] = None
            pending_question["question"] = None
            pending_question["used_docs"] = set()
    
    else:
        # Step 1 - Retrieve with score filtering
        docs_and_scores = vectorstore.similarity_search_with_score(req.question, k=3)
        relevance_threshold = 0.8
        relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]

        if relevant_docs:
            # Check if the retrieved content is actually relevant using LLM
            context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
            relevance_check = llm.invoke([{
                "role": "user",
                "content": f"Is the following context relevant to answering the question '{req.question}'? "
                        f"Answer only 'Yes' or 'No'.\n\nContext:\n{context_text}"
            }]).content.strip().lower()

            if relevance_check.startswith("yes"):
                # ✅ Use RAG mode
                question_with_context = f"Answer using the following documents:\n{context_text}\n\nQuestion: {req.question}"
                answer = ask_llm(question_with_context, chat_history.messages)
                source_names = [doc.metadata.get("source", "unknown").replace("FAQ-docs\\", "") for doc in relevant_docs]
                source_summary = ", ".join(set(source_names))
                # Set up for potential follow-up
                pending_question["type"] = "RAG"
                pending_question["question"] = req.question
                pending_question["used_docs"] = set(source_names)
            else:
                # ❌ Not actually relevant — use general knowledge directly
                answer = ask_llm(req.question, chat_history.messages)
                source_summary = None
        else:
            # No relevant documents found at all - use general knowledge directly
            answer = ask_llm(req.question, chat_history.messages)
            source_summary = None

    # ✅ Always update chat history
    chat_history.add_user_message(req.question)
    chat_history.add_ai_message(answer)
    
    return {"answer": answer, "source": source_summary}
