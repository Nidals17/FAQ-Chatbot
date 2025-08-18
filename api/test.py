import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import traceback

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
    try:
        formatted_prompt = prompt.format_prompt(
            chat_history=history,
            question=question
        ).to_messages()
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        print(f"❌ Error in ask_llm: {e}")
        return "I'm sorry, I encountered an error while processing your question."

@app.post("/chat")
async def chat(req: QuestionRequest):
    try:
        print(f"🔵 Received question: '{req.question}'")
        
        query = req.question.strip()
        if not query:
            return {"answer": "Please ask a question.", "source": None}
        
        # Step 1: Try to find relevant documents
        try:
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=5)
            print(f"🔍 Found {len(docs_and_scores)} documents from vector search")
            
            # Filter by relevance threshold
            relevance_threshold = 0.7
            relevant_docs = [doc for doc, score in docs_and_scores if score >= relevance_threshold]
            print(f"📄 {len(relevant_docs)} documents passed relevance threshold ({relevance_threshold})")
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            relevant_docs = []
        
        # Step 2: Decide whether to use RAG or general knowledge
        answer = None
        source_summary = None
        
        if relevant_docs:
            try:
                # Use RAG - answer from documents
                context_text = "\n\n".join(doc.page_content for doc in relevant_docs)
                question_with_context = f"Answer using the following documents:\n{context_text}\n\nQuestion: {query}"
                
                print("📚 Using RAG mode (documents found)")
                answer = ask_llm(question_with_context, chat_history.messages)
                
                # Get source names
                source_names = []
                for doc in relevant_docs:
                    source = doc.metadata.get("source", "unknown")
                    # Clean up source path
                    source = source.replace("FAQ-docs\\", "").replace("FAQ-docs/", "")
                    source_names.append(source)
                
                source_summary = ", ".join(set(source_names))
                print(f"✅ RAG response generated with sources: {source_summary}")
                
            except Exception as e:
                print(f"❌ Error in RAG mode: {e}")
                # Fallback to general knowledge
                answer = ask_llm(query, chat_history.messages)
                source_summary = None
                print("🧠 Fell back to general knowledge due to RAG error")
        else:
            # No relevant documents - use general knowledge
            print("🧠 Using general knowledge (no relevant documents)")
            answer = ask_llm(query, chat_history.messages)
            source_summary = None
        
        # Ensure we have a valid answer
        if not answer or answer.strip() == "":
            answer = "I'm sorry, I couldn't generate a response to your question."
        
        # Update chat history
        try:
            chat_history.add_user_message(query)
            chat_history.add_ai_message(answer)
            print("💾 Chat history updated")
        except Exception as e:
            print(f"⚠️ Warning: Could not update chat history: {e}")
        
        print(f"✅ Returning response (length: {len(answer)} characters)")
        return {"answer": answer, "source": source_summary}
        
    except Exception as e:
        print(f"💥 FATAL ERROR in chat endpoint:")
        print(traceback.format_exc())
        
        # Return a proper error response instead of crashing
        return {"answer": "I'm sorry, I encountered a technical error. Please try again.", "source": None}