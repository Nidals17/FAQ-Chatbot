import os
import hashlib
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain.text_splitter import CharacterTextSplitter

CHROMA_DIR = "chroma_db"
DATA_FOLDER = "FAQ-docs"
HASH_FILE = "docs_hash.pkl"

def get_folder_hash(folder_path):
    """Generate a hash based on all file contents in the folder."""
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

def create_or_load_chroma():
    # Check if chroma exists and if docs have changed
    current_hash = get_folder_hash(DATA_FOLDER)
    if os.path.exists(CHROMA_DIR) and os.path.exists(HASH_FILE):
        with open(HASH_FILE, "rb") as f:
            saved_hash = pickle.load(f)
        if saved_hash == current_hash:
            print("✅ No changes detected. Loading existing Chroma DB...")
            return load_chroma_db()

    print("📦 Changes detected or DB missing. Rebuilding Chroma DB...")
    vectorstore = create_chroma_db()
    with open(HASH_FILE, "wb") as f:
        pickle.dump(current_hash, f)
    return vectorstore

def create_chroma_db():
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".json": JSONLoader
    }
    documents = []
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            if ext == ".json":
                loader = JSONLoader(file_path, jq_schema=".text")
            else:
                loader = loaders[ext](file_path)
            documents.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    vectorstore.persist()
    return vectorstore

def load_chroma_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
