# build_vectorstore.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load resume from TXT file
with open("./docs/resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Split into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_text(resume_text)
docs = [Document(page_content=chunk) for chunk in splits]

# Build vectorstore
embedding_model = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save vectorstore
vectorstore.save_local("vectorstore")

print("✅ Vectorstore built from TXT successfully.")
