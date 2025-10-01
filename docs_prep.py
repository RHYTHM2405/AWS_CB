import boto3
import os
import faiss
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings

# --- AWS Setup ---
s3 = boto3.client("s3")
bucket_name = "chat-bucket-rag-us"
index_file = "faiss_index"

# --- Step 1: Download PDFs from S3 ---
def download_pdfs_from_s3():
    objs = s3.list_objects_v2(Bucket=bucket_name)
    if "Contents" not in objs:
        print("No files found in S3.")
        return []

    pdf_files = []
    for obj in objs["Contents"]:
        key = obj["Key"]
        if key.lower().endswith(".pdf"):
            local_path = os.path.join("tmp.pdf")
            s3.download_file(bucket_name, key, local_path)
            pdf_files.append(local_path)
            print(f"Found and downloaded PDF: {key}")
    return pdf_files

# --- Step 2: Load Documents ---
def load_documents(pdf_files):
    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    return docs

# --- Step 3: Split & Embed ---
def build_faiss_index(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    vectorstore = FAISS.from_documents(splits, embeddings)

    # ✅ Save FAISS index locally instead of pickling
    vectorstore.save_local("faiss_index")

    print("FAISS index built and saved locally at ./faiss_index")


if __name__ == "__main__":
    pdf_files = download_pdfs_from_s3()
    if pdf_files:
        documents = load_documents(pdf_files)
        build_faiss_index(documents)
    else:
        print("No PDF files processed.")
