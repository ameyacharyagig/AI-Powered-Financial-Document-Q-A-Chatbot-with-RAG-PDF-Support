import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="📊 Hybrid Finance Chatbot", layout="wide")
st.title("💼 Financial Report Chatbot")

# STEP 1: Company Selection
company_list = ["reliance", "tcs", "Upload Manually"]
company_choice = st.selectbox("Choose a company or upload your own PDF:", company_list)

# Mapping company to correct file path
company_paths = {
    "reliance": r"D:/Summer/Chatbot/RIL-Integrated-Annual-Report-2023-24.pdf",
    "tcs": r"D:/Summer/Chatbot/annual-report-2023-2024.pdf"
}

def extract_text(pdf_path_or_file):
    reader = PdfReader(pdf_path_or_file)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

@st.cache_resource(show_spinner="🔍 Generating embeddings... Please wait.")
def build_vectorstore(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

vectorstore = None
file_label = None

# STEP 2A: Preloaded Company
if company_choice != "Upload Manually":
    filepath = company_paths.get(company_choice)
    if filepath and os.path.exists(filepath):
        st.success(f"✅ Loaded report for {company_choice.title()}")
        text = extract_text(filepath)
        vectorstore = build_vectorstore(text)
    else:
        st.warning("⚠️ Report not found. Please upload manually.")

# STEP 2B: Manual Upload
else:
    uploaded_file = st.file_uploader("📁 Upload company annual report", type="pdf")
    if uploaded_file:
        file_label = uploaded_file.name
        st.success(f"✅ '{file_label}' uploaded.")
        text = extract_text(uploaded_file)
        vectorstore = build_vectorstore(text)

# STEP 3: QA Chat
if vectorstore:
    llm = Ollama(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    st.markdown("### 💬 Ask your question:")
    user_query = st.text_input("e.g., What is the profit? What are the challenges?")
    if user_query:
        with st.spinner("Thinking..."):
            result = qa_chain.run(user_query)
        st.write("🧠 **Answer:**")
        st.success(result)
