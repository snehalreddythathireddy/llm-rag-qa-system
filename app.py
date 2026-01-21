import streamlit as st
import tempfile
from rag_pipeline import create_qa_chain

st.title("📚 LLM-powered RAG QA System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

qa_chain = None   # ✅ IMPORTANT: define it first

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    qa_chain = create_qa_chain(pdf_path)
    st.success("PDF processed successfully!")

question = st.text_input("Ask a question from the document")

if question and qa_chain is not None:
    response = qa_chain({"query": question})
    st.subheader("Answer")
    st.write(response["result"])

