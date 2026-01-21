import streamlit as st
from rag_pipeline import create_qa_chain
import tempfile

st.title("📚 LLM-powered RAG QA System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
 with tempfile.NamedTemporaryFile(delete=False) as tmp:
     tmp.write(uploaded_file.read())
     pdf_path = tmp.name

 qa = create_qa_chain(pdf_path)

 question = st.text_input("Ask a question from the document")

 if question:
     answer = qa.run(question)
     st.subheader("Answer:")
     st.write(answer)
