from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

from transformers import pipeline
from typing import Optional, List


# Custom LLM wrapper (stable)
class HFText2TextLLM(LLM):
    pipeline: any

    @property
    def _llm_type(self) -> str:
        return "huggingface_text2text"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt)
        return response[0]["generated_text"]


def create_qa_chain(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    # HuggingFace model
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HFText2TextLLM(pipeline=hf_pipeline)

    # Prompt
    prompt = PromptTemplate(
        template="""
You are an AI assistant answering questions strictly based on the provided context.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
