from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.prompts import PromptTemplate


def create_qa_chain(pdf_path):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 👈 better accuracy
        chunk_overlap=150
    )
    texts = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store embeddings in FAISS
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Load LLM
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )

    llm = HuggingFacePipeline(pipeline=llm_pipeline)

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

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
