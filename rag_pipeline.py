from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline


def create_qa_chain(pdf_path):
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split text into chunks (⬅️ CHANGE CHUNK SIZE HERE)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # 👈 THIS is chunk size
        chunk_overlap=100
    )
    texts = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 4. Store embeddings in FAISS
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 5. Load LLM
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)

    # 6. Strict prompt to avoid hallucination
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

    # 7. Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
