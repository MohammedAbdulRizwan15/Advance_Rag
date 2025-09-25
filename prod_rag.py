import streamlit as st
import os
import nltk
import redis
from langchain.cache import RedisCache
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings


# Download the punkt tokenizer
nltk.download('punkt')

# Set up Redis cache for caching
# redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Set OpenAI API key
OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key", type="password")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_excel(file_path):
    """Load and parse the Excel file."""
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    return docs

def split_text(docs):
    """Split the text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


def build_dense_retriever(chunks):
    """Build a dense retriever using embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(chunks, embeddings)
    dense_retriever = vector_store.as_retriever()
    return dense_retriever

def build_sparse_retriever(chunks):
    """Build a sparse retriever using BM25."""
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_fn=word_tokenize)
    return bm25_retriever

def hybrid_retriever(chunks):
    """Combine dense and sparse retrievers into an ensemble retriever."""
    semantic_retriever = build_dense_retriever(chunks)
    sparse_retriever = build_sparse_retriever(chunks)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )
    return ensemble_retriever

# def get_from_redis_cache(query):
#     """Check Redis cache for an existing answer."""
#     cache_key = f"question:{query.strip().lower()}"
#     cached_answer = redis_client.get(cache_key)
#     if cached_answer:
#         return cached_answer.decode('utf-8')
#     return None

def generate_response(query, related_docs):
    """Generate response using LLM and store it in Redis cache."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the question."),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])
    llm = ChatOpenAI(model="gpt-4")
    context = "\n".join([doc.page_content for doc in related_docs])
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": query})


    # Store the answer in Redis cache
    # cache_key = f"question:{query.strip().lower()}"
    # redis_client.set(cache_key, answer)
    
    # Display the answer
    return answer.content

# Set up the Streamlit UI
st.title("Excel Q and A Application")
uploaded_file = st.file_uploader("Upload your Excel file",
                                  type=["xlsx", "xls"])

if uploaded_file:
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    if "retriever" not in st.session_state:
        docs = load_excel(file_path) # calling
        chunks = split_text(docs)
        retriever = hybrid_retriever(chunks)
        st.session_state.retriever = retriever
  

question = st.chat_input("Ask your question about the document")

if question and "retriever" in st.session_state:
    # storing question on the interface
    st.chat_message("user").write(question)

    # answer_from_redis = get_from_redis_cache(question)
    # if answer_from_redis:
    #     st.chat_message("assistant").write(answer_from_redis)
    # else:
    # # extracting related documents from the retriever
    related_docs = st.session_state.retriever.invoke(question)
    answer = generate_response(question, related_docs)
    st.chat_message("assistant").write(answer) # returning the answer