import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Initialize embedding model
local_embeddings = OllamaEmbeddings(model="llama3.1:8b")

# Directory to store processed batches and vector database
persist_root = "./medqna/vectordb"
vectorstore = Chroma(
    persist_directory=persist_root,
    embedding_function=local_embeddings
)

# Initialize Q&A model
model = ChatOllama(model="llama3.1:8b")

# Template for RAG (Retrieval-Augmented Generation) prompt
RAG_TEMPLATE = """
You are a medical assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

# Helper function to format documents
def format_docs(docs):
    if not docs:
        return "No relevant documents were found."
    return "\n\n".join(doc.page_content for doc in docs)

# Set up retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Define the Q&A chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

# Initialize Streamlit state
if "processing" not in st.session_state:
    st.session_state.processing = False
    st.session_state.answer = None
    st.session_state.error = None

# Streamlit app layout
st.title("Medical Question Answering Assistant")
st.write("Enter a medical question, and the assistant will retrieve relevant documents and provide a concise answer.")

# Input field for user question
question = st.text_area("Enter your question here:", height=150)

# Disable 'Get Answer' if already processing
get_answer_disabled = st.session_state.processing

# Button to start processing the answer
if st.button("Get Answer", disabled=get_answer_disabled) and question:
    st.session_state.processing = True
    st.session_state.answer = None
    st.session_state.error = None
    
    # Start answer retrieval in a spinner
    with st.spinner("Retrieving and processing..."):
        try:
            # Retrieve and process the answer
            st.session_state.answer = qa_chain.invoke(question)
        except Exception as e:
            st.session_state.error = f"An error occurred: {e}"
        finally:
            st.session_state.processing = False

# Display answer or error
if st.session_state.answer:
    st.success(st.session_state.answer)

if st.session_state.error:
    st.error(st.session_state.error)

# Show retrieved context checkbox
if st.session_state.answer and st.checkbox("Show Retrieved Context"):
    docs = retriever.invoke(question)
    st.subheader("Retrieved Contexts")
    st.write(format_docs(docs))
