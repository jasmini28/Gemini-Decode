import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import base64
from io import BytesIO

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from uploaded PDF files.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the text into manageable chunks.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and store the vector store for similarity searches.
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Build the conversational chain using a custom prompt.
def get_conversational_chain():
    prompt_template = """You have the following PDF context. Analyze it carefully to answer the question below.
Your answer should be detailed, accurate, and based solely on the provided context.
If the answer is not found in the context, respond with: "answer is not available in the context."
Refrain from providing any incorrect or fabricated information.
    Context: \n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.markdown("<h4>Bot:</h4>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="response-box">
            {response["output_text"]}
        </div>
        """,
        unsafe_allow_html=True
    )

def add_custom_css():
    st.markdown(
        """
        <style>
        /* Custom header styling */
        .header {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #ffffff;
            background-color: #084b66; 
            padding: 20px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        /* Remove default Streamlit padding around the main block */
        .main > div {
            padding-top: 0px;
        }
        /* Custom footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #084b66;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        .response-box {
            background-color: #050d21;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Add a footer with your name.
def add_footer():
    st.markdown(
        """
        <div class="footer">
            Developed by Jasmini Miriyala
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Chat With Multiple PDF", layout="centered")
    add_custom_css()
    st.markdown('<div class="header"><h1>Gemini Decode: PDF Bot </h1></div>', unsafe_allow_html=True)

    pdf_docs = st.file_uploader("Upload Your PDF Files", accept_multiple_files=True, type='pdf')
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")
            else:
                st.warning("Please upload at least one PDF before processing.")

    user_question = st.text_input("Ask any Question from the PDF Files")
    if user_question:
        user_input(user_question)

    add_footer()

if __name__ == "__main__":
    main()
