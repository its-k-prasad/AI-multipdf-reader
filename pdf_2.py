import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import pyaudio
import threading
import time

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context from PDF documents.
    
    Instructions:
    - If the answer is available in the context, provide a detailed response
    - If the answer is NOT available in the context, simply respond: "This information is not present in the docs"
    - Do not make up information or provide answers not supported by the context
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def record_audio():
    """Record audio from microphone and convert to text"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        # Adjust for ambient noise
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        st.info("🎤 Listening... Speak now!")
        
        # Record audio
        with microphone as source:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        
        st.success("Recording complete! Converting to text...")
        
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        return text
    
    except sr.WaitTimeoutError:
        return "Listening timeout - no speech detected"
    except sr.UnknownValueError:
        return "Could not understand the audio clearly"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error accessing microphone: {e}"


def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents":docs, "question": user_question}
            , return_only_outputs=True)

        print(response)
        
        # Display the response in a properly formatted container
        st.markdown(f"""
        <div class="response-container">
            <h4>🤖 AI Response:</h4>
            <p>{response["output_text"]}</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Please upload and process PDF documents first! Error: {str(e)}")


def main():
    # Page configuration
    st.set_page_config(
        page_title="Chat PDF",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 15px;
        font-size: 16px;
        width: 100%;
    }
    .stButton > button {
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .chat-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .question-container {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .response-container {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .voice-status {
        background: #28a745;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
    }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .voice-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="chat-container">
        <h1>💬 Chat with PDF using AI</h1>
        <p>Upload your PDF documents and ask questions using text or voice input</p>
    </div>
    """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 💭 Ask Your Question")
        user_question = st.text_input(
            "Type your question here...",
            placeholder="What would you like to know about your documents?",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if user_question:
            # Display user question
            st.markdown(f"""
            <div class="question-container">
                <h4>❓ Your Question:</h4>
                <p>{user_question}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Processing your question..."):
                user_input(user_question)
    
    with col2:
        st.markdown('<div class="voice-section">', unsafe_allow_html=True)
        st.markdown("### 🎤 Voice Input")
        if st.button("🎤 Start Recording", type="primary", use_container_width=True):
            if os.path.exists("faiss_index"):
                with st.spinner("Getting ready to record..."):
                    transcribed_text = record_audio()
                
                if transcribed_text and not any(error in transcribed_text for error in ["Could not", "Error", "Listening"]):
                    st.markdown(f'<div class="voice-status">✅ Voice input successful!</div>', unsafe_allow_html=True)
                    
                    # Display transcribed question
                    st.markdown(f"""
                    <div class="question-container">
                        <h4>❓ Your Question:</h4>
                        <p>{transcribed_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner("Processing your question..."):
                        user_input(transcribed_text)
                else:
                    st.error(transcribed_text)
            else:
                st.warning("⚠️ Please upload and process PDF documents first!")
        st.markdown('</div>', unsafe_allow_html=True)

    # Instructions section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📁 Step 1: Upload PDFs
        - Go to the sidebar
        - Upload your PDF files
        - Click "Submit & Process"
        """)
    
    with col2:
        st.markdown("""
        ### 💬 Step 2: Ask Questions
        - Type your question in the text box
        - Or use voice input by clicking the microphone
        - Get detailed answers from your documents
        """)
    
    with col3:
        st.markdown("""
        ### 🎙️ Voice Input Tips
        - Speak clearly and at moderate pace
        - Ensure minimal background noise
        - Wait for "Listening..." message
        - You have 15 seconds to ask
        """)

    # Sidebar for PDF upload
    with st.sidebar:
        st.markdown("### 📄 Document Upload")
        
        pdf_docs = st.file_uploader(
            "Choose PDF files",
            accept_multiple_files=True,
            type=['pdf'],
            help="Upload one or more PDF documents to chat with"
        )
        
        if pdf_docs:
            st.markdown("**Uploaded files:**")
            for pdf in pdf_docs:
                st.markdown(f"📄 {pdf.name}")
        
        if st.button("🚀 Submit & Process", type="primary", use_container_width=True):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Documents processed successfully!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.error("Please upload PDF files first!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This app uses advanced AI to help you chat with your PDF documents. 
        
        **Features:**
        - 🤖 AI-powered responses
        - 🎤 Voice input support
        - 📚 Multi-document support
        - 🔍 Semantic search
        """)


if __name__ == "__main__":
    main()

