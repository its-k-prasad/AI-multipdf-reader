# 💬 AI multipdf reader

An interactive Streamlit application that allows you to upload PDF documents and chat with them using natural language queries. The app supports both text and voice input, powered by Google's Generative AI and advanced text processing capabilities.

## ✨ Features

- 🤖 **AI-Powered Responses** - Get intelligent answers from your PDF documents
- 🎤 **Voice Input Support** - Ask questions using speech-to-text functionality
- 📚 **Multi-Document Support** - Upload and process multiple PDFs simultaneously
- 🔍 **Semantic Search** - Advanced vector-based document search using FAISS
- 💬 **Conversational Interface** - Natural language question-answering
- 🎨 **Modern UI** - Clean, responsive design with gradient styling
- ⚡ **Real-time Processing** - Fast document indexing and query responses

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: Google Generative AI (Gemini 2.0 Flash), LangChain
- **Document Processing**: PyPDF2, RecursiveCharacterTextSplitter
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Speech Recognition**: SpeechRecognition, PyAudio
- **Environment Management**: python-dotenv

## 📋 Prerequisites

- Python 3.8+
- Google API Key for Generative AI
- Microphone access (for voice input feature)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pdf-chat-ai.git
   cd pdf-chat-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install streamlit PyPDF2 langchain google-generativeai faiss-cpu python-dotenv speechrecognition pyaudio langchain-google-genai
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Get Google API Key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload PDF Documents**
   - Use the sidebar to upload one or more PDF files
   - Click "Submit & Process" to index the documents

3. **Ask Questions**
   - **Text Input**: Type your question in the main input field
   - **Voice Input**: Click the microphone button and speak your question

4. **Get AI Responses**
   - The app will search through your documents
   - Receive detailed, context-aware answers
   - If information isn't found, you'll get a clear "not present in docs" response

## 📁 Project Structure

```
pdf-chat-ai/
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
├── faiss_index/          # Vector database storage (auto-created)
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Generative AI API key

### Model Settings
- **Embedding Model**: `models/embedding-001`
- **Chat Model**: `gemini-2.0-flash`
- **Temperature**: 0.3 (for consistent responses)
- **Chunk Size**: 10,000 characters
- **Chunk Overlap**: 1,000 characters

## 🎤 Voice Input Requirements

### Windows
```bash
pip install pyaudio
```

### macOS
```bash
brew install portaudio
pip install pyaudio
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install python3-pyaudio
pip install pyaudio
```

## 📝 How It Works

1. **Document Processing**
   - PDFs are extracted using PyPDF2
   - Text is split into manageable chunks
   - Chunks are converted to embeddings using Google's embedding model
   - Vector database (FAISS) is created for fast similarity search

2. **Question Answering**
   - User questions are embedded using the same model
   - Similar document chunks are retrieved using vector search
   - Google's Gemini model generates answers based on retrieved context
   - Responses are formatted and displayed with proper styling

3. **Voice Processing**
   - Microphone input is captured using PyAudio
   - Speech is converted to text using Google Speech Recognition
   - Text is processed through the same Q&A pipeline

## 🎨 UI Features

- **Gradient Header**: Eye-catching purple gradient design
- **Responsive Layout**: Two-column layout with question input and voice controls
- **Status Indicators**: Real-time feedback for processing and voice input
- **Styled Containers**: Beautiful card-like containers for questions and responses
- **Interactive Elements**: Hover effects and smooth transitions

## ⚠️ Limitations

- Requires active internet connection for AI processing
- Voice input timeout is 15 seconds per query
- PDF text extraction may not work perfectly with image-heavy or complex layouts
- Vector database is stored locally and needs reprocessing after restart

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Google Generative AI for powerful language models
- LangChain for document processing utilities
- Streamlit for the amazing web framework
- FAISS for efficient vector similarity search

## 🐛 Troubleshooting

**Issue**: "Module not found" errors
- **Solution**: Ensure all dependencies are installed in your virtual environment

**Issue**: Google API errors
- **Solution**: Verify your API key is correct and has proper permissions

**Issue**: Microphone not working
- **Solution**: Check microphone permissions and PyAudio installation

**Issue**: PDF processing fails
- **Solution**: Ensure PDFs are not password-protected or corrupted

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

⭐ If you find this project helpful, please give it a star on GitHub!
