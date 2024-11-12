# MedQNA-Answering-System
Created an LLM application using langchain and llama 3.1 for answering medical questions in english using document retrieval system. It can be classified as an RAG system.
The streamlit code has been given in the repo

1. The chromadb vector files can be downloaded from [https://drive.google.com/file/d/1q9jWfLbHzLkXOqCAvr_RMH8cFeHZOzK4/view?usp=drivesdk](https://drive.google.com/file/d/1q9jWfLbHzLkXOqCAvr_RMH8cFeHZOzK4/view?usp=drivesdk)
The persist_root variable in the streamlit code is set as ./medqna/vectordb  
Change it to the downloaded vector directory after extraction.
2. An Ollama instance and python installation should also be available for this project. Download and install ollama from [https://ollama.com/download](https://ollama.com/download) 
```sh
ollama serve
ollama pull llama3.1:8b
```
5. Run the Streamlit application:
```sh
pip install -r requirements.txt
streamlit run streamlit_app.py
```
