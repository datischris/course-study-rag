# Course Study Tool Using Retrieval-Augmented Generation (RAG)

This project implements an offline study assistant using **retrieval-augmented generation (RAG)** to answer course-specific questions grounded in PDF lecture materials. It supports visual slide context and natural language response generation to help graduate students master complex topics.

## Abstract

This project presents a course study tool using a retrieval-augmented generation (RAG) system designed to assist graduate students in mastering concepts using PDF course materials. The system is built around three core tasks: 

1. Retrieving semantically relevant content from a collection of lecture slide PDFs using vector-based similarity search
2. Generating clear, detailed answers using a tuned Google FLAN-T5 large language model (LLM)
3. Enriching responses with relevant slide thumbnails retrieved from the original lecture material to provide visual context and support the RAG response. 

Using LangChain to orchestrate retrieval and structure the LLM inference pipeline, the system uses sentence-transformer embeddings and a FAISS (Facebook AI Similarity Search) vector store for efficient document indexing and similarity scoring. The extracted top two relevant slide images are rendered alongside responses in a full-stack interface, using Streamlit, to mimic a conversational tutoring environment. The RAG-based course study tool offers an offline, fully interactive study tool that grounds the generated explanations in the lecture content, improving clarity, retention, and accessibility for students.

## Repository Structure
``` 
├── dl-study-rag/ 
│   ├── app.py # Main Streamlit application containing RAG
│   ├── data/ # Folder containing lecture PDFs (user-provided) 
│   ├── faiss_index/ # FAISS index for semantic search (auto-generated) 
│   ├── requirements.txt # Pip dependencies for conda env 
│   ├── environment.txt # Conda environment specification 
│   ├── README.md # Project documentation
│   ...
```

