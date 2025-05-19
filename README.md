# Course Study Tool Using Retrieval-Augmented Generation (RAG)

This project implements an offline study assistant using **retrieval-augmented generation (RAG)** to answer course-specific questions grounded in PDF lecture materials. It shows corresponding slides as context when generating a natural language response generation to help graduate students master complex topics.

## Abstract

This project presents a course study tool using a retrieval-augmented generation (RAG) system designed to assist graduate students in mastering concepts using PDF course materials. The system is built around three core tasks: 

1. Retrieving semantically relevant content from a collection of lecture slide PDFs using vector-based similarity search
2. Generating clear, detailed answers using a tuned Google FLAN-T5 large language model (LLM)
3. Enriching responses with relevant slide thumbnails retrieved from the original lecture material to provide visual context and support the RAG response. 

Using LangChain to orchestrate retrieval and structure the LLM inference pipeline, the system uses sentence-transformer embeddings and a FAISS (Facebook AI Similarity Search) vector store for efficient document indexing and similarity scoring. The extracted top two relevant slide images are rendered alongside responses in a full-stack interface, using Streamlit, to mimic a conversational tutoring environment. The RAG-based course study tool offers an offline, fully interactive study tool that grounds the generated explanations in the lecture content, improving clarity, retention, and accessibility for students.

## Repository Structure
``` 
├── dl-study-rag/ 
│   ├── app.py              # Main Streamlit application containing RAG
│   ├── data/               # Folder containing lecture PDFs (user-provided) 
│   ├── faiss_index/        # FAISS index folder for semantic search (auto-generated)
│   │   ├── index.faiss     # Contains compressed representation of vector store
│   │   ├── index.pkl       # Chunking METADATA for vector store
│   ├── requirements.txt    # Pip dependencies for conda env 
│   ├── environment.txt     # Conda environment specification 
│   ├── README.md           # Project documentation
│   ...
```

> **Note:** PDF slides are not included due to copyright. Please place your own slide decks into the `data/` folder and follow below on how to create the embeddings.


## How to Run

1. **Set up the environment using Conda**
   ```bash
   conda create --name rag-env --file environment.txt
   conda activate rag-env
2. **Install the necessary Python dependencies**
   ```bash
   pip install -r requirements.txt
3. **Place all of your course PDF slides into the `data/` folder.**
4. **Run the app.**
   ```bash
   streamlit run app.py
5. **Navigate to the sidebar and click *Rebuild the embeddings*.**

## Compute Requirements
This project was developed and tested on a local Windows 11 machine with the following specifications:

| **Component**    | **Specification**                                      |
|------------------|--------------------------------------------------------|
| OS               | Windows 11 Home (Build 22631)                          |
| Processor        | Intel Core i7-1255U, 10 cores (12 threads), 2.6 GHz    |
| RAM              | 32 GB                        |
| System Type      | x64-based PC                                           |

The system ran entirely on CPU with no dedicated GPU, and the FLAN-T5 large model inference remained responsive due to efficient CPU deployment using the transformers.pipeline module.

This configuration is suitable for small to medium PDF sets. For larger datasets or faster inference, using a GPU or cloud-based setup is recommended.