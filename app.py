import os
import shutil
import fitz
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

## trigger a rebuild of embeddings when button clicked on st.rerun
if "rebuild_embeddings" not in st.session_state:
    st.session_state.rebuild_embeddings = False

## webpage styling and added favicon png image
st.set_page_config(
    page_title="Course Study RAG", 
    layout="centered"
)

st.markdown(
    """
    <style>
      /* Make the main content column narrow and centered */
      .block-container {
        max-width: 700px;
        margin: 0 auto;
        padding-top: 1rem;
        padding-bottom: 1rem;
      }
      /* Tighter spacing between elements */
      .element-container {
        padding: 0.5rem 0;
      }
      /* Hide Streamlit header & footer for a cleaner look */
      header, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center">
      <h1 style="margin-bottom:4px;">Your Course Study RAG</h1>
      <p style="
          color: rgba(100,100,100,0.6);
          font-size: 0.8rem;
          margin-top:0;
          margin-bottom:16px;
      ">
        I've indexed all your lecture PDFs. I will answer your queries and highlight key slide thumbnails.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# global variables for paths
DATA_DIR  = "data"
INDEX_DIR = "faiss_index"

# sidebar styling and features
with st.sidebar:
    st.markdown("# Course Study RAG", unsafe_allow_html=True)

    st.sidebar.header("Lecture PDFs for Download")

    # map filenames to nice labels
    pdf_files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")])

    for idx, fname in enumerate(pdf_files, start=1):
        path = os.path.join(DATA_DIR, fname)
        with open(path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.sidebar.download_button(
            label=f"📄 {fname}",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            key=f"dl_{fname}"
        )

    # options section
    st.subheader("Options")

    # embeddings button
    if st.button("Rebuild the embeddings"):
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        st.session_state.rebuild_embeddings = True
        st.rerun()
    
    # slide thumbnail toggle
    show_images = st.checkbox("Show slide thumbnails", value=True)


# caching images for quick chat retreival used to prevent repeated PDF parsing and image rendering
# without caching you can expect slow speeds in response generation
# iterates through the pdfs and grabs screenshots (as pixmaps) and stores at DATA_DIR
@st.cache_data(show_spinner=False)
def load_pdf_images(data_dir):
    images = {}
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".pdf"):
            continue
        pdf = fitz.open(os.path.join(data_dir, fn))
        for i in range(pdf.page_count):
            pix = pdf.load_page(i).get_pixmap()
            # store as PNG bytes to avoid repeated conversion
            png = pix.pil_tobytes(format="PNG")
            images[(fn, i+1)] = png
    return images

PDF_IMAGES = load_pdf_images(DATA_DIR)


# creating or retrieving vector embeddings (if path exists) using MiniLM-L6 model
@st.cache_resource(show_spinner=False)
def get_vectorstore(data_dir, index_dir, force_reload):
    
    # init embedding model (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
    # using a small context length model (max 256 tokens) which ideal for lightweight semantic retrieval within PDFs
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # if embeddings already exist, return them
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    # otherwise we build the embeddings and let user know via progress bar
    docs = []
    pdf_files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith(".pdf")]
    progress = st.progress(0.0, text="Splitting slides into chunks…")
    for idx, fn in enumerate(pdf_files):
        path = os.path.join(data_dir, fn)
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        docs.extend(pages)
        progress.progress((idx+1)/len(pdf_files), text="Splitting slides into chunks…")

    # this controls the overall splitting of the text for passing to embedder
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # flag check and reset
    if st.session_state.rebuild_embeddings:
        st.session_state.rebuild_embeddings = False

    # create FAISS (Facebook AI Similarity Search) vector store database and save embeddings in INDEX_DIR directory
    # enables fast similarity search by indexing and retrieving the most semantically similar vectors using approximate nearest neighbor algorithms.
    vs = FAISS.from_documents(chunks, embedder)
    vs.save_local(index_dir)
    
    # removing progress bar
    progress.empty()
    return vs

vectorstore = get_vectorstore(DATA_DIR, INDEX_DIR, st.session_state.rebuild_embeddings) # creating vector store variable for retrieval in model creation

# setting up flan-t5 on local cpu for processing and LM output (https://huggingface.co/google/flan-t5-large)
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1,
    max_length=512,       # limit generation length
    do_sample=True,       # enable sampling (temperature unlocked)
    temperature=0.9       # adjust as needed
)

# setting up the LM pipeline and retriever
local_llm = HuggingFacePipeline(pipeline=pipe)

# creating a chat persistant state -- allowing message stacking from previous queries
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat input and response output controller
prompt = st.chat_input("Ask a question about CSE676: Deep Learning!")

if prompt:
    # append input to user content and declare it as a "prompt" for flan-t5-large model
    st.session_state.messages.append({"role": "user", "content": prompt})

    # creating placeholder blurb for bot to think and let user know output is coming
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message("assistant"):
            st.write("Course Bot is thinking...")

    # create spinner and do the heavy work
    with placeholder:

        # conduct a similarity search using embeddings and 
        docs = vectorstore.similarity_search(prompt,k=8)

        # build context + full_prompt from similarity search results
        context_list = []
        for d in docs:
            fn = os.path.basename(d.metadata["source"])
            pg = d.metadata.get("page", 1)
            context_list.append(f"[{fn} — slide {pg}]\n{d.page_content}")
        context = "\n\n".join(context_list)

        # create a prompt for flan-t5-large model
        full_prompt = (
            "You are a university professor teaching from these lecture slides. "
            "Use ONLY these excerpts to craft a clear, step-by-step answer.\n\n"
            f"{context}\n\nQUESTION: {prompt}\n\nANSWER:"
        )

        # let the local_llm handle retrieval + formatting + truncation
        ans = local_llm(full_prompt)

        # grab the top-2 slide refs yourself if you still want thumbnails
        docs = vectorstore.similarity_search(prompt, k=4)
        slides = []
        if show_images:
            for d in docs[:2]:
                fn = os.path.basename(d.metadata["source"])
                pg = d.metadata.get("page", 1)
                slides.append((fn, pg))

    # clear placeholder thinking prompt and reveal the answer
    placeholder.empty()
    st.session_state.messages.append({
        "role":    "assistant",
        "content": ans,
        "slides":  slides
    })

# render all messages from previous + current chat history and output to page
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_images:
            for fn, pg in msg.get("slides", []):
                png = PDF_IMAGES.get((fn, pg))
                if png:
                    st.image(png, caption=f"{fn} — slide {pg}", width=350)

# ensrue auto-scroll anchor down to most recent output when it is revealed to the user
st.markdown("<div id='scroll-anchor'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <script>
      const anchor = window.parent.document.getElementById('scroll-anchor');
      if (anchor) {
        anchor.scrollIntoView({behavior: 'smooth', block: 'end'});
      }
    </script>
    """,
    unsafe_allow_html=True,
)