import streamlit as st
import openai
import os
import faiss
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------------
# 🔐 Set OpenAI API Key
# ---------------------------
openai.api_key = ""
# ---------------------------
# 🎨 Streamlit UI - Book Selection
# ---------------------------
st.title("📚 Book QA System")
st.markdown("Choose to use the default book (*Pride and Prejudice*) or upload your own `.txt` file.")

book_name = st.text_input("📘 Enter Book Name (or type 'default' to use Pride and Prejudice):").strip().lower()

book_text = None
book_file_path = f"{book_name}.txt"

# Check if the file already exists, if so, use it
if os.path.exists(book_file_path):
    with open(book_file_path, 'r', encoding='utf-8') as f:
        book_text = f.read()

if book_name == "default" and not os.path.exists(book_file_path):
    # Download Pride and Prejudice only if it doesn't exist already
    st.markdown("*📥 Downloading Pride and Prejudice...*")


    @st.cache_data(show_spinner=False)
    def fetch_default_book():
        url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
        response = requests.get(url)
        with open(book_file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        return response.text


    book_text = fetch_default_book()
    book_name = "pride_and_prejudice"

elif book_name != "default":
    uploaded_file = st.file_uploader("📄 Upload your book (.txt only):", type=["txt"])
    if uploaded_file is not None:
        book_text = uploaded_file.read().decode("utf-8")
        # Save the uploaded book text to the local file system
        with open(book_file_path, 'w', encoding='utf-8') as f:
            f.write(book_text)

# Proceed only if text is available
if book_text:
    # ---------------------------
    # 🧩 Chunking Function
    # ---------------------------
    @st.cache_data(show_spinner=False)
    def chunk_text(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)


    chunks = chunk_text(book_text)


    # ---------------------------
    # 🤖 Load Sentence Embedding Model
    # ---------------------------
    @st.cache_resource(show_spinner=False)
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")


    model = load_model()


    # ---------------------------
    # 🧠 Create or Load FAISS Index
    # ---------------------------
    @st.cache_resource(show_spinner=False)
    def create_or_load_faiss_index(chunks, book_name):
        index_file = f"{book_name}_index.faiss"

        if os.path.exists(index_file):
            index = faiss.read_index(index_file)
        else:
            embeddings = model.encode(chunks)
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            faiss.write_index(index, index_file)

        return index


    with st.spinner("🔍 Indexing book into vector database..."):
        index = create_or_load_faiss_index(chunks, book_name)
        st.success("✅ Indexed into vector DB")


    # ---------------------------
    # 🔍 Search Top Chunks
    # ---------------------------
    def get_top_chunks(query, k=3):
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]


    # ---------------------------
    # 💡 Generate Answer
    # ---------------------------
    def generate_answer(chunks, query):
        context = "\n".join(chunks)
        prompt = f"""
            You are an assistant that answers questions about a book using the provided context.

            Context:
            \"\"\"
            {context}
            \"\"\"

            Question: {query}
            Answer:"""

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message['content'].strip()


    # ---------------------------
    # 🧠 Ask Question
    # ---------------------------
    st.markdown("---")
    st.markdown("### 💬 Ask a question about the book")
    user_query = st.text_input("🔍 Enter your question:")

    if user_query:
        relevant_chunks = get_top_chunks(user_query)

        # Log retrieved chunks
        print("\n🔍 Retrieved Chunks:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i + 1} ---\n{chunk}\n")

        answer = generate_answer(relevant_chunks, user_query)

        # Log final answer
        print("\n✅ Generated Answer:")
        print(answer)

        # Show in app
        st.markdown("### 📖 Answer")
        st.write(answer)

else:
    if book_name != "":
        st.warning("⚠️ Please upload a valid `.txt` file to continue.")
