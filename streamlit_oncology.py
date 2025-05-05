import streamlit as st
import openai
import os
import faiss
import requests
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from py2neo import Graph

openai.api_key = os.environ['OPENAI_API_KEY']

st.title("📚 Book QA System")

book_choice = st.selectbox("Select a Book", options=["Pride and Prejudice", "New Book"])

book_text = None
book_name = "pride_and_prejudice" if book_choice == "Pride and Prejudice" else None

if book_choice == "Pride and Prejudice":
    if not os.path.exists("pride_and_prejudice.txt"):
        st.markdown("*📥 Downloading Pride and Prejudice...*")


        @st.cache_data(show_spinner=False)
        def fetch_default_book():
            url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
            response = requests.get(url)
            with open("pride_and_prejudice.txt", 'w', encoding='utf-8') as f:
                f.write(response.text)
            return response.text


        book_text = fetch_default_book()
    else:
        with open("pride_and_prejudice.txt", 'r', encoding='utf-8') as f:
            book_text = f.read()

elif book_choice == "New Book":
    book_file = st.file_uploader("Upload a `.txt` file for the new book", type="txt")
    if book_file is not None:
        book_text = book_file.read().decode("utf-8")
        book_name = book_file.name.replace(".txt", "")

if book_text:
    @st.cache_data(show_spinner=False)
    def chunk_text(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_text(text)


    chunks = chunk_text(book_text)


    @st.cache_resource(show_spinner=False)
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")


    model = load_model()


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


    def get_top_chunks(query, k=3):
        query_embedding = model.encode([query])
        D, I = index.search(query_embedding, k)
        return [chunks[i] for i in I[0]]


    def run_neo4j_query(query):
        try:
            graph = Graph("bolt://localhost:7687",
                          auth=("neo4j", "ontology123"))
            results = graph.run(query).data()
            return results
        except Exception as e:
            st.error(f"Failed to run Neo4j query: {e}")
            return []


    def generate_answer(chunks, query):
        # Fetch relevant chunks from FAISS
        context = "\n".join(chunks)

        prompt = f"""
            You are an assistant that answers questions about a book using the provided context.

            Context:
            \"\"\"
            {context}
            \"\"\"

            Now, I also want you to consider the information from the graph database (Neo4j). Here is some additional context:

            (Add Neo4j context here when you run the query)

            Question: {query}
            Answer:"""

        # Use Neo4j query to fetch related entities or relationships (optional)
        neo4j_query = f"MATCH (n)-[r]->(m) WHERE n.name CONTAINS '{query}' RETURN n.name, type(r), m.name LIMIT 5"
        neo4j_results = run_neo4j_query(neo4j_query)

        # Append Neo4j results to the context for the model
        if neo4j_results:
            graph_context = "\n".join(
                [f"{res['n.name']} - {res['type(r)']} - {res['m.name']}" for res in neo4j_results])
            prompt = prompt.replace("(Add Neo4j context here when you run the query)", graph_context)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )

        return response.choices[0].message['content'].strip()


    st.markdown("---")
    st.markdown("### 💬 Ask a question about the book")
    user_query = st.text_input("🔍 Enter your question:")

    if user_query:
        relevant_chunks = get_top_chunks(user_query)

        print("\n🔍 Retrieved Chunks:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i + 1} ---\n{chunk}\n")

        answer = generate_answer(relevant_chunks, user_query)

        print("\n✅ Generated Answer:")
        print(answer)

        st.markdown("### 📖 Answer")
        st.write(answer)

else:
    if book_choice != "":
        st.warning("⚠️ Please upload a valid `.txt` file or choose a book.")

