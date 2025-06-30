from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit as st
import os
import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.graphs import Neo4jGraph
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import re
from langchain_groq import ChatGroq

# Load secure secrets from .streamlit/secrets.toml
NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USERNAME = st.secrets["neo4j"]["username"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]
NEO4J_DATABASE = st.secrets["neo4j"]["database"]

groq_api_key = st.secrets["groq"]["api_key"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = st.secrets["langchain"]["api_key"]

@st.cache_resource
def get_embedding_model():
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
    return model, tokenizer

class JinaEmbeddings(Embeddings):
    def __init__(self):
        self.model, self.tokenizer = get_embedding_model()

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

@st.cache_resource
def initialize_llm():
    return ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

def get_kg():
    return Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

@st.cache_data
def process_pdf(uploaded_pdf):
    pdf_file = PdfReader(uploaded_pdf)
    return "".join([page.extract_text() for page in pdf_file.pages])

@st.cache_data
def scrape_website(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]): script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return '\n'.join(chunk for chunk in chunks if chunk)
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None

def build_vector_index(text):
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding = JinaEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    return VectorStoreRetriever(vectorstore=vectorstore)

def del_nodes():
    get_kg().query("MATCH (n) DETACH DELETE n")

def doc2graph(processed_text, llm):
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_text(processed_text)
    if len(chunks) > 30:
        st.warning(f"Large document ({len(chunks)} chunks), using first 30 for graph.")
        chunks = chunks[:30]
    transformer = LLMGraphTransformer(llm=llm)
    return transformer.convert_to_graph_documents([Document(page_content=c) for c in chunks])

def add_nodes(graph_documents):
    get_kg().add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

def load_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    session = driver.session()
    net = Network(bgcolor="#222222", font_color="white", height="600px", width="100%")
    for r in session.run("MATCH (n) RETURN n"):
        node = r["n"]
        node_id = node.element_id
        node_label = list(node.labels)[0] if node.labels else "Unknown"
        node_name = node.get("id", str(node_id))
        net.add_node(node_id, label=node_name, color="#FF5733")
    for r in session.run("MATCH (n)-[r]->(m) RETURN n, r, m"):
        net.add_edge(r["n"].element_id, r["m"].element_id, title=r["r"].type, color="#33C1FF")
    session.close()
    html_file = "graph.html"
    net.write_html(html_file)
    return html_file

def generate_cypher_query(question, llm):
    entity_prompt = f"""Extract key entities from: \"{question}\". Return only a comma-separated list."""
    try:
        entities = [e.strip() for e in llm.invoke(entity_prompt).content.split(",")]
        basic_query = f"""MATCH (n) WHERE {" OR ".join([f'n.id CONTAINS "{e}"' for e in entities if e])} RETURN n LIMIT 10"""
        property_query = f"""MATCH (n) WHERE {" OR ".join([f'ANY(prop IN keys(n) WHERE n[prop] CONTAINS "{e}")' for e in entities if e])} RETURN n LIMIT 10"""
        relationship_query = f"""MATCH (n)-[r]-(m) WHERE {" OR ".join([f'n.id CONTAINS "{e}" OR m.id CONTAINS "{e}"' for e in entities if e])} RETURN n, r, m LIMIT 15"""
        path_query = f"""MATCH path=(n)-[*1..3]-(m) WHERE {" OR ".join([f'n.id CONTAINS "{e}"' for e in entities if e])} RETURN path LIMIT 5"""
        custom_query = llm.invoke(f"""Write a Cypher query for: \"{question}\". Return only the query.""").content
        return {
            "entities": entities,
            "basic_query": basic_query,
            "property_query": property_query,
            "relationship_query": relationship_query,
            "path_query": path_query,
            "custom_query": re.sub(r"```(?:cypher|sql)?", "", custom_query).strip("` \n")
        }
    except Exception as e:
        st.error(f"Query generation error: {str(e)}")
        return {"entities": [], "basic_query": "MATCH (n) RETURN n LIMIT 5", "custom_query": "MATCH (n) RETURN n LIMIT 5"}

def process_document(document_text, llm, progress_bar=None):
    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(build_vector_index, document_text)
        try:
            del_nodes()
            graph_docs = doc2graph(document_text, llm)
            add_nodes(graph_docs)
            results["graph_html"] = load_graph()
            if progress_bar: progress_bar.progress(0.8, text="Knowledge graph created")
        except Exception as e:
            st.error(f"Graph creation error: {str(e)}")
            results["graph_error"] = str(e)
        try:
            results["retriever"] = vector_future.result()
            if progress_bar: progress_bar.progress(1.0, text="Processing complete!")
        except Exception as e:
            st.error(f"Vector DB error: {str(e)}")
            results["vector_error"] = str(e)
    return results

def main():
    st.title("Chatbot with Knowledge Graph & Vector Search")
    st.session_state.setdefault("graph_html_file", None)
    st.session_state.setdefault("retriever", None)
    st.session_state.setdefault("qa_chain", None)
    st.session_state.setdefault("llm", initialize_llm())
    st.session_state.setdefault("document_text", None)
    st.session_state.setdefault("processing", False)
    st.session_state.setdefault("source_type", None)

    source_type = st.radio("Select source type", ["PDF Document", "Web URL"])
    if source_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and not st.session_state.processing and st.button("Process PDF"):
            st.session_state.processing = True
            st.session_state.source_type = "pdf"
            progress_bar = st.progress(0, text="Starting processing...")
            pdf_text = process_pdf(uploaded_file)
            st.session_state.document_text = pdf_text
            progress_bar.progress(0.2, text="PDF extracted")
            results = process_document(pdf_text, st.session_state.llm, progress_bar)
            if "retriever" in results:
                st.session_state.retriever = results["retriever"]
                st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=st.session_state.llm, retriever=results["retriever"])
            if "graph_html" in results:
                st.session_state.graph_html_file = results["graph_html"]
            st.session_state.processing = False
            st.rerun()
    else:
        url = st.text_input("Enter Website URL", "https://example.com")
        if url and not st.session_state.processing and st.button("Process URL"):
            st.session_state.processing = True
            st.session_state.source_type = "url"
            progress_bar = st.progress(0, text="Starting processing...")
            website_text = scrape_website(url)
            if website_text:
                st.session_state.document_text = website_text
                progress_bar.progress(0.2, text="Website content extracted")
                results = process_document(website_text, st.session_state.llm, progress_bar)
                if "retriever" in results:
                    st.session_state.retriever = results["retriever"]
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=st.session_state.llm, retriever=results["retriever"])
                if "graph_html" in results:
                    st.session_state.graph_html_file = results["graph_html"]
            st.session_state.processing = False
            st.rerun()

    tab1, tab2 = st.tabs(["Knowledge Graph", "Question Answering"])
    with tab1:
        if st.session_state.processing:
            st.info("Processing document... please wait")
        elif st.session_state.graph_html_file:
            with open(st.session_state.graph_html_file, "r") as f:
                st.components.v1.html(f.read(), height=600)
        else:
            st.info("Upload and process a document to visualize the knowledge graph")

    with tab2:
        st.subheader("Ask a question")
        user_query = st.text_input("Enter question")
        if user_query and st.session_state.qa_chain:
            with st.spinner("Searching for answer..."):
                response = st.session_state.qa_chain.run(user_query)
                cypher_data = generate_cypher_query(user_query, st.session_state.llm)
                st.write(response)
                st.write("### Detected Entities")
                st.write(", ".join(cypher_data["entities"]))
                query_tabs = st.tabs(["LLM Custom Query", "Basic Query", "Property Query", "Relationship Query", "Path Query"])
                st.code(cypher_data["custom_query"], language="cypher")
                st.code(cypher_data["basic_query"], language="cypher")
                st.code(cypher_data["property_query"], language="cypher")
                st.code(cypher_data["relationship_query"], language="cypher")
                st.code(cypher_data["path_query"], language="cypher")
        elif user_query:
            st.warning("Please process a document first")

if __name__ == "__main__":
    main()
