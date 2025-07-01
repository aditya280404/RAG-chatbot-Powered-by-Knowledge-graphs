from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit as st
import os
from dotenv import load_dotenv
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
import re


load_dotenv()

NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USERNAME = st.secrets["neo4j"]["username"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]
NEO4J_DATABASE = st.secrets["neo4j"]["database"]
groq_api_key = st.secrets["groq"]["api_key"]
LANGCHAIN_API_KEY = st.secrets["langchain"]["api_key"]
# os.environ["LANGCHAIN_TRACING_V2"] = "true"



from langchain_groq import ChatGroq

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
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")



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
    file_content = ""
    for page in pdf_file.pages:
        file_content += page.extract_text()
    return file_content


@st.cache_data
def scrape_website(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        for script in soup(["script", "style"]):
            script.extract()
            
        
        text = soup.get_text()
        
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error scraping website: {str(e)}")
        return None


def build_vector_index(text):
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    embedding = JinaEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def del_nodes():
    kg = get_kg()
    kg.query("MATCH (n) DETACH DELETE n")

def doc2graph(processed_text, llm):
    from langchain.text_splitter import TokenTextSplitter
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_core.documents import Document
    
    
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.split_text(processed_text)
    
    if len(chunks) > 30:
        st.warning(f"Document is large ({len(chunks)} chunks). Processing a subset for graph visualization.")
        chunks = chunks[:30]  
    
    transformer = LLMGraphTransformer(llm=llm)
    return transformer.convert_to_graph_documents([Document(page_content=c) for c in chunks])

def add_nodes(graph_documents):
    kg = get_kg()
    kg.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


def load_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    session = driver.session()
    net = Network(bgcolor="#222222", font_color="white", height="600px", width="100%")

    nodes = session.run("MATCH (n) RETURN n")
    for r in nodes:
        node = r['n']
        node_id = node.element_id
        node_label = list(node.labels)[0] if node.labels else "Unknown"
        node_name = node.get('id', str(node_id))
        net.add_node(node_id, label=node_name, color="#FF5733")

    rels = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
    for r in rels:
        net.add_edge(r['n'].element_id, r['m'].element_id, title=r['r'].type, color="#33C1FF")

    session.close()
    html_file = "graph.html"
    net.write_html(html_file)
    return html_file


def generate_cypher_query(question, llm):
    
    entity_prompt = f"""
    Extract the key entities, topics, and concepts from this question:
    
    "{question}"
    
    Return ONLY the entities as a comma-separated list, with no additional text or explanation.
    """
    
    try:
        entities_result = llm.invoke(entity_prompt).content
        entities = [e.strip() for e in entities_result.split(",")]
        
        
        basic_query = f"""
        // Basic entity matching
        MATCH (n)
        WHERE {" OR ".join([f'n.id CONTAINS "{entity}"' for entity in entities if entity])}
        RETURN n LIMIT 10
        """
        
        property_query = f"""
        // Property matching
        MATCH (n)
        WHERE 
        {" OR ".join([f'ANY(prop IN keys(n) WHERE n[prop] CONTAINS "{entity}")' for entity in entities if entity])}
        RETURN n LIMIT 10
        """
        
        relationship_query = f"""
        // Relationship exploration
        MATCH (n)-[r]-(m)
        WHERE 
        {" OR ".join([f'n.id CONTAINS "{entity}" OR m.id CONTAINS "{entity}"' for entity in entities if entity])}
        OR {" OR ".join([f'type(r) CONTAINS "{entity}"' for entity in entities if entity])}
        RETURN n, r, m LIMIT 15
        """
        
        path_query = f"""
        // Path exploration
        MATCH path = (n)-[*1..3]-(m)
        WHERE 
        {" OR ".join([f'n.id CONTAINS "{entity}"' for entity in entities if entity])}
        AND {" OR ".join([f'm.id CONTAINS "{entities[-1] if entities else ""}"' if entities else 'TRUE'])}
        RETURN path LIMIT 5
        """
        
        query_prompt = f"""
        Based on the question: "{question}" 
        and the identified entities: {entities},
        
        Create a Neo4j Cypher query that would find relevant information in a knowledge graph.
        The query should be appropriate for the complexity of the question.
        
        Return ONLY the Cypher query with no explanation.
        """
        
        custom_query = llm.invoke(query_prompt).content
        
        custom_query = re.sub(r'```cypher|```|```sql', '', custom_query).strip()
        
        return {
            "entities": entities,
            "basic_query": basic_query,
            "property_query": property_query, 
            "relationship_query": relationship_query,
            "path_query": path_query,
            "custom_query": custom_query
        }
    
    except Exception as e:
        st.error(f"Error generating Cypher query: {str(e)}")
        return {
            "entities": [],
            "basic_query": "MATCH (n) RETURN n LIMIT 5",
            "custom_query": "MATCH (n) RETURN n LIMIT 5"
        }


def process_document(document_text, llm, progress_bar=None):
    results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        
        vector_future = executor.submit(build_vector_index, document_text)
        
        
        try:
            
            del_nodes()
            
            graph_docs = doc2graph(document_text, llm)
            
            add_nodes(graph_docs)
        
            graph_html = load_graph()
            results["graph_html"] = graph_html
            if progress_bar:
                progress_bar.progress(0.8, text="Knowledge graph created")
        except Exception as e:
            st.error(f"Error creating knowledge graph: {str(e)}")
            results["graph_error"] = str(e)
        
        
        try:
            retriever = vector_future.result()
            results["retriever"] = retriever
            if progress_bar:
                progress_bar.progress(1.0, text="Processing complete!")
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            results["vector_error"] = str(e)
    
    return results


def main():
    st.title("Chatbot with Knowledge Graph & Vector Search")

    
    if "graph_html_file" not in st.session_state:
        st.session_state.graph_html_file = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "llm" not in st.session_state:
        st.session_state.llm = initialize_llm()
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "source_type" not in st.session_state:
        st.session_state.source_type = None


    source_type = st.radio("Select source type", ["PDF Document", "Web URL"])
    
    if source_type == "PDF Document":
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file and not st.session_state.processing:
            if st.button("Process PDF"):
                st.session_state.processing = True
                st.session_state.source_type = "pdf"
                
                
                progress_bar = st.progress(0, text="Starting processing...")
                
                
                pdf_text = process_pdf(uploaded_file)
                st.session_state.document_text = pdf_text
                progress_bar.progress(0.2, text="PDF extracted")
                
                
                results = process_document(pdf_text, st.session_state.llm, progress_bar)
                
                
                if "retriever" in results:
                    st.session_state.retriever = results["retriever"]
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=st.session_state.llm,
                        retriever=st.session_state.retriever
                    )
                
                if "graph_html" in results:
                    st.session_state.graph_html_file = results["graph_html"]
                
                st.session_state.processing = False
                st.rerun()
    
    else:  
        url = st.text_input("Enter Website URL", "https://example.com")
        if url and not st.session_state.processing:
            if st.button("Process URL"):
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
                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=st.session_state.llm,
                            retriever=st.session_state.retriever
                        )
                    
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
            try:
                with st.spinner("Searching for answer..."):
                    response = st.session_state.qa_chain.run(user_query)
                    
                    cypher_data = generate_cypher_query(user_query, st.session_state.llm)
                    
                    
                    st.write(response)
                    
                    
                    st.write("### Detected Entities")
                    st.write(", ".join(cypher_data["entities"]))
                    
                    
                    query_tabs = st.tabs(["LLM Custom Query", "Basic Query", "Property Query", 
                                          "Relationship Query", "Path Query"])
                    
                    with query_tabs[0]:
                        st.write("#### LLM-Generated Cypher Query")
                        st.code(cypher_data["custom_query"], language="cypher")
                        st.info("This query is generated based on the question but not executed.")
                    
                    with query_tabs[1]:
                        st.write("#### Basic Entity Matching")
                        st.code(cypher_data["basic_query"], language="cypher")
                    
                    with query_tabs[2]:
                        st.write("#### Property Matching")
                        st.code(cypher_data["property_query"], language="cypher")
                    
                    with query_tabs[3]:
                        st.write("#### Relationship Exploration")
                        st.code(cypher_data["relationship_query"], language="cypher")
                    
                    with query_tabs[4]:
                        st.write("#### Path Exploration")
                        st.code(cypher_data["path_query"], language="cypher")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif user_query:
            st.warning("Please process a document first")

if __name__ == "__main__":
    main()
      
