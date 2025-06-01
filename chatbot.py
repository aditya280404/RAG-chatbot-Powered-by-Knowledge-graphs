from pyvis.network import Network
from neo4j import GraphDatabase
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.graphs import Neo4jGraph
import requests
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Any

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

groq_api_key = os.getenv("groq_api_key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

from langchain_groq import ChatGroq

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

def del_nodes():
    kg = get_kg()
    kg.query("MATCH (n) DETACH DELETE n")

def doc2graph(processed_text, llm):
    from langchain.text_splitter import TokenTextSplitter
    from langchain_experimental.graph_transformers import LLMGraphTransformer
    from langchain_core.documents import Document
    
    text_splitter = TokenTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = text_splitter.split_text(processed_text)
    
    if len(chunks) > 40:
        st.warning(f"Document is large ({len(chunks)} chunks). Processing first 40 chunks.")
        chunks = chunks[:40]
    
    transformer = LLMGraphTransformer(llm=llm)
    
    return transformer.convert_to_graph_documents([Document(page_content=c) for c in chunks])

def add_nodes(graph_documents):
    kg = get_kg()
    kg.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

def get_graph_schema():
    kg = get_kg()
    
    node_labels = kg.query("""
        CALL db.labels() YIELD label
        RETURN collect(label) as labels
    """)
    
    relationship_types = kg.query("""
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN collect(relationshipType) as types
    """)
    
    sample_entities = {}
    if node_labels and node_labels[0]['labels']:
        for label in node_labels[0]['labels']:
            try:
                samples = kg.query(f"""
                    MATCH (n:`{label}`)
                    RETURN n.id as entity
                    LIMIT 5
                """)
                sample_entities[label] = [s['entity'] for s in samples if s['entity']]
            except:
                continue
    
    return {
        'node_labels': node_labels[0]['labels'] if node_labels else [],
        'relationship_types': relationship_types[0]['types'] if relationship_types else [],
        'sample_entities': sample_entities
    }

def load_graph():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    session = driver.session()
    
    net = Network(
        bgcolor="#1e1e1e", 
        font_color="white", 
        height="700px", 
        width="100%",
        directed=True
    )
    
    nodes_query = session.run("MATCH (n) RETURN n, labels(n) as labels")
    
    unique_labels = set()
    node_data = []
    
    for record in nodes_query:
        node = record['n']
        labels = record['labels']
        node_data.append((node, labels))
        unique_labels.update(labels)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#85C1E9', '#F8C471']
    
    label_colors = {}
    for i, label in enumerate(unique_labels):
        label_colors[label] = colors[i % len(colors)]
    
    for node, labels in node_data:
        node_id = node.element_id
        primary_label = labels[0] if labels else "Unknown"
        node_name = node.get('id', str(node_id))
        color = label_colors.get(primary_label, "#FF5733")
        
        degree_query = f"MATCH (n)-[r]-() WHERE elementId(n) = '{node_id}' RETURN count(r) as degree"
        degree_result = session.run(degree_query)
        degree = list(degree_result)[0]['degree'] if degree_result else 0
        size = max(15, min(50, 15 + degree * 3))
        
        net.add_node(
            node_id, 
            label=node_name, 
            color=color, 
            size=size, 
            title=f"Type: {primary_label}\nConnections: {degree}"
        )

    rels = session.run("MATCH (n)-[r]->(m) RETURN n, r, m, type(r) as rel_type")
    for record in rels:
        rel_type = record['rel_type']
        net.add_edge(
            record['n'].element_id, 
            record['m'].element_id, 
            title=rel_type, 
            color="#66BB6A",
            width=2,
            arrows="to"
        )

    session.close()
    
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    html_file = "graph.html"
    net.write_html(html_file)
    return html_file

def generate_cypher_query(question: str, graph_schema: Dict, llm) -> str:
    schema_info = f"""
    Current Graph Schema:
    - Node Labels: {graph_schema['node_labels']}
    - Relationship Types: {graph_schema['relationship_types']}
    - Sample Entities: {graph_schema['sample_entities']}
    """
    
    prompt = f"""
    You are a Neo4j Cypher query expert. Based on the user's question and the current graph schema, generate an appropriate Cypher query.

    {schema_info}

    User Question: "{question}"

    Guidelines:
    1. Use the actual node labels and relationship types from the schema
    2. Use appropriate WHERE conditions to filter based on the question
    3. Use CONTAINS, regex (=~), or exact matches as appropriate
    4. Return relevant information based on what the user is asking
    5. Limit results to avoid overwhelming output (LIMIT 20 or less)
    6. Use case-insensitive matching where appropriate: =~ '(?i).*keyword.*'
    7. Consider using multiple MATCH patterns if needed
    8. Return node properties, relationships, and paths as relevant

    Return ONLY the Cypher query, no explanation or formatting.
    """
    
    try:
        response = llm.invoke(prompt)
        query = response.content.strip()
        
        query = re.sub(r'```cypher|```sql|```|`', '', query).strip()
        
        return query
    except Exception as e:
        st.error(f"Error generating Cypher query: {str(e)}")
        return "MATCH (n) RETURN n LIMIT 10"

def execute_cypher_query(query: str) -> List[Dict]:
    try:
        kg = get_kg()
        results = kg.query(query)
        return results
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return []

def generate_answer_from_results(question: str, query: str, results: List[Dict], llm) -> str:
    if not results:
        return "I couldn't find any relevant information in the knowledge graph to answer your question. The graph might not contain data related to your query, or you might want to rephrase your question."
    
    results_text = []
    for i, result in enumerate(results[:15]):
        results_text.append(f"Result {i+1}: {result}")
    
    results_summary = "\n".join(results_text)
    
    prompt = f"""
    Based on the following query results from a knowledge graph, provide a comprehensive and natural answer to the user's question.

    User Question: "{question}"
    
    Cypher Query Used: {query}
    
    Query Results:
    {results_summary}

    Instructions:
    1. Provide a direct, comprehensive answer to the user's question
    2. Use the specific information from the query results
    3. Organize the information logically
    4. If relationships are found, explain them clearly
    5. Be specific and mention actual entities, relationships, and properties
    6. If the results don't fully answer the question, mention what information is available
    7. Make the answer conversational and easy to understand

    Answer:
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nRaw results: Found {len(results)} records in the knowledge graph."

def process_document(document_text, llm, progress_bar=None):
    results = {}
    
    try:
        if progress_bar:
            progress_bar.progress(0.1, text="Clearing existing graph...")
        
        del_nodes()
        
        if progress_bar:
            progress_bar.progress(0.3, text="Extracting entities and relationships...")
        
        graph_docs = doc2graph(document_text, llm)
        
        if progress_bar:
            progress_bar.progress(0.6, text="Building knowledge graph...")
        
        add_nodes(graph_docs)
        
        if progress_bar:
            progress_bar.progress(0.8, text="Generating visualization...")
        
        graph_html = load_graph()
        results["graph_html"] = graph_html
        
        kg = get_kg()
        node_count = kg.query("MATCH (n) RETURN count(n) as count")[0]["count"]
        rel_count = kg.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        
        results["stats"] = {
            "nodes": node_count,
            "relationships": rel_count
        }
        
        results["schema"] = get_graph_schema()
        
        if progress_bar:
            progress_bar.progress(1.0, text="Processing complete!")
            
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        results["error"] = str(e)
    
    return results

def main():
    st.set_page_config(page_title="Flexible KG Chatbot", layout="wide")
    st.title("🤖 Flexible Knowledge Graph Chatbot")
    st.markdown("*Dynamically discovers entities and generates queries from your data*")

    if "graph_html_file" not in st.session_state:
        st.session_state.graph_html_file = None
    if "llm" not in st.session_state:
        st.session_state.llm = initialize_llm()
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "graph_stats" not in st.session_state:
        st.session_state.graph_stats = None
    if "graph_schema" not in st.session_state:
        st.session_state.graph_schema = None

    with st.sidebar:
        st.header("📄 Document Processing")
        
        source_type = st.radio("Select source type", ["PDF Document", "Web URL"])
        
        if source_type == "PDF Document":
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded_file and not st.session_state.processing:
                if st.button("🔄 Process PDF", type="primary"):
                    st.session_state.processing = True
                    
                    progress_bar = st.progress(0, text="Starting processing...")
                    
                    pdf_text = process_pdf(uploaded_file)
                    st.session_state.document_text = pdf_text
                    
                    results = process_document(pdf_text, st.session_state.llm, progress_bar)
                    
                    if "graph_html" in results:
                        st.session_state.graph_html_file = results["graph_html"]
                    
                    if "stats" in results:
                        st.session_state.graph_stats = results["stats"]
                    
                    if "schema" in results:
                        st.session_state.graph_schema = results["schema"]
                    
                    st.session_state.processing = False
                    st.rerun()
        
        else:
            url = st.text_input("Enter Website URL", placeholder="https://example.com")
            if url and not st.session_state.processing:
                if st.button("🔄 Process URL", type="primary"):
                    st.session_state.processing = True
                    
                    progress_bar = st.progress(0, text="Starting processing...")
                    
                    website_text = scrape_website(url)
                    if website_text:
                        st.session_state.document_text = website_text
                        
                        results = process_document(website_text, st.session_state.llm, progress_bar)
                        
                        if "graph_html" in results:
                            st.session_state.graph_html_file = results["graph_html"]
                        
                        if "stats" in results:
                            st.session_state.graph_stats = results["stats"]
                        
                        if "schema" in results:
                            st.session_state.graph_schema = results["schema"]
                    
                    st.session_state.processing = False
                    st.rerun()
        
        if st.session_state.graph_stats:
            st.success("✅ Knowledge Graph Ready!")
            st.metric("Nodes", st.session_state.graph_stats["nodes"])
            st.metric("Relationships", st.session_state.graph_stats["relationships"])
            
            if st.session_state.graph_schema:
                with st.expander("📊 Discovered Schema"):
                    st.write("**Node Types:**")
                    st.write(", ".join(st.session_state.graph_schema['node_labels']))
                    st.write("**Relationship Types:**")  
                    st.write(", ".join(st.session_state.graph_schema['relationship_types']))

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🧠 Knowledge Graph Visualization")
        if st.session_state.processing:
            st.info("🔄 Processing document... please wait")
        elif st.session_state.graph_html_file:
            with open(st.session_state.graph_html_file, "r") as f:
                st.components.v1.html(f.read(), height=700)
        else:
            st.info("📤 Upload and process a document to visualize the knowledge graph")
    
    with col2:
        st.subheader("💬 Ask Questions")
        
        if st.session_state.graph_stats and st.session_state.graph_stats["nodes"] > 0:
            user_query = st.text_area(
                "Enter your question:", 
                placeholder="Ask anything about your data...\ne.g., What are the main entities?\nHow are X and Y related?\nShow me all connections to Z",
                height=100
            )
            
            if st.button("🔍 Get Answer", type="primary") and user_query:
                with st.spinner("Generating query and searching..."):
                    try:
                        cypher_query = generate_cypher_query(
                            user_query, 
                            st.session_state.graph_schema, 
                            st.session_state.llm
                        )
                        
                        st.write("**Generated Cypher Query:**")
                        st.code(cypher_query, language="cypher")
                        
                        results = execute_cypher_query(cypher_query)
                        
                        answer = generate_answer_from_results(
                            user_query, 
                            cypher_query, 
                            results, 
                            st.session_state.llm
                        )
                        
                        st.write("**Answer:**")
                        st.write(answer)
                        
                        if results:
                            with st.expander(f"📋 Raw Results ({len(results)} records)"):
                                for i, result in enumerate(results[:10]):
                                    st.json(result)
                        
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
            
            if st.session_state.graph_schema and st.session_state.graph_schema['sample_entities']:
                st.write("**Quick Examples:**")
                sample_entities = st.session_state.graph_schema['sample_entities']
                
                example_questions = []
                for label, entities in sample_entities.items():
                    if entities:
                        example_questions.append(f"Tell me about {entities[0]}")
                        example_questions.append(f"What is related to {entities[0]}?")
                
                for i, example in enumerate(example_questions[:4]):
                    if st.button(f"💡 {example}", key=f"example_{i}"):
                        st.session_state.example_query = example
                        st.rerun()
        
        else:
            st.info("📥 Process a document first to start asking questions")

if __name__ == "__main__":
    main()
