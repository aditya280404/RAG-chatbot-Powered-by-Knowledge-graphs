

This project is an end-to-end system that transforms unstructured data into a structured and interactive **knowledge graph**, enabling users to ask natural language questions and receive meaningful answers extracted directly from their data. Built using **Streamlit** for the front-end, **Neo4j** for graph storage, and **Groq LLaMA-3** models via **LangChain**, this app seamlessly integrates document processing, graph construction, and intelligent querying into a single workflow.



The core motivation behind this project is to **automate the process of extracting insights from textual data**—be it research papers, technical documentation, or web content—by converting it into an interactive, queryable knowledge graph. Traditional document search is often keyword-based and linear. This app introduces **semantic understanding and contextual relationships** by utilizing large language models (LLMs) to convert text into a structured graph format.

This project builds on my interests in **LLM-based automation**, **knowledge graph reasoning**, and **real-world document QA**, all of which are valuable in domains like biomedical research, education, transportation data, and digital governance. It’s especially useful when you need to quickly understand the entities, relationships, and facts embedded within complex documentation.


The user uploads a PDF document or enters a web URL. The system reads the content, breaks it into chunks, and uses a Groq-hosted LLaMA-3 model to identify entities and relationships. These are then stored in a **Neo4j** graph database. A PyVis-powered visualization lets the user explore the graph interactively. The real power lies in the chatbot interface: users can ask questions in natural language, and the app generates **Cypher queries** dynamically using LLMs, executes them on the graph, and translates the results into human-friendly answers.


- Supports PDF and website content as data sources  
- LLM-powered graph construction using LangChain’s experimental graph transformers  
- Neo4j integration for scalable graph storage and querying  
- Dynamic Cypher query generation based on user questions  
- Interactive PyVis network visualization embedded in Streamlit  
- Natural language answer generation from graph query results  





This project demonstrates how we can combine **graph databases**, **LLMs**, and **semantic parsing** to turn passive data into actionable knowledge. It reflects my continuing exploration into **retrieval-augmented generation (RAG)**, graph-based AI, and real-time data understanding.
