import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings.base import Embeddings
from langchain.chains import RetrievalQA
import torch
from transformers import AutoModel, AutoTokenizer

load_dotenv()
groq_api_key = os.getenv("groq_api_key")

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

TEST_CASES = [
    {
        "question": "How much capital did Drylab raise in their investment round?",
        "ground_truth": "Drylab raised 2.13 MNOK to match the 2.05 MNOK loan from Innovation Norway, with a total of 5 MNOK new capital including the development agreement with Filmlance International.",
        "key_entities": ["2.13 MNOK", "2.05 MNOK", "5 MNOK", "Innovation Norway", "Filmlance International"]
    },
    {
        "question": "What is the return customer rate for Drylab?",
        "ground_truth": "The return customer rate is 80%.",
        "key_entities": ["80%", "return customer rate"]
    },
    {
        "question": "Who joined Drylab as a mentor?",
        "ground_truth": "Caitlin Burns joined as a mentor, along with Oscar-winning VFX supervisor Dave Stump who joined earlier.",
        "key_entities": ["Caitlin Burns", "Dave Stump", "mentor", "VFX supervisor"]
    },
    {
        "question": "Where will the launch of Drylab 3.0 take place?",
        "ground_truth": "The launch of Drylab 3.0 will take place at the International Broadcasters Convention in Amsterdam in September.",
        "key_entities": ["Drylab 3.0", "International Broadcasters Convention", "Amsterdam", "September"]
    },
    {
        "question": "What is the revenue for the first four months compared to 2016?",
        "ground_truth": "Revenue for the first four months is 200 kNOK, compared to 339 kNOK for all of 2016.",
        "key_entities": ["200 kNOK", "339 kNOK", "first four months", "2016"]
    },
    {
        "question": "Which cities did Pontus and Audun visit in the US?",
        "ground_truth": "Pontus and Audun visited New York, St. Louis, San Francisco, and Los Angeles.",
        "key_entities": ["Pontus", "Audun", "New York", "St. Louis", "San Francisco", "Los Angeles"]
    },
    {
        "question": "Who are some of the new owners that joined the Drylab family?",
        "ground_truth": "New owners include Unni Jacobsen, Torstein Jahr, Suzanne Bolstad, Eivind Bergene, Turid Brun, Vigdis Trondsen, Lea Blindheim, Kristine Holmsen, Torstein Hansen, and Jostein Aanensen.",
        "key_entities": ["Unni Jacobsen", "Torstein Jahr", "Suzanne Bolstad", "new owners"]
    },
    {
        "question": "What happened at NAB and who did they meet?",
        "ground_truth": "Andreas and Audun traveled to NAB in Las Vegas where they met with PIX System (a competitor), DITs at the DIT-WIT party, Pomfort, Apple, ARRI, Teradek/Paralinx, Amazon, Google, and IBM.",
        "key_entities": ["NAB", "Las Vegas", "PIX System", "DIT-WIT", "Pomfort", "Apple"]
    },
    {
        "question": "Why did Drylab decide not to attend Cine Gear?",
        "ground_truth": "Drylab decided not to attend Cine Gear in L.A. because feedback from users about the show was mixed and their planned beta version of 3.0 was slightly delayed.",
        "key_entities": ["Cine Gear", "L.A.", "feedback", "mixed", "beta", "3.0", "delayed"]
    },
    {
        "question": "When will the Annual General Meeting be held?",
        "ground_truth": "Drylab's AGM will be held on June 16th at 15:00.",
        "key_entities": ["AGM", "June 16th", "15:00"]
    }
]


class JinaEmbeddings(Embeddings):
    def __init__(self):
        self.model, self.tokenizer = self.get_embedding_model()

    def get_embedding_model(self):
        model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
        return model, tokenizer

    def embed_documents(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def initialize_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")


def build_vector_index(content: str) -> VectorStoreRetriever:
    from langchain_text_splitters import TokenTextSplitter
    
    
    splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=64)
    chunks = splitter.split_text(content)
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    
    embedding = JinaEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    
    return retriever


def evaluate_faithfulness(answer: str, context: str, llm) -> float:
    """Evaluate if the answer is supported by the context."""
    prompt = f"""
    You are an expert evaluator for question-answering systems.
    
    Given the retrieved context and the answer provided by a QA system, determine if the answer is fully supported by the context.
    
    Context: {context}
    
    Answer: {answer}
    
    Rate the faithfulness on a scale of 0 to 1, where:
    - 0: The answer contains information NOT present in the context or contradicts the context
    - 0.5: The answer is partially supported by the context but includes unsupported claims
    - 1: The answer is fully supported by the context with no unsupported claims
    
    Return only the numerical score without explanation.
    """
    
    response = llm.invoke(prompt).content
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)  # Ensure score is between 0 and 1
    except:
        print(f"Failed to parse faithfulness score: {response}")
        return 0.0

def evaluate_answer_relevancy(question: str, answer: str, llm) -> float:
    """Evaluate if the answer is relevant to the question."""
    prompt = f"""
    You are an expert evaluator for question-answering systems.
    
    Determine if the answer directly addresses the question asked.
    
    Question: {question}
    
    Answer: {answer}
    
    Rate the relevance on a scale of 0 to 1, where:
    - 0: The answer is completely unrelated to the question
    - 0.5: The answer is somewhat relevant but misses key aspects of the question
    - 1: The answer directly and completely addresses the question
    
    Return only the numerical score without explanation.
    """
    
    response = llm.invoke(prompt).content
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)
    except:
        print(f"Failed to parse relevancy score: {response}")
        return 0.0

def evaluate_context_precision(question: str, context: str, llm) -> float:
    """Evaluate if the retrieved context is relevant to the question."""
    prompt = f"""
    You are an expert evaluator for retrieval systems.
    
    Given a question and retrieved context, determine how much of the context is relevant to answering the question.
    
    Question: {question}
    
    Retrieved Context: {context}
    
    Rate the precision on a scale of 0 to 1, where:
    - 0: None of the retrieved context is relevant to the question
    - 0.5: About half of the retrieved context is relevant
    - 1: All of the retrieved context is relevant to the question
    
    Return only the numerical score without explanation.
    """
    
    response = llm.invoke(prompt).content
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)
    except:
        print(f"Failed to parse precision score: {response}")
        return 0.0

def evaluate_context_recall(question: str, ground_truth: str, context: str, llm) -> float:
    """Evaluate if the context contains the information needed to answer the question."""
    prompt = f"""
    You are an expert evaluator for retrieval systems.
    
    Given a question, the ground truth answer, and retrieved context, determine if the context contains all the information needed to formulate the ground truth answer.
    
    Question: {question}
    
    Ground Truth Answer: {ground_truth}
    
    Retrieved Context: {context}
    
    Rate the recall on a scale of 0 to 1, where:
    - 0: The context contains none of the information needed for the ground truth
    - 0.5: The context contains some but not all key information
    - 1: The context contains all information needed to formulate the ground truth
    
    Return only the numerical score without explanation.
    """
    
    response = llm.invoke(prompt).content
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)
    except:
        print(f"Failed to parse recall score: {response}")
        return 0.0


def evaluate_answer_correctness(answer: str, ground_truth: str, key_entities: List[str], llm) -> float:
    """Evaluate if the answer is factually correct compared to the ground truth."""
    entity_str = ", ".join(key_entities)
    
    prompt = f"""
    You are an expert evaluator for question-answering systems.
    
    Compare the system's answer to the ground truth, focusing especially on these key entities/facts that should be present: {entity_str}
    
    System Answer: {answer}
    
    Ground Truth Answer: {ground_truth}
    
    Rate the correctness on a scale of 0 to 1, where:
    - 0: The answer is completely incorrect or contradicts the ground truth
    - 0.5: The answer has some correct elements but misses or incorrectly states key facts
    - 1: The answer captures all key facts correctly
    
    Return only the numerical score without explanation.
    """
    
    response = llm.invoke(prompt).content
    try:
        score = float(response.strip())
        return min(max(score, 0), 1)
    except:
        print(f"Failed to parse correctness score: {response}")
        return 0.0


def evaluate_rag_system(content: str):
   
    llm = initialize_llm()
    retriever = build_vector_index(content)
    
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )
    
    
    results = {metric: [] for metric in METRICS}
    results["questions"] = []
    results["answers"] = []
    results["ground_truth"] = []
    results["contexts"] = []
    
    
    for i, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]
        key_entities = test_case["key_entities"]
        
        print(f"\nEvaluating Q{i+1}: {question}")
        
        
        result = qa_chain.invoke(question)
        answer = result["result"]
        context = " ".join([doc.page_content for doc in retriever.get_relevant_documents(question)])
        
        
        results["questions"].append(question)
        results["answers"].append(answer)
        results["ground_truth"].append(ground_truth)
        results["contexts"].append(context)
        
        
        print("Evaluating faithfulness...")
        faithfulness = evaluate_faithfulness(answer, context, llm)
        results["faithfulness"].append(faithfulness)
        
        print("Evaluating answer relevancy...")
        relevancy = evaluate_answer_relevancy(question, answer, llm)
        results["answer_relevancy"].append(relevancy)
        
        print("Evaluating context precision...")
        precision = evaluate_context_precision(question, context, llm)
        results["context_precision"].append(precision)
        
        print("Evaluating context recall...")
        recall = evaluate_context_recall(question, ground_truth, context, llm)
        results["context_recall"].append(recall)
        
        print("Evaluating answer correctness...")
        correctness = evaluate_answer_correctness(answer, ground_truth, key_entities, llm)
        results["answer_correctness"].append(correctness)
        
        print(f"Q{i+1} Scores: Faithfulness={faithfulness:.2f}, Relevancy={relevancy:.2f}, "
              f"Precision={precision:.2f}, Recall={recall:.2f}, Correctness={correctness:.2f}")
    
    
    avg_results = {metric: np.mean(results[metric]) for metric in METRICS}
    
    
    print("\n===== EVALUATION SUMMARY =====")
    for metric, score in avg_results.items():
        print(f"{metric}: {score:.3f}")
    
    
    with open("rag_evaluation_results.json", "w") as f:
        json.dump({
            "test_cases": [
                {
                    "question": q,
                    "ground_truth": gt,
                    "answer": a,
                    "metrics": {
                        metric: results[metric][i] for metric in METRICS
                    }
                }
                for i, (q, gt, a) in enumerate(zip(
                    results["questions"], 
                    results["ground_truth"], 
                    results["answers"]
                ))
            ],
            "averages": avg_results
        }, f, indent=2)
    
    return avg_results

def read_pdf(pdf_path):
    from PyPDF2 import PdfReader
    
    pdf_file = PdfReader(pdf_path)
    text = ""
    for page in pdf_file.pages:
        text += page.extract_text()
    return text


if __name__ == "__main__":
    
    pdf_path = "drylab.pdf"
    
    
    pdf_content = read_pdf(pdf_path)
    
 
    results = evaluate_rag_system(pdf_content)
    
    print("\nEvaluation complete! Results saved to rag_evaluation_results.json")