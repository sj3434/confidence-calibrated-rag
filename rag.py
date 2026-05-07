import os
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

class ConfidenceCalibratedRAG:
    def __init__(self, data_path: str, model_name="gpt-4o-mini", default_threshold=0.7):
        self.threshold = default_threshold
        self.is_calibrated = False
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
            
        loader = TextLoader(data_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0, separator="\n")
        docs = text_splitter.split_documents(documents)
        
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Answer the question based ONLY on the provided context. If the context does not contain the answer, say 'I don't know'."),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}")
        ])
        
        self.judge_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a strict evaluation judge. Review the provided context and the question. Does the context contain sufficient information to answer the question? Reply with only 'YES' or 'NO'."),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}")
        ])
        
    def calibrate(self, calibration_queries):
        """
        Algorithmic Innovation: Auto-calibrates the similarity threshold based on a small set 
        of known in-domain and out-of-domain queries to maximize abstention accuracy.
        """
        print("Calibrating abstention threshold...")
        scores_in_domain = []
        scores_out_domain = []
        
        for q_type, q_text in calibration_queries:
            results = self.vectorstore.similarity_search_with_relevance_scores(q_text, k=1)
            best_score = results[0][1] if results else 0
            if q_type == "in":
                scores_in_domain.append(best_score)
            else:
                scores_out_domain.append(best_score)
                
        # Calculate optimal threshold (midpoint between min in-domain and max out-domain)
        min_in = np.min(scores_in_domain) if scores_in_domain else 0.8
        max_out = np.max(scores_out_domain) if scores_out_domain else 0.5
        
        if min_in > max_out:
            self.threshold = (min_in + max_out) / 2.0
        else:
            self.threshold = max_out + 0.05  # Fallback to slightly above out-of-domain noise
            
        self.is_calibrated = True
        print(f"Calibration complete. Optimal threshold set to: {self.threshold:.3f}")
        return self.threshold

    def retrieve_with_scores(self, question: str):
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=2)
        return results

    def query(self, question: str, enable_abstention: bool = True):
        results = self.retrieve_with_scores(question)
        if not results:
            return "I don't know (Abstained: No context found)", []
            
        context = "\n".join([doc.page_content for doc, score in results])
        
        if enable_abstention:
            best_score = results[0][1] if results else 0
            # Guardrail 1: Quantitative Dynamic Thresholding
            if best_score < self.threshold:
                return f"I don't know (Abstained: Max similarity score {best_score:.3f} < threshold {self.threshold:.3f})", results

            # Guardrail 2: Qualitative LLM Judge
            judge_chain = self.judge_prompt | self.llm | StrOutputParser()
            judge_response = judge_chain.invoke({"context": context, "question": question})
            
            if "NO" in judge_response.strip().upper():
                return "I don't know (Abstained: LLM Judge deemed context insufficient)", results
            
        qa_chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = qa_chain.invoke({"context": context, "question": question})
        return answer, results
