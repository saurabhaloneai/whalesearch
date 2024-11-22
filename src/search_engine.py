# search_engine.py
import os
from typing import List, Dict
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from serpapi import GoogleSearch

class SearchEngine:
    def __init__(self):
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        
    def _init_llm(self):
        return LlamaCpp(
            model_path="Phi-3-mini-4k-instruct-fp16.gguf",
            n_gpu_layers=-1,
            max_tokens=500,
            n_ctx=2048,
            seed=42,
            verbose=False
        )
    
    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name='thenlper/gte-small'
        )
    
    def get_serp_results(self, query: str, num_results: int = 5) -> List[Document]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": num_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        documents = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                doc = Document(
                    page_content=f"Title: {result.get('title', '')}\n\nSnippet: {result.get('snippet', '')}\n\nContent: {result.get('content', '')}",
                    metadata={
                        "source": result.get("link", ""),
                        "title": result.get("title", ""),
                        "position": result.get("position", 0)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def create_vector_db(self, documents: List[Document]) -> Chroma:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        splits = text_splitter.split_documents(documents)
        
        return Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def setup_rag(self, db: Chroma) -> RetrievalQA:
        template = """<|user|>
        Relevant information:
        {context}

        Question: {question}
        
        Please provide a detailed answer using the above information. Include relevant citations.
        <|end|>
        <|assistant|>"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def search_and_answer(self, query: str) -> Dict:
        search_results = self.get_serp_results(query)
        db = self.create_vector_db(search_results)
        rag_chain = self.setup_rag(db)
        response = rag_chain.invoke(query)
        
        sources = [{
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("source", ""),
            "position": doc.metadata.get("position", 0)
        } for doc in response["source_documents"]]
        
        return {
            "answer": response["result"],
            "sources": sources
        }