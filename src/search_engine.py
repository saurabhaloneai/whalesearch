import os
from typing import List, Dict, Optional
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
            temperature=0.7,
            top_p=0.95,
            seed=42,
            verbose=False
        )
    
    def _init_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name='thenlper/gte-small'
        )
    
    def get_serp_results(self, query: str, is_image_search: bool = False, num_results: int = 5) -> List[Document]:
        try:
            params = {
                "engine": "google_images" if is_image_search else "google",
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": num_results,
                "safe": "active",
                "hl": "en",
                **({"tbm": "isch"} if is_image_search else {})
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            documents = []
            if is_image_search and "images_results" in results:
                for result in results["images_results"]:
                    doc = Document(
                        page_content=f"Title: {result.get('title', '')}\n\nDescription: {result.get('snippet', '')}\n\nContent: Image URL: {result.get('original', '')}",
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "position": result.get("position", 0),
                            "image_url": result.get("original", ""),
                            "thumbnail": result.get("thumbnail", "")
                        }
                    )
                    documents.append(doc)
            elif not is_image_search and "organic_results" in results:
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
                    
            if not documents:
                # if no results were found hehehe
                doc = Document(
                    page_content=f"No results found for query: {query}",
                    metadata={
                        "source": "",
                        "title": "No Results",
                        "position": 0
                    }
                )
                documents.append(doc)
                
            return documents
            
        except Exception as e:
            # in case of API failures :)
            doc = Document(
                page_content=f"Error occurred while searching: {str(e)}",
                metadata={
                    "source": "",
                    "title": "Error",
                    "position": 0
                }
            )
            return [doc]
    
    def create_vector_db(self, documents: List[Document]) -> Optional[Chroma]:
        if not documents:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            return None
            
        return Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
    
    def setup_rag(self, db: Optional[Chroma], is_image_search: bool = False) -> Optional[RetrievalQA]:
        if db is None:
            return None
            
        if is_image_search:
            template = """<|user|>
            Relevant information including images:
            {context}
            Question: {question}
            
            Please provide a detailed answer using the above information. Include relevant citations and image descriptions where available.
            <|end|>
            <|assistant|>"""
        else:
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
            retriever=db.as_retriever(
                search_kwargs={
                    "k": 5,
                    "fetch_k": 8,
                    "score_threshold": 0.5
                }
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def search_and_answer(self, query: str, is_image_search: bool = False) -> Dict:
        search_results = self.get_serp_results(query, is_image_search)
        
        if not search_results:
            return {
                "answer": "No results found for your query.",
                "sources": [],
                "is_image_search": is_image_search
            }
        
        db = self.create_vector_db(search_results)
        if db is None:
            return {
                "answer": "Unable to process search results.",
                "sources": [],
                "is_image_search": is_image_search
            }
            
        rag_chain = self.setup_rag(db, is_image_search)
        if rag_chain is None:
            return {
                "answer": "Unable to generate response.",
                "sources": [],
                "is_image_search": is_image_search
            }
            
        try:
            response = rag_chain.invoke(query)
            
            sources = []
            for doc in response["source_documents"]:
                source_info = {
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("source", ""),
                    "position": doc.metadata.get("position", 0)
                }
                
                if is_image_search:
                    source_info["image_url"] = doc.metadata.get("image_url", "")
                    source_info["thumbnail"] = doc.metadata.get("thumbnail", "")
                
                sources.append(source_info)
            
            return {
                "answer": response["result"],
                "sources": sources,
                "is_image_search": is_image_search
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "is_image_search": is_image_search
            }
