import os
from typing import List, Dict, Tuple
import gradio as gr
from serpapi import GoogleSearch
from langchain_community.llms import LlamaCpp

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import time



class SearchEngine:
    def __init__(self):
        self.llm = self._init_llm()
        self.embeddings = self._init_embeddings()
        
    def _init_llm(self):
        """Initialize the LLM model"""
        return LlamaCpp(
            model_path="Phi-3-mini-4k-instruct-fp16.gguf",
            n_gpu_layers=-1,
            max_tokens=500,
            n_ctx=2048,
            seed=42,
            verbose=False
        )
    
    def _init_embeddings(self):
        """Initialize the embedding model"""
        return HuggingFaceEmbeddings(
            model_name='thenlper/gte-small'
        )
    
    def get_serp_results(self, query: str, num_results: int = 5) -> List[Document]:
        """Get search results from SerpAPI"""
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
        """Setup the RAG pipeline"""
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
        # Get search results
        search_results = self.get_serp_results(query)
        
        # Create vector DB
        db = self.create_vector_db(search_results)
        
        # Setup RAG
        rag_chain = self.setup_rag(db)
        
        # Get response
        response = rag_chain.invoke(query)
        
        # Format sources
        sources = [{
            "title": doc.metadata.get("title", ""),
            "url": doc.metadata.get("source", ""),
            "position": doc.metadata.get("position", 0)
        } for doc in response["source_documents"]]
        
        return {
            "answer": response["result"],
            "sources": sources
        }


class SearchEngineUI:
    def __init__(self):
        self.search_engine = SearchEngine()
    
    def process_query(self, query: str, api_key: str) -> Tuple[str, str]:
        # Set API key
        os.environ["SERPAPI_API_KEY"] = api_key
        
        # Show processing message
        yield "Processing your query...", ""
        
        # Get results
        results = self.search_engine.search_and_answer(query)
        
        # Format sources
        sources_text = "\n\nSources:\n"
        for idx, source in enumerate(results["sources"], 1):
            sources_text += f"{idx}. [{source['title']}]({source['url']})\n"
        
        # Return answer and sources
        yield results["answer"], sources_text

    def create_interface(self):

        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# WhaleSearch 🐳", elem_id="app-title")

            with gr.Row():
                with gr.Column():
                    # Input components
                    query_input = gr.Textbox(
                        label="explore ",
                        placeholder="enter your search query here...",
                        lines=2
                    )
                    api_key = gr.Textbox(
                        label="SerpAPI Key",
                        placeholder="enter your SerpAPI key",
                        type="password"
                    )
                    search_button = gr.Button("Search", variant="primary")
                
            with gr.Row():
                with gr.Column():
                    # Output components
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
            
            # Handle search button click
            search_button.click(
                fn=self.process_query,
                inputs=[query_input, api_key],
                outputs=[answer_output, sources_output],
                show_progress=True
            )
            
            # Example queries
            gr.Examples(
                examples=[
                    ["What are the latest developments in quantum computing?"],
                    ["Explain the basics of machine learning"],
                    ["What is the current state of renewable energy?"]
                ],
                inputs=query_input
            )
            
            # Footer
            gr.Markdown("""
            ### Notes:
            - you need a valid SerpAPI key to use this search engine
            - results may take a few seconds to generate, completly depends on your hardware 
            - sources are provided with links to original content
            """)
        
        return interface


def main():
    # Create UI instance
    ui = SearchEngineUI()
    
    # Launch the interface
    ui.create_interface().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()
