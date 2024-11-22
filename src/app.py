# ui.py
from typing import Tuple
import gradio as gr
from whalesearch.src.search_engine import SearchEngine
import os

class SearchEngineUI:
    def __init__(self):
        self.search_engine = SearchEngine()
    
    def process_query(self, query: str, api_key: str) -> Tuple[str, str]:
        os.environ["SERPAPI_API_KEY"] = api_key
        yield "Processing your query...", ""
        
        results = self.search_engine.search_and_answer(query)
        
        sources_text = "\n\nSources:\n"
        for idx, source in enumerate(results["sources"], 1):
            sources_text += f"{idx}. [{source['title']}]({source['url']})\n"
        
        yield results["answer"], sources_text

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# WhaleSearch 🐳", elem_id="app-title")

            with gr.Row():
                with gr.Column():
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
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
            
            search_button.click(
                fn=self.process_query,
                inputs=[query_input, api_key],
                outputs=[answer_output, sources_output],
                show_progress=True
            )
            
            gr.Examples(
                examples=[
                    ["what are the latest developments in octopus recepie ?"],
                    ["explain the basics of machine learning like jake paul's opponent i mean 58 year old"],
                    ["what is the current state of internally achieved agi?"]
                ],
                inputs=query_input
            )
            
            gr.Markdown("""
            ### Notes:
            - you need a valid SerpAPI key to use this search engine
            - results may take a few seconds to generate, completly depends on your hardware 
            - sources are provided with links to original content
            """)
        
        return interface

if __name__ == "__main__":
    ui = SearchEngineUI()
    ui.create_interface().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )