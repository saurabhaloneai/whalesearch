# ui.py
from typing import Tuple
import gradio as gr
from whalesearch.src.search_engine import SearchEngine
import os



class SearchEngineUI:
    def __init__(self):
        self.search_engine = SearchEngine()
    
    def process_query(self, query: str, api_key: str, is_image_search: bool) -> Tuple[str, str]:
        os.environ["SERPAPI_API_KEY"] = api_key
        yield "Processing your query...", "", None
        
        results = self.search_engine.search_and_answer(query, is_image_search)
        
        sources_text = "\n\nSources:\n"
        gallery_images = []
        
        for idx, source in enumerate(results["sources"], 1):
            if results["is_image_search"]:
                sources_text += f"{idx}. [{source['title']}]({source['url']}) - [View Image]({source['image_url']})\n"
                if source['image_url']:
                    gallery_images.append(source['image_url'])
            else:
                sources_text += f"{idx}. [{source['title']}]({source['url']})\n"
        
        yield results["answer"], sources_text, gr.Gallery(value=gallery_images) if gallery_images else None
    
    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("# WhaleSearch 🐳", elem_id="app-title")
            
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Explore",
                        placeholder="Enter your search query here...",
                        lines=2
                    )
                    api_key = gr.Textbox(
                        label="SerpAPI Key",
                        placeholder="Enter your SerpAPI key",
                        type="password"
                    )
                    is_img_query = gr.Checkbox(label="Image Search")
                    search_button = gr.Button("Search", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
                    image_gallery = gr.Gallery(label="Image Results", visible=False)
            
            def update_gallery_visibility(is_image):
                return gr.Gallery(visible=is_image)
            
            is_img_query.change(
                fn=update_gallery_visibility,
                inputs=[is_img_query],
                outputs=[image_gallery]
            )
            
            search_button.click(
                fn=self.process_query,
                inputs=[query_input, api_key, is_img_query],
                outputs=[answer_output, sources_output, image_gallery],
                show_progress=True
            )
            
            gr.Examples(
                examples=[
                    ["What are the latest developments in octopus recipes?"],
                    ["Show me pictures of rare deep sea creatures"],
                    ["What is the current state of internally achieved AGI?"]
                ],
                inputs=query_input
            )
            
            gr.Markdown("""
            ### Notes:
            - You need a valid SerpAPI key to use this search engine
            - Results may take a few seconds to generate, completely depends on your hardware
            - Sources are provided with links to original content
            - Toggle 'Image Search' to include image results in your search
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