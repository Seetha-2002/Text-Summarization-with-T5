from transformers import T5ForConditionalGeneration, T5Tokenizer
import gradio as gr

class TextSummarizer:
    def __init__(self, model_name="t5-small", max_input_length=512):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_input_length = max_input_length

    def summarize(self, text, max_length=150):
        # Preprocess the input text
        input_text = "summarize: " + text
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=self.max_input_length, truncation=True)

        # Generate summary
        summary_ids = self.model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

def summarize_interface(text, max_length):
    summarizer = TextSummarizer()
    return summarizer.summarize(text, max_length=max_length)

# Define the Gradio interface
gr_interface = gr.Interface(
    fn=summarize_interface,
    inputs=["text", gr.Slider(50, 200, value=150, label="Max Summary Length")],
    outputs="text",
    title="Text Summarization",
    description="Enter text to summarize using the T5 model."
)

# Launch the app
if __name__ == "__main__":
    gr_interface.launch()
