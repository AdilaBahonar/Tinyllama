

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------- Prompt Formatter (your original stable version) --------
def format_prompt(message, history):
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>:\n{user_msg}\n<|assistant|>:\n{bot_msg}</s>\n"
    prompt += f"<|user|>:\n{message}\n<|assistant|>:"
    return prompt

# -------- Chat Function with Streaming --------
def chat(message, history):
    prompt = format_prompt(message, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

    # Run generation in background thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    reply = ""

    # Stream tokens
    for new_text in streamer:
        partial_text += new_text

        # Extract assistant reply exactly like your original code
        if "<|assistant|>:" in partial_text:
            reply = partial_text.split("<|assistant|>:")[-1]
            if "</s>" in reply:
                reply = reply.split("</s>")[0]
        else:
            reply = partial_text

        yield reply.strip()   # Live streaming output

    # Final reply stored in chat history
    return reply.strip()

# -------- Launch Gradio UI --------
gr.ChatInterface(
    fn=chat,
    title="🦙 TinyLlama ChatBot (Streaming)",
    description="TinyLlama 1.1B — Stable Streaming Chat (No Self-Conversation)",
    examples=[
        "What is Python?",
        "Tell me a joke.",
        "Who is the president of USA?"
    ]
).launch()
