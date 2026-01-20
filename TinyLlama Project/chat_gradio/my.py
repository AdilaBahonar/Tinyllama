import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Format conversation history into prompt
def format_prompt(message, history):
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"<|user|>:\n{user_msg}\n<|assistant|>:\n{bot_msg}</s>\n"
    prompt += f"<|user|>:\n{message}\n<|assistant|>:"
    return prompt

# Generate response
def chat(message, history):
    prompt = format_prompt(message, history)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's new response
    if "<|assistant|>:" in decoded:
        reply = decoded.split("<|assistant|>:")[-1].strip()
        if "</s>" in reply:
            reply = reply.split("</s>")[0].strip()
    else:
        reply = decoded.strip()

    return reply

# Launch Gradio Chat Interface
gr.ChatInterface(fn=chat,
                 title="🦙 TinyLlama ChatBot",
                 description="Chat with TinyLlama 1.1B. Ask anything!",
                 examples=["What is Python?", "Tell me a joke.", "Who is the president of USA?"]
                ).launch()
