TinyLlama Research Paper Implementation
Project Overview

This repository contains our implementation of the research-based TinyLlama-1.1B-Chat model as part of our academic assignment.
The objective of this project was to study, implement, and deploy a lightweight Large Language Model (LLM) based on the TinyLlama research paper and create an interactive chatbot interface using Gradio.

We successfully integrated the TinyLlama chat model into a local web-based interface and resolved practical implementation issues related to conversation handling and response generation.

👨‍💻 Team Members

This project was implemented by:

Muhammad Danish

Adila M. Aslam

Ali Ahmed

📄 Research Paper Credit

This implementation is based on the research work:

TinyLlama: An Open-Source Small Language Model
Developed by the TinyLlama Research Team.

Original Model Repository:
https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

All rights and original research contributions belong to the respective authors of the TinyLlama project.

⚙️ Implementation Notes

During implementation, we encountered a small issue related to chat turn handling and model self-conversation behavior.
This issue was resolved by correctly formatting chat prompts and managing assistant response boundaries.

The final working and stable implementation can be tested by running:

chat_gradio/app_update.py


💻 Installation & Setup
1️⃣ Clone the Repository
git clone <your-github-repo-link>
cd <repo-folder>

2️⃣ Create Virtual Environment (Optional but Recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/Mac

3️⃣ Install Required Dependencies
pip install torch
pip install transformers
pip install gradio


Note: The TinyLlama model will be automatically downloaded from HuggingFace when the script is first executed.

▶️ Running the Application

Navigate to the chat interface folder:

cd chat_gradio
python app_update.py


After running, Gradio will provide a local URL in the terminal, for example:

http://127.0.0.1:7860


Open this link in your browser to use the chatbot.

📦 Dependencies

Python 3.9

torch>=2.0

transformers>=4.35.0

gradio>=4.13.0


HuggingFace Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

🎓 Educational Purpose Disclaimer

This project is implemented strictly for educational and academic purposes.
We do not claim ownership of the TinyLlama model or its research contributions.
All model rights belong to the original researchers and respective license holders.

📚 Citation

If referencing the original TinyLlama work:

@article{tinyllama2023,
  title={TinyLlama: An Open-Source Small Language Model},
  author={TinyLlama Research Team},
  year={2023},
  url={https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0}
}

✅ Summary

Research Paper Studied: TinyLlama

Implementation: Local Chatbot using Gradio

Issue Handling: Fixed chat self-conversation bug

Purpose: Academic Submission
