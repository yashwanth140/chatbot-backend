# mistral_engine.py

from llama_cpp import Llama

# Update this path to point to your downloaded model file
MODEL_PATH = "./mistral-7b-instruct-v0.2.Q8_0.gguf"

# Load Mistral model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,         # Context size (can tune later)
    n_threads=4,        # Use 6 CPU threads (adjust as per your CPU)
    verbose=True        # Show loading logs
)

def generate_response(prompt):
    response = llm(
        prompt=f"[INST] {prompt} [/INST]",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["</s>"]
    )
    return response["choices"][0]["text"].strip()
