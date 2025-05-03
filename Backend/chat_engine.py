# chat_engine.py

from llama_cpp import Llama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Mistral model
model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=6,
    n_batch=512,
    use_mlock=True
)

# Load vectorstore (resume knowledge)
embedding_model = HuggingFaceEmbeddings()
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)


def generate_response(user_query):
    """
    Generate a clean response using local Mistral model and resume knowledge.
    """
    try:
        # Step 1: Search resume for best matching info
        results = vectorstore.similarity_search(user_query, k=3)
        context_text = results[0].page_content if results else "No relevant resume information found."

        # Step 2: Strict instruction prompt
        prompt = (
            f"### Instruction:\n"
            f"Use ONLY the following resume information to answer the user's question.\n"
            f"DO NOT invent or add extra information.\n\n"
            f"Resume Information:\n{context_text}\n\n"
            f"User Question:\n{user_query}\n\n"
            f"### Response:\n"
        )

        # Step 3: Generate response
        output = llm(prompt, max_tokens=512, temperature=0.2, top_p=0.8, stop=["###"])
        response_text = output["choices"][0]["text"].strip()

        # Always return a string
        if not response_text:
            response_text = "I'm sorry, I couldn't find that information."

        return response_text

    except Exception as e:
        print(f"[ERROR] Failed to generate response: {e}")
        return "I'm sorry, something went wrong while processing your request."
