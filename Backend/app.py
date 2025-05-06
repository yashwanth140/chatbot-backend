from flask import Flask, request, jsonify
from flask_cors import CORS 
from chat_engine import generate_response

app = Flask(__name__)
CORS(app)  # Enables cross-origin requests

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message")
        reply = generate_response(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"[ERROR] Failed to generate chat reply: {e}")
        return jsonify({"reply": "Internal server error occurred."}), 500

@app.route("/")
def home():
    return jsonify({"message": "Mistral backend is running."})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
