# Backend/app.py

from flask import Flask, request, jsonify
from chat_engine import generate_response

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message")
        reply = generate_response(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"[ERROR] Failed to generate chat reply: {e}")
        return jsonify({"reply": "Internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Expose on VM IP
