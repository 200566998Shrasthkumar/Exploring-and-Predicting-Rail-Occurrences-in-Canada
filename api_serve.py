from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'Message is required'}), 400
    
    client = OpenAI()

    completion = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:personal::BCAJlgAY",
    messages=[{"role":"user", "content": user_input}])
    bot_response = f"{completion.choices[0].message.content}"

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)