'''

from flask import Flask, render_template, request, jsonify
from modelo import respuesta_chatbot

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.form['user_message']
    bot_response = respuesta_chatbot(user_message)
    return jsonify({'bot_response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)

'''
from flask import Flask, render_template, request, jsonify
from modelo import respuesta_chatbot

app = Flask(__name__)

@app.get("/")
def index_get():
	return render_template("index.html")

@app.post("/predict")
def predict():
	text = request.get_json().get("message")
	response = respuesta_chatbot(text)
	message = {"answer": response}
	return jsonify(message)
	
if __name__ == "__main__":
	app.run(debug=True)

