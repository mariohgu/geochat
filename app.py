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

