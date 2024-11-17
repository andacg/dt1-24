from flask import Flask, request as req, jsonify
import requests
import logging
import os

app = Flask(__name__)


def query(payload, model_id, api_token):
    headers = {"Authorization": f"Bearer {api_token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


@app.route("/chat")
def call_huggingface_chat_model():
    model_id = req.args.get("model_id")
    logging.debug(f"The model ID for the Huggingface model is {model_id}")
    huggingface_token = req.args.get("huggingface_token")
    logging.debug(f"Huggingface API Token: {huggingface_token}")
    questions = req.args.get("input")
    
    # Validate required parameters
    if not all([model_id, huggingface_token, questions]):
        return jsonify({"error": "Missing required query parameters"}), 400

    logging.debug(f"The model ID for the Huggingface model is {model_id}")
    logging.debug(f"Huggingface API Token: {huggingface_token}")
    data = query(
        {
            "inputs": f'{questions.replace("_"," ")}',
            "options": {"wait_for_model": True},
            "parameters": {"return_full_text": False, "max_time": 30},
        },
        model_id,
        huggingface_token,
    )
    logging.debug(f"Model output: {data}")
     if "error" in data:
        return jsonify({"error": data["error"]}), 500
    try:
        output = jsonify({"ack": data[0]["generated_text"]})
    except (IndexError, KeyError):
        logging.error(f"Unexpected API response format: {data}")
        return jsonify({"error": "Invalid API response"}), 500

    output.headers.add("Access-Control-Allow-Origin", "*")
    return output



@app.route("/load_model")
def load_huggingface_chat_model():
    model_id = req.args.get("model_id")
    logging.debug(f"The model ID for the Huggingface model is {model_id}")
    huggingface_token = req.args.get("huggingface_token")
    logging.debug(f"Huggingface API Token: {huggingface_token}")
    data = query(
        {
            "inputs": "hello",
            "parameters": {"return_full_text": True},
        },
        model_id,
        huggingface_token,
    )
    logging.debug(f"Model output: {data}")
    output = {}
    if "generated_text" in data[0]:
        output = jsonify({"ack": "model loaded and ready"})
    else:
        output = jsonify({"ack": data[0][list(data[0].keys())[0]]})
    output.headers.add("Access-Control-Allow-Origin", "*")
    return output


@app.route("/ping")
def ping():
    output = jsonify({"ack": "pong"})
    output.headers.add("Access-Control-Allow-Origin", "*")
    return output


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
