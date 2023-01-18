from flask import Flask,jsonify,request


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/ner', methods=['POST'])
def named_entity_recognition():
    body = request.get_json()

    statement = body.get("statement", None)
    #prepare the questions

    # infer the questions if same from the siamese Model
    
