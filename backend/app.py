from flask import Flask,jsonify,request


app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/compare', methods=['POST'])
def create_search_question():
    body = request.get_json()

    first_statement = body.get("question", None)
    second_statement = body.get("answer", None)
    #prepare the questions

    # infer the questions if same from the siamese Model
    
