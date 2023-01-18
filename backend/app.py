from flask import Flask,jsonify,request
from ner import get_vocab,initialize_model,predict


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
    vocab,tags=get_vocab("./models/ner/words.txt","./models/ner/tags.txt")
    model=initialize_model(tags,"./models/ner/model.pkl.gz")
    sentence=statement
    preds=predict(sentence, model, vocab, tags)

    return jsonify(zip(sentence,preds))
    
