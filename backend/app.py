from flask import Flask,jsonify,request
from ner import get_vocab,initialize_model,predict
from errors import errors
from sentiment import classifier



def create_app():
    app = Flask(__name__)
    app.register_blueprint(errors)
    return app
app=create_app()

@app.route("/", methods=['POST','GET'])
def health():
    return jsonify("Healthy")

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
    predictions={}
    for index,pred in enumerate(preds):
        predictions[sentence.split(' ')[index]]=pred
    

    return jsonify({"data":predictions,
    'success': True})



    
