from flask import Flask,jsonify,request
from ner import get_vocab,initialize_model,predict
from errors import errors



def create_app():
    app = Flask(__name__)
    app.register_blueprint(errors)
    return app
app=create_app()

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Named entity recognition endpoint 
@app.route('/ner', methods=['POST'])
def named_entity_recognition():
    body = request.get_json()

    # get the statement from user
    statement = body.get("statement", None)
   
    # generate the vocabulary from the words and tags data
    vocab,tags=get_vocab("./models/ner/words.txt","./models/ner/tags.txt")
    #initialize the model
    model=initialize_model(tags,"./models/ner/model.pkl.gz")

    sentence=statement

    #Predict named entities given the sentence
    preds=predict(sentence, model, vocab, tags)
    
    #Package the predicitions with original word in a dictionary i.e. for returning a json object
    predictions={}
    for index,pred in enumerate(preds):
        predictions[sentence.split(' ')[index]]=pred
    
    # Return the serialized json with data(original word with prediction) and success message
    return jsonify({
        "data":predictions,
        'success': True
        })
    
