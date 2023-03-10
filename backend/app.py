from flask import Flask,jsonify,request
from ner import get_vocab,initialize_model,predict
from errors import errors
from sentiment import classifier
from siamese import * 
from text_summarization import TransformerLM,greedy_decode



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

@app.route('/siamese',methods=['POST'])

def similar():
    body = request.get_json()

    statement_one= body.get("statement_one", None)
    statement_two= body.get("statement_two", None)
    #prepare the questions

    model=Siamese()

    model.init_from_file("./models/siamese/model.pkl.gz")
    #print(model(np.ones(shape=(256),dtype=np.int32)))
    with open('./models/siamese/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    prediction=predict(statement_one,statement_two,0.96, model, vocab, data_generator=data_generator, verbose=False)
    print(prediction[0][0])
    return  jsonify({"similar":str(prediction[0][0]),
    'success': True})

@app.route('/summarize',methods=['POST'])

def similar():
    body = request.get_json()

    article= body.get("article", None)
    

    # Get the model architecture
    model = TransformerLM(mode='eval')

    # Load the pre-trained weights
    model.init_from_file("./models/transformer/model.pkl.gz", weights_only=True)
    
    summary=greedy_decode(article, model)
    
    return  jsonify({"summary":summary,
    'success': True})



    
