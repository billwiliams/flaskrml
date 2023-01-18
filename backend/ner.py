import trax 

from trax.supervised import training
from trax import layers as tl
import numpy as np

# Create Named entity recognition model


def NER(tags, vocab_size=35181, d_model=50):
    '''
      Input: 
        tag_map - dictionary that maps the tags to numbers
        vocab_size - integer containing the size of the vocabulary
        d_model - integer describing the embedding size
      Output:
        model - a trax serial model
    '''
   
    model = tl.Serial( 
      tl.Embedding(vocab_size=vocab_size,d_feature=d_model), # Embedding layer
      tl.LSTM(d_model), # LSTM layer
      tl.Dense(len(tags)), # Dense layer with len(tags) units
      tl.LogSoftmax() # LogSoftmax layer
      ) 

    return model

def initialize_model(tag_map,path):
    # initializing your model
    model = NER(tag_map)
    # display your model
    print(model)
    model.init(trax.shapes.ShapeDtype((1, 1), dtype=np.int32))

    # Load the pretrained model
    model.init_from_file(path, weights_only=True)

    return model


def predict(sentence, model, vocab, tag_map):
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
    batch_data = np.ones((1, len(s)))
    batch_data[0][:] = s
    sentence = np.array(batch_data).astype(int)
    output = model(sentence)
    outputs = np.argmax(output, axis=2)
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx = outputs[0][i] 
        pred_label = labels[idx]
        pred.append(pred_label)
    return pred


def get_vocab(vocab_path, tags_path):
    vocab = {}
    with open(vocab_path) as f:
        for i, l in enumerate(f.read().splitlines()):
            vocab[l] = i  # to avoid the 0
        # loading tags (we require this to map tags to their indices)
    vocab['<PAD>'] = len(vocab) # 35180
    tag_map = {}
    with open(tags_path) as f:
        for i, t in enumerate(f.read().splitlines()):
            tag_map[t] = i 
    
    return vocab, tag_map


