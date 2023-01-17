import trax 

from trax.supervised import training
from trax import layers as tl

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