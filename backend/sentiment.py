import trax
import trax.fastmath.numpy as np
from trax import layers as tl
from trax import fastmath

def classifier(vocab_size=9088, embedding_dim=256, output_dim=2, mode='train'):
    

        
    # create embedding layer
    embed_layer = tl.Embedding( 
        vocab_size=vocab_size, # Size of the vocabulary
        d_feature=embedding_dim # Embedding dimension
    ) 
    
    # Create a mean layer, to create an "average" word embedding
    mean_layer = tl.Mean(axis=1)
    
    # Create a dense layer, one unit for each output
    dense_output_layer = tl.Dense(n_units = output_dim)
    
    # Create the log softmax layer (no parameters needed)
    log_softmax_layer = tl.LogSoftmax()
    
    # Use tl.Serial to combine all layers
    # and create the classifier
    # of type trax.layers.combinators.Serial
    model = tl.Serial( 
      embed_layer, # embedding layer
      mean_layer, # mean layer
      dense_output_layer, # dense output layer
      log_softmax_layer # log softmax layer
    ) 
    
    # return the model of type
    return model 
model=classifier()
model.init(trax.shapes.ShapeDtype((1, 1), dtype=np.int32))
model.init_from_file("./models/sentiment/model.pkl.gz", weights_only=True)
print(model('[1,0,0,1]'))



