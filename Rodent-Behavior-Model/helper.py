# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 08:53:49 2018

@author: cr201692
"""
from keras.preprocessing import image

def decode_predictions_custom(preds, top=4):
    import numpy as np
    import json
    global CLASS_INDEX
    CLASS_INDEX = None
    CLASS_INDEX_PATH = './rbc_custom_class.json'
    if len(preds.shape) != 2 or preds.shape[1] != 4:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 4)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))
    results = []
    
    for pred in preds:
        top_indices = np.argsort(-preds)[::-1][0]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        #result = dict((CLASS_INDEX,pred) for (CLASS_INDEX,pred) in top_indices)
        results.append(result)
        return results
    


    
#model eval
def image_preprocess(path):
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input
    import numpy as np
    x = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    x = x/255
    return(x)