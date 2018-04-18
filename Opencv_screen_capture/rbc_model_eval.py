from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from helper import decode_predictions_custom, image_preprocess

#Custom model path and test image path
custom_model = 'rbc_custom_model.h5'
test_image_path = './test_data/h1.jpg'

#loading custom model
rbc_model = load_model(custom_model)

#Display test image
image.load_img(test_image_path)


def model_prediction(test_image_path):
    test_image = image_preprocess(test_image_path)
    y_prob = rbc_model.predict(test_image)
    predictions = decode_predictions_custom(y_prob)
    return predictions

(id, action, prob)=model_prediction(test_image_path)[0][0]

action

if __name__ == '__main__':
    print(model_prediction(test_image_path)[0])


    
    
    
