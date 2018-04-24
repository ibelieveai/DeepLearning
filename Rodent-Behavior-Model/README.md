# Trained image classification models for Keras

This repository contains code for the Rodent behaviors Keras models:

- Transfer learning on VGG16 and with 5 frozen layers


All architectures are compatible with both TensorFlow. Custom weights has to be loaded from the repository.

**Note that using these models requires the latest version of Keras (from the Github repo, not PyPI).**

## Examples

### Classify videos

```python
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from  keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from helper import decode_predictions_custom, image_preprocess
import argparse
import cv2
import numpy as np
import os
import random
import glob
import sys
from sklearn.utils import shuffle
import time

file = 'video_2.mpg'

#loading model
custom_model = 'rbc_custom_model.h5'
model = load_model(custom_model)

cap = cv2.VideoCapture(file)
time.sleep(2)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  - 1

frames = 1
while frames < video_length:
    
    
    ret,original = cap.read()
    # Load the image using Keras helper ultility
    print("[INFO] loading and preprocessing image...")
    frame = cv2.resize(original, (224, 224)) 
    frame = image_utils.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    preds = model.predict(frame)
    (inID, label, prob) = decode_predictions_custom(preds)[0][0]
    # Display the predictions
    print("RBC ID: {}, Label: {}, Prob: {}".format(inID, label, prob))
    cv2.putText(original, "Label: {}, Prob: {}".format(label, prob), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)
    cv2.waitKey(1)
    frames += 1
```

### Capturing image from video and extract features

```python
cap = cv2.VideoCapture(file)
while frames < video_length:
    
    
    ret,original = cap.read()
    # Load the image using Keras helper ultility
    print("[INFO] loading and preprocessing image...")
    frame = cv2.resize(original, (224, 224)) 
    frame = image_utils.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)


```


## References

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) - please cite this paper if you use the VGG models in your work.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - please cite this paper if you use the ResNet model in your work.
- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567) - please cite this paper if you use the Inception v3 model in your work.

Additionally, don't forget to [cite Keras](https://keras.io/getting-started/faq/#how-should-i-cite-keras) if you use these models.


## License

- All code in this repository is under the MIT license as specified by the LICENSE file.
- The ResNet50 weights are ported from the ones [released by Kaiming He](https://github.com/KaimingHe/deep-residual-networks) under the [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE).
- The VGG16 and VGG19 weights are ported from the ones [released by VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/).
- The Inception v3 weights are trained by ourselves and are released under the MIT license.
