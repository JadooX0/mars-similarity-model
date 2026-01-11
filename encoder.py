from pyexpat import model
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from tensorflow.keras.models import Model
from torch import cosine_similarity

from modeltrain.unet import load_img

encoder_model = Model(inputs=model.input, outputs=model.get_layer(name=None, index=3).output) 



def get_real_embedding(path):
    img = load_img(path)
    
    
    feature_map = encoder_model.predict(img[None, ...], verbose=0)
    
    
    embedding = np.mean(feature_map, axis=(1, 2)) 
    return embedding


e1 = get_real_embedding("mars1.jpg")
e2 = get_real_embedding("mars2.jpg")

sim = cosine_similarity(e1, e2)[0][0]
print(f"Improved Feature Similarity: {sim:.4f}")