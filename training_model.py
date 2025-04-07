import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity

IMG_SIZE = (128, 128)

def load_data(dataset_path):
    images = []
    labels = []
    for root, _, files in os.walk(dataset_path): 
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')): 
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = cv2.resize(image, IMG_SIZE) / 255.0
                    images.append(image)
                    labels.append(file)
    
    X = np.array(images).reshape(-1, 128, 128, 1)
    return X, labels

def create_feature_extractor():
    inputs = Input(shape=(128, 128, 1))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x) 
    model = Model(inputs, x)
    return model

dataset_path = "dataset" 
X_train, labels_train = load_data(dataset_path)

feature_extractor = create_feature_extractor()
feature_extractor.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.CosineSimilarity()) 
feature_extractor.fit(X_train, feature_extractor.predict(X_train), epochs=10, batch_size=8)  

feature_extractor.save("fingerprint_matcher_model.keras")
print(" Model saved successfully as fingerprint_matcher_model.keras")

loaded_model = tf.keras.models.load_model("fingerprint_matcher_model.keras")
print(" Model loaded successfully!")

stored_embeddings = loaded_model.predict(X_train)

def match_fingerprint(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "Invalid image"
    image = cv2.resize(image, IMG_SIZE) / 255.0
    image = image.reshape(1, 128, 128, 1)
    
    query_embedding = loaded_model.predict(image)
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_match_label = labels_train[best_match_idx]
    similarity_score = similarities[best_match_idx] * 100
    
    return f"Best Match: {best_match_label} with {similarity_score:.2f}% similarity"

print(match_fingerprint("dataset/train_data/00000_00.jpg"))
