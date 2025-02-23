import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import io
import os
import random
import string
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Load the saved models and feature files
@st.cache_resource
def load_models():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/extracted_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open("models/filename.pkl", "rb") as f:
        image_paths = pickle.load(f)
    return model, features, image_paths

# Function to extract features from the image

def extract_features(img, model):
    img = img.resize((224, 224))  # Resize image
    img_array = np.array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = preprocess_input(img_array)  # Preprocess input
    result = model.predict(img_array).flatten()  # Get model predictions
    norm_result = result / np.linalg.norm(result)  # Normalize features
    return norm_result

# Function to get top 5 similar images
# def get_similar_images(query_features, features, image_paths):
#     distances = np.linalg.norm(features - query_features, axis=1)
#     indices = np.argsort(distances)[:5]
#     return [image_paths[i] for i in indices]

def get_similar_images(query_features, features, image_paths):
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors([query_features])
    return [image_paths[i] for i in indices[0]]


# Function to save the uploaded image and update stored features
# def save_new_image(img, query_features, image_paths, features):
#     image_dir = "images/"
#     os.makedirs(image_dir, exist_ok=True)
#     img_name = generate_random_filename()
#     img_path = os.path.join(image_dir, img_name)
#     img.save(img_path)
#     image_paths.append(img_path)
#     features.append(query_features)
    
#     # Save updated feature list and filenames
#     with open("models/extracted_features.pkl", "wb") as f:
#         pickle.dump(features, f)
#     with open("models/filename.pkl", "wb") as f:
#         pickle.dump(image_paths, f)
    
#     return img_path

# Function to generate a random filename
def generate_random_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".jpg"


# Streamlit UI
st.title("Fashion Product Recommendation System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Load models
    model, features, image_paths = load_models()

    # Extract features from the uploaded image
    query_features = extract_features(image, model)

    # Get similar images
    recommended_images = get_similar_images(query_features, features, image_paths)
    
    # Checkbox to save the new image
    # save_image = st.checkbox("Save this image for future recommendations")
    
    # if save_image:
    #     new_image_path = save_new_image(image, query_features, image_paths, features)

    # Display recommended images
    st.write("### Recommended Images:")
    cols = st.columns(5)
    for col, img_path in zip(cols, recommended_images):
        col.image("images/" + img_path, width=100)  # Adjust width as needed

