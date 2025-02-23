# Fashion Product Recommendation System

## Overview
This project is a **Fashion Product Recommendation System** that utilizes **ResNet50** for feature extraction. The system takes a query image and suggests the most visually similar fashion products from a dataset of **44,400 images**. Due to the large size of the dataset and extracted features, they are not included in this repository but can be generated using the provided notebook.
**Note**: The current dataset size is limited, which may result in suboptimal recommendations. Training the model on a larger dataset will significantly improve recommendation accuracy.

## Preview

![Screenshot 2025-02-23 214146](https://github.com/user-attachments/assets/9ad92dc4-bfb7-4ba1-aee7-8c8284395f29)


![Screenshot 2025-02-23 214419](https://github.com/user-attachments/assets/80953f0a-f60a-4d7f-9e68-0555ed65a416)


![Screenshot 2025-02-23 215001](https://github.com/user-attachments/assets/53431078-641d-47b1-b935-e992aa42fdaa)


## Features
- Uses **ResNet50** for feature extraction.
- Recommends visually similar products.
- **Streamlit** application for easy user interaction.
- **Pre-trained feature extraction** to speed up recommendations.

## How It Works
1. **Feature Extraction**: ResNet50 extracts feature vectors from each image in the dataset.
2. **Similarity Search**: The system finds the most similar products using cosine similarity.
3. **Streamlit Interface**: Users can upload an image and get top product recommendations.

## Installation
### Clone the Repository
```sh
git clone https://github.com/tejas-130704/Fashion-Recommender.git
cd Fashion-Recommender
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Download Dataset
- **Download Images Dataset**: [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) 

### Generate Extracted Features
Run the provided Jupyter Notebook to extract features:
```sh
jupyter notebook fashion-product-recommendation-system.ipynb
```

This will generate the following files:
- `models/model.pkl` - The trained recommendation model.
- `models/extracted_features.pkl` - Extracted feature vectors for image similarity.
- `models/filename.pkl` - Paths to the dataset images.

Ensure these files are placed in the `models/` directory before running the application.

## Running the Application
```sh
streamlit run main.py
```

## Usage
1. Upload a fashion product image.
2. Get top recommended products based on visual similarity.
3. Click on recommendations to view similar items.

## File Structure
```
Fashion-Recommender/
│── main.py                 # Streamlit app
│── fashion-product-recommendation-system.ipynb      # Jupyter Notebook for training
│── requirements.txt        # Required libraries
│── images/                 # Directory for images (not included, link is provided)
│── models/                 # Directory for exported models and features (user-generated)
    │── model.pkl           # The trained recommendation model.
    │── extracted_features.pkl  # Extracted feature vectors for image similarity
    │── filename.pkl        # Paths to the dataset images.
```

## Contributing
If you'd like to contribute, feel free to fork the repo and submit a pull request!


