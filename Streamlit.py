import streamlit as st
import pandas as pd
import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained models 
def load_model(model_name, role, feature_set):
    model_path = f"best_models/{role}/{feature_set}/{model_name}_best_model.pkl"
    return joblib.load(model_path)

# Load LSTM model separately
def load_lstm_model():
    model_path = "best_models/best_model_lstm_search_1_features_50.pt"
    model = torch.load(model_path)
    model.eval()
    return model

# Function to lemmatize text
def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Function to preprocess the input data
def preprocess_input(data):
    # Apply lemmatization to the text fields
    data['Recipe Title'] = data['Recipe Title'].apply(lemmatize_text)
    data['Ingredients'] = data['Ingredients'].apply(lemmatize_text)
    data['Tags'] = data['Tags'].apply(lemmatize_text)
    data['Preparation Steps'] = data['Preparation Steps'].apply(lemmatize_text)
    
    # TF-IDF vectorization for each text field
    tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    
    # Vectorize and concatenate the TF-IDF matrices
    tfidf_title = tfidf_vectorizer.fit_transform(data['Recipe Title'].values)
    tfidf_ingredients = tfidf_vectorizer.fit_transform(data['Ingredients'].values)
    tfidf_tags = tfidf_vectorizer.fit_transform(data['Tags'].values)
    tfidf_prep_steps = tfidf_vectorizer.fit_transform(data['Preparation Steps'].values)
    
    # Convert TF-IDF matrices to DataFrames
    df_tfidf_title = pd.DataFrame(tfidf_title.toarray(), columns=[f'title_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()])
    df_tfidf_ingredients = pd.DataFrame(tfidf_ingredients.toarray(), columns=[f'ingredients_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()])
    df_tfidf_tags = pd.DataFrame(tfidf_tags.toarray(), columns=[f'tags_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()])
    df_tfidf_prep_steps = pd.DataFrame(tfidf_prep_steps.toarray(), columns=[f'prep_{feat}' for feat in tfidf_vectorizer.get_feature_names_out()])
    
    # Combine all TF-IDF DataFrames with the original data
    preprocessed_data = pd.concat([data.reset_index(drop=True), df_tfidf_title, df_tfidf_ingredients, df_tfidf_tags, df_tfidf_prep_steps], axis=1)
    
    # Drop the original text columns that have been vectorized
    preprocessed_data.drop(columns=['Recipe Title', 'Ingredients', 'Tags', 'Preparation Steps'], inplace=True)
    
    return preprocessed_data

# Streamlit app
st.title("Recipe Popularity Predictor")

# Recipe input fields
title = st.text_input("Recipe Title")
tags = st.text_input("Tags (separate with commas)")
ingredients = st.text_area("Ingredients (one per line)")
prep_steps = st.text_area("Preparation Steps (one per line)")

# Nutritional values input fields
calories = st.number_input("Calories", min_value=0)
fat = st.number_input("Fat (g)", min_value=0.0, format="%.2f")
carbs = st.number_input("Carbohydrates (g)", min_value=0.0, format="%.2f")
fiber = st.number_input("Fiber (g)", min_value=0.0, format="%.2f")
sugar = st.number_input("Sugar (g)", min_value=0.0, format="%.2f")
protein = st.number_input("Protein (g)", min_value=0.0, format="%.2f")

# Cooking times input fields
prep_time = st.number_input("Preparation Time (minutes)", min_value=0)
cook_time = st.number_input("Cooking Time (minutes)", min_value=0)
total_time = st.number_input("Total Time (minutes)", min_value=0)

# Publisher or Community
role = st.selectbox("Is this a Publisher or Community recipe?", ("Publisher", "Community"))

# Submit button
if st.button("Predict Popularity"):
    # Combine inputs into a single DataFrame
    input_data = pd.DataFrame({
        'Recipe Title': [title],
        'Tags': [tags],
        'Ingredients': [ingredients],
        'Preparation Steps': [prep_steps],
        'Calories': [calories],
        'Fat': [fat],
        'Carbohydrates': [carbs],
        'Fiber': [fiber],
        'Sugar': [sugar],
        'Protein': [protein],
        'Preparation Time': [prep_time],
        'Cooking Time': [cook_time],
        'Total Time': [total_time],
        'Role': [0 if role == 'Publisher' else 1]
    })
    
    # Preprocess the input data
    preprocessed_data = preprocess_input(input_data)

    if role == "Publisher":
        model_name = "SimpleLSTM"
        model = load_lstm_model()
        prediction = model(torch.tensor(preprocessed_data.toarray(), dtype=torch.float32))
    elif role == "Community":
        model_name = "ElasticNet Regression"    
    
    # Display the result
    st.success(f"Predicted Popularity Score: {prediction[0]:.2f}")

if st.button("Reset"):
    st.experimental_rerun()
