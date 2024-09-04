import streamlit as st
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.stem import WordNetLemmatizer
import re

# Load necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to normalize and process text
def normalize_ingredient(ingredient):
    ingredient = re.sub(r'\b(½|¼|¾|⅓|⅔)\b', '', ingredient)
    ingredient = re.sub(r'\d+\s*/\s*\d+', '', ingredient)
    ingredient = re.sub(r'\d+', '', ingredient)
    ingredient = re.sub(r'\b(cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|oz|ounce|ounces|lb|pound|pounds|g|grams|kg|kilograms|ml|milliliters|liter|liters)\b', '', ingredient, flags=re.IGNORECASE)
    ingredient = re.sub(r'\([^)]*\)', '', ingredient)
    ingredient = re.sub(r'[^\w\s]', '', ingredient)
    ingredient = ' '.join(ingredient.split())
    return ingredient.strip().lower()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Function to count ingredients and steps
def count_ingredients(ingredients):
    return len(ingredients.split('; '))

def count_steps(preparation_steps):
    return len(preparation_steps.split('. '))

# Define the LSTM model architecture
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # hidden state
        c_0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)  # cell state
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  
        return out

# Preprocess the input data
def preprocess_input(data):
    # Normalize and lemmatize text fields
    data['Recipe Title'] = data['Recipe Title'].apply(normalize_ingredient).apply(lemmatize_text)
    data['Ingredients'] = data['Ingredients'].apply(normalize_ingredient).apply(lemmatize_text)
    data['Tags'] = data['Tags'].apply(lemmatize_text)
    data['Preparation Steps'] = data['Preparation Steps'].apply(lemmatize_text)
    data['Categories'] = data['Categories'].apply(lemmatize_text)

    # Count ingredients and steps
    data['Number of Ingredients'] = data['Ingredients'].apply(count_ingredients)
    data['Number of Steps'] = data['Preparation Steps'].apply(count_steps)

    return data

# Streamlit app
st.title("Recipe Popularity Predictor")

# Recipe input fields
title = st.text_input("Recipe Title")
tags = st.text_input("Tags (separate with commas)")
ingredients = st.text_area("Ingredients (one per line)")
prep_steps = st.text_area("Preparation Steps (one per line)")
categories = st.text_area("Dish categories (breakfast, dinner, dessert, etc)")

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
        'Carbs': [carbs],
        'Fiber': [fiber],
        'Sugar': [sugar],
        'Protein': [protein],
        'Prep Time': [prep_time],
        'Cook Time': [cook_time],
        'Total Time': [total_time],
        'Categories': [categories],
        'Role': [0 if role == 'Publisher' else 1]
    })
    
    # Preprocess the input data
    X_test = preprocess_input(input_data)
    
    # Scale numerical features first
    numerical_columns = ['Cook Time', 'Prep Time', 'Total Time', 
                         'Protein', 'Fat', 'Calories', 'Sugar', 
                         'Carbs', 'Fiber', 'Number of Ingredients', 
                         'Number of Steps']
    
    scaler = StandardScaler()
    X_test[numerical_columns] = scaler.fit_transform(X_test[numerical_columns])
    
    # Vectorize text features after scaling
    tfidf_vectorizer_title = joblib.load('Preprocessing/tfidf_vectorizer_recipe_title.pkl')
    tfidf_vectorizer_ingredients = joblib.load('Preprocessing/tfidf_vectorizer_ingredients.pkl')
    tfidf_vectorizer_prep_steps = joblib.load('Preprocessing/tfidf_vectorizer_preparation_steps.pkl')
    tfidf_vectorizer_tags = joblib.load('Preprocessing/tfidf_vectorizer_tags.pkl')
    tfidf_vectorizer_categories = joblib.load('Preprocessing/tfidf_vectorizer_categories.pkl')

    tfidf_title = pd.DataFrame(tfidf_vectorizer_title.transform(X_test['Recipe Title']).toarray(), columns=tfidf_vectorizer_title.get_feature_names_out())
    tfidf_ingredients = pd.DataFrame(tfidf_vectorizer_ingredients.transform(X_test['Ingredients']).toarray(), columns=tfidf_vectorizer_ingredients.get_feature_names_out())
    tfidf_prep_steps = pd.DataFrame(tfidf_vectorizer_prep_steps.transform(X_test['Preparation Steps']).toarray(), columns=tfidf_vectorizer_prep_steps.get_feature_names_out())
    tfidf_tags = pd.DataFrame(tfidf_vectorizer_tags.transform(X_test['Tags']).toarray(), columns=tfidf_vectorizer_tags.get_feature_names_out())
    tfidf_categories = pd.DataFrame(tfidf_vectorizer_categories.transform(X_test['Categories']).toarray(), columns=tfidf_vectorizer_categories.get_feature_names_out())

    # Combine all features into a single DataFrame
    X_test = pd.concat([tfidf_title, tfidf_ingredients, tfidf_prep_steps, tfidf_tags, tfidf_categories,
                        X_test[numerical_columns]], axis=1)
    
    # Load the appropriate model based on the recipe role
    if role == "Publisher":

        input_size = X_test.shape[1]
        hidden_size = 128 
        output_size = 1
        model = SimpleLSTM(input_size, hidden_size, output_size)

        # Load the state dict and apply it to the model
        model = torch.load('./best_models/publisher/simple_lstm_best.pth')
        model.eval()

        # Convert the input to a PyTorch tensor, add batch and sequence dimensions
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32).unsqueeze(1)
        
        # Make predictions
        with torch.no_grad():
            prediction = model(X_test_tensor)
            prediction = prediction.item()  # Convert from tensor to float

        prediction_percentage = max(0, min(prediction * 100, 100))
    else:
    
        model = joblib.load('./best_models/community/ElasticNet Regression_best_model.pkl')
    
        # Handle missing columns by adding them with zeros
        missing_in_test = set(model.feature_names_in_) - set(X_test.columns)
        for col in missing_in_test:
            X_test[col] = 0

        # Remove extra columns that were not in the model training
        extra_in_test = set(X_test.columns) - set(model.feature_names_in_)
        X_test = X_test.drop(columns=extra_in_test)

        # Ensure columns are in the same order as expected by the model
        X_test = X_test[model.feature_names_in_]

        # Make predictions
        prediction = model.predict(X_test)[0]

        # Convert prediction to percentage
        prediction_percentage = max(0, min(prediction * 100, 100))
    
    # Display the result
    st.success(f"Predicted Popularity Score: {prediction_percentage:.2f}%")

if st.button("Reset"):
    st.experimental_rerun()
