import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Global variables
data = None
vectorizer = None
tfidf_matrix = None

# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Predefined options for each preference category
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science","Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}

# Order of the categories
category_order = ["emotions", "occasion", "interests", "audience", "personality"]

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[^;]*;', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(lemmatized_words)
    # Filter stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = ' '.join(filtered_words)
    return text

def initialize_data():
    global data, vectorizer, tfidf_matrix
    data = pd.read_csv("output.csv")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    data["preprocessed_title"] = data["Title"].apply(preprocess_text)
    data["preprocessed_text"] = data.apply(lambda row: " ".join(set(str(row["preprocessed_title"]).split() + str(row["preprocessed_description"]).split())), axis=1)
    # Create TF-IDF matrix - expliquer ici : https://imgur.com/Zn4rxWS

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

@app.route("/")
def home():    
    return render_template("index.html")

@app.route('/questions', methods=['POST'])
def questions():
    if request.method == 'POST':
        initialize_data()
        emotions = request.form['emotions']
        occasion = request.form['occasion']
        interests = request.form['interests']
        audience = request.form['audience']
        personality = request.form['personality']

        # Combine all inputs into a single string
        all_text = f"{emotions} {occasion} {interests} {audience} {personality}"

        # Preprocess user answers
        preprocessed_answers = preprocess_text(all_text)

        # Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([preprocessed_answers])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

        # Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        titles = recommended_products["Title"].tolist()
        urls = recommended_products["URL (Web)"].tolist()
        images = recommended_products["Image 1"].tolist()

        return render_template('index2.html', questions_data=zip(titles, urls, images), preprocessed_answers=preprocessed_answers)
    
    # This line should never be reached as we're only allowing POST requests
    return "Method not allowed", 405


@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    if request.method == 'POST':
        initialize_data()
        user_input = request.form['user_input']
        user_input = preprocess_text(user_input)
        
        predicted_labels = []
        seen_categories = set()
        processed_classes = []
        """
        The role of the classifier function (zero shot) in the provided code is to classify the user's input text into relevant categories or labels. It performs a zero-shot classification, which means it can classify the text into categories that were not seen during training by leveraging a pre-trained model's knowledge.

        Here's a breakdown of how the classifier function is used in the code:

        The classifier function is called with the user_input text and a list of labels (labels) specific to a certain category. 
        The classifier function returns a result that contains the predicted labels for the given text.
        The predicted labels are then iterated through in the for loop. The loop checks if a label has already been seen 
        (label not in seen_categories). If the label is new, it is appended to the predicted_labels list and the processed_classes list. 
        The purpose of the processed_classes list is to store all the unique labels that have been predicted for later use.
        After iterating through all the categories, the processed_classes list is joined into a single line using " ".join(processed_classes). 
        This creates a string representation of all the unique labels predicted for the user's input.
        The processed_classes string is then used in subsequent steps for similarity calculations and generating recommended products.
        """
        for category in category_order:
            labels = options[category]
            result = classifier(user_input, labels)

            for label in result["labels"]:
                if label not in seen_categories:
                    predicted_labels.append(label)
                    processed_classes.append(label)
                    seen_categories.add(label)
                    break

        processed_classes = " ".join(processed_classes)

        # Give more weight to label 1 and label 5
        weighted_processed_classes = processed_classes
        if len(predicted_labels) >= 1:
            weighted_processed_classes += f" {predicted_labels[0]}" * 3  # Repeat label 1 three times
        if len(predicted_labels) >= 5:
            weighted_processed_classes += f" {predicted_labels[4]}" * 3  # Repeat label 5 three times

        # Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([weighted_processed_classes])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

        # Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        recommended_titles = recommended_products["Title"].tolist()
        recommended_urls = recommended_products["URL (Web)"].tolist()
        recommended_images = recommended_products["Image 1"].tolist()

        if 'nlp-submit' in request.form:
            return render_template('index1.html', user_input=user_input, predicted_labels=predicted_labels, processed_classes=processed_classes,
                            recommended_data=zip(recommended_titles, recommended_urls, recommended_images))

    return render_template('index1.html')
        

if __name__ == '__main__':
    app.run()