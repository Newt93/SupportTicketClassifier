import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
import dlib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('tickets.csv')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[['message', 'image']], data['department'], test_size=0.2)

# Convert the messages into a numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['message'])
X_test_vec = vectorizer.transform(X_test['message'])

# extract features using openCV and dlib 
detector = dlib.get_frontal_face_detector()

def extract_features(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use dlib to detect faces in the image
    faces = detector(gray)
    # Extract the features and return them
    return len(faces)

# Use NLTK to extract additional features
def extract_features(message):
    # Tokenize the message
    words = word_tokenize(message)
    # Remove the stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    # Calculate the number of unique words


X_train_image = X_train['image'].apply(extract_features)
X_test_image = X_test['image'].apply(extract_features)

# concatenate image features with text features
X_train_vec =  np.hstack((X_train_vec.toarray(), X_train_image.values.reshape(-1, 1)))
X_test_vec =  np.hstack((X_test_vec.toarray(), X_test_image.values.reshape(-1, 1)))

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Function to assign a ticket to the right department
def assign_ticket(ticket_message, image_path):
    message_features = vectorizer.transform([ticket_message])
    image_features = extract_features(image_path)
    features = np.hstack((message_features.toarray(), image_features.reshape(-1, 1)))
    prediction = classifier.predict(features)
    return prediction[0]

# Example usage
ticket_message = "I need help with my account"
image_path = "screenshot.jpg"
department = assign_ticket(ticket_message, image_path)
print("Ticket should be assigned to the", department, "department")
