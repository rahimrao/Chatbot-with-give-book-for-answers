# Chatbot-with-give-book-for-answers

import re
import random
import warnings
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK tokenizer resources are available
nltk.download('punkt')
nltk.download('punkt_tab')  # Download punkt_tab if needed

warnings.filterwarnings('ignore')

# Define the correct file paths
pdf_file_path = r"C:\Users\Administrator\Documents\myProjects\book.pdf"
text_file_path = r"C:\Users\Administrator\Documents\myProjects\student_book.txt"

# Initialize an empty string to store extracted text
extracted_text = ""

# Extract text from the PDF and save it to a text file
try:
    with fitz.open(pdf_file_path) as pdf:
        # Check if PDF has any pages
        if pdf.page_count == 0:
            raise ValueError("PDF is empty.")
        
        # Extract text from each page
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            extracted_text += page.get_text("text")
    
    # Ensure text was extracted before saving
    if extracted_text.strip() == "":
        print("No text could be extracted from the PDF.")
    else:
        # Save the extracted text to a .txt file
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)
        print(f"Text extracted and saved to {text_file_path}")

except FileNotFoundError:
    print(f"Error: The file {pdf_file_path} was not found.")
except Exception as e:
    print("Error:", e)

# Load and process text if extraction was successful
if extracted_text.strip():
    # Tokenize the book content into sentences for easier searching
    book_content = extracted_text
    try:
        book_sections = nltk.sent_tokenize(book_content)  # Tokenize into sentences
    except Exception as e:
        print(f"Error in tokenization: {e}")
        book_sections = []

    # Initialize the TF-IDF vectorizer and fit it on the book sections
    vectorizer = TfidfVectorizer().fit(book_sections)

    # Define a function to find the most relevant answer from the book content
    def find_best_answer(question):
        try:
            question = vectorizer.transform([question])
            book = vectorizer.transform(book_sections)
            similarities = cosine_similarity(question, book).flatten()
            best = similarities.argmax()
            best_answ = book_sections[best]
            return best_answ
        except Exception as e:
            return "I'm sorry, I couldn't find an answer to your question. Error: " + str(e)

    # Predefined chatbot responses
    responses = {
        "hello": ["Hi there!", "Hello!", "Hey! How can I help you?"],
        "how are you": ["I'm a bot, but I'm here to help you!"],
        "bye": ["Goodbye!", "See you later!", "Take care!"],
        "can you eat?": ["As a robot, I cannot eat anything, but I can be charged."],
        "what is your name": ["I'm your friendly chatbot!", "I’m called Chatbot.", "You can call me AI Bot!"],
        "what can you do": ["I can chat with you, answer questions, and help you with information!"],
        "how old are you": ["I'm ageless, but I was launched in 2022!", "Age is just a concept for me!"],
        "who made you": ["I was created by OpenAI.", "The talented team at OpenAI brought me to life!"],
        "where are you from": ["I'm from the digital world, created by OpenAI.", "I live in the cloud!"],
        "what's the weather like": ["I can't check the weather, but you can look outside!", "Check your local weather app for that!"],
        "tell me a joke": ["Why did the computer go to the doctor? Because it had a virus!"],
        "thank you": ["You're welcome!", "Happy to help!", "Anytime!"]
    }

    # Define the chatbot response function
    def chatbot(user_input):
        user_input = user_input.lower()
        
        # Check for predefined responses
        for key in responses:
            if re.search(r"\b" + key + r"\b", user_input):
                return random.choice(responses[key])
        
        # If no predefined response matches, search the book for an answer
        return find_best_answer(user_input)

    # Main chatbot loop
    print("Chatbot: Hello! Ask me anything or type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            print("Chatbot: Goodbye!")
            break
        else:
            print("Chatbot:", chatbot(user_input))
