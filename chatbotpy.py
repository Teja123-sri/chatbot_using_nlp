import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



# Load intents from JSON file
file_path = os.path.abspath(os.path.join("C:/Users/srira/OneDrive/Desktop/aicte/intents.json"))

with open(file_path, "r") as file:
    intents = json.load(file)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# UI Design
def main():
    st.set_page_config(page_title="AI Chatbot", layout="wide")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
            /* Gradient background for the sidebar */
            .sidebar .sidebar-content {
                background: linear-gradient(145deg, #6a11cb, #2575fc);
                color: white;
                padding: 20px;
            }
            /* Custom font for the entire app */
            html, body, [class*="css"] {
                font-family: 'Georgia', serif;
            }
            /* Chat bubbles styling */
            .user-bubble {
                background-color: #0078D4;
                color: white;
                padding: 12px 16px;
                border-radius: 15px 15px 0 15px;
                margin: 8px 0;
                max-width: 70%;
                align-self: flex-end;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .bot-bubble {
                background-color: #f1f1f1;
                color: #333;
                padding: 12px 16px;
                border-radius: 15px 15px 15px 0;
                margin: 8px 0;
                max-width: 70%;
                align-self: flex-start;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            /* Input box styling */
            .stTextInput>div>div>input {
                border-radius: 20px;
                padding: 10px 16px;
                font-size: 16px;
                border: 1px solid #ddd;
            }
            /* Animation for chat bubbles */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .user-bubble, .bot-bubble {
                animation: fadeIn 0.5s ease-in-out;
            }
            /* Footer styling */
            .footer {
                text-align: center;
                padding: 10px;
                margin-top: 20px;
                color: #666;
                font-size: 14px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar menu with styling
    st.sidebar.markdown(
        """
        <h2 style='text-align: center; color: black;'>Menu</h2>
        """,
        unsafe_allow_html=True,
    )
    menu = ["Chatbot", "Conversation History", "About"]
    choice = st.sidebar.radio("", menu)

    # Ensure chat log file exists
    if not os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

    # Main content area
    if choice == "Chatbot":
        st.markdown(
            "<h1 style='text-align: center; color: #2575fc;'>🤖 AI Chatbot</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h4 style='text-align: center; color: #666;'>Welcome! Type your message below to start chatting.</h4>",
            unsafe_allow_html=True,
        )

        # Use a session state to store the user input
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Input field with a unique key
        user_input = st.text_input(
            "", 
            value=st.session_state.user_input, 
            key="chat_input", 
            placeholder="Type your message here...", 
            label_visibility="collapsed"
        )

        # Process user input
        if user_input:
            response = chatbot(user_input)

            # Display messages in UI
            st.markdown(
                f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div class='user-bubble'>
                        {user_input}
                    </div>
                    <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" width="30" height="30" style="margin-left:10px;"/>
                </div>
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" width="30" height="30" style="margin-right:10px;"/>
                    <div class='bot-bubble'>
                        {response}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Save to chat history
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv.writer(csvfile).writerow([user_input, response, timestamp])

            # Clear the input field after processing
            st.session_state.user_input = ""

            if response.lower() in ["goodbye", "bye"]:
                st.markdown(
                    "<h4 style='text-align: center; color: #666;'>Thank you for chatting with me. Have a great day! 😊</h4>",
                    unsafe_allow_html=True,
                )
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    st.markdown(
                        f"""
                        <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                            <div class='user-bubble'>
                                {row[0]}
                            </div>
                            <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" width="30" height="30" style="margin-left:10px;"/>
                        </div>
                        <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png" width="30" height="30" style="margin-right:10px;"/>
                            <div class='bot-bubble'>
                                {row[1]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    elif choice == "About":
        st.header("About")
        st.subheader("Project Overview")
        st.write("""
        This project aims to develop a chatbot that understands and responds to user inputs based on predefined intents.
        It utilizes *Natural Language Processing (NLP)* and *Logistic Regression* to extract intent and generate
        meaningful responses. The chatbot interface is built using *Streamlit*, enabling an interactive and
        user-friendly experience.
        """)

        st.write("""
        The project consists of two main parts:
        1. *Training the Chatbot:*
           - The chatbot is trained using NLP techniques and Logistic Regression.
           - It learns from labeled intents to classify user inputs and generate appropriate responses.
        2. *Building the Chatbot Interface:*
           - Streamlit is used to create a simple web-based chatbot interface.
           - Users can enter text and receive responses in real-time.
        """)

        st.subheader("Dataset")
        st.write("""
        The chatbot is trained on a dataset containing labeled intents, entities, and text samples.
        - *Intents:* Categorize user queries (e.g., "greeting", "budget", "about").
        - *Entities:* Extract meaningful keywords (e.g., "Hi", "How do I create a budget?", "What is your purpose?").
        - *Text Samples:* Provide real-world examples for training.
        """)

        st.subheader("Streamlit Chatbot Interface")
        st.write("""
        The chatbot's interface is designed using *Streamlit*, which enables:
        - A text input box for user queries.
        - A chat window to display chatbot responses.
        - A conversation history feature for reviewing past interactions.
        """)

        st.subheader("Conclusion")
        st.write("""
        This project successfully builds a chatbot capable of understanding user intent and responding intelligently.
        Using *NLP and Logistic Regression, it processes text inputs efficiently. The **Streamlit-based interface*
        enhances user interaction. Future improvements can include expanding the dataset, incorporating advanced NLP
        models, and integrating deep learning techniques for more sophisticated responses.
        """)

    # Footer
    # st.markdown(
    #     """
    #     <div class="footer">
    #         <p>© 2023 AI Chatbot. All rights reserved.</p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True,
    # )

if __name__ == "__main__":
    main()