import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 FAQ Chatbot")

questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?",
    "What is Streamlit?",
    "What is GitHub?"
]

answers = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Python is a popular programming language.",
    "Machine Learning is a branch of AI where systems learn from data.",
    "Streamlit is a Python library used to build web apps easily.",
    "GitHub is a platform to host and manage code using Git."
]

user_question = st.text_input("Ask a Question")

if st.button("Get Answer"):
    if user_question:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(questions + [user_question])
        similarity = cosine_similarity(vectors[-1], vectors[:-1])
        index = similarity.argmax()
        st.success(answers[index])
    else:
        st.warning("Please enter a question.")