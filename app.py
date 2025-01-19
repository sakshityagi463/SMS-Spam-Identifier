import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)  # Return the transformed text

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom CSS for button styling
st.markdown(
    """
    <style>
    /* Predict button color */
    .stButton button[data-testid="baseButton-secondary"] {
        background-color: #babfe7;  /* Green background */
        color: white;  /* White text */
        border-radius: 5px;  /* Rounded corners */
        font-weight: bold;  /* Bold text */
    }
    /* Reset button color */
    .stButton button[data-testid="baseButton-secondary"]:nth-child(2) {
        background-color: #FF0000;  /* Red background */
        color: white;  /* White text */
        border-radius: 5px;  /* Rounded corners */
        font-weight: bold;  /* Bold text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("SMS Spam Identifier")

# Initialize session state
if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ""
if 'result' not in st.session_state:
    st.session_state.result = None

# Text area for input
input_sms = st.text_area("Enter the SMS here....", value=st.session_state.input_sms)

# Update session state when input changes
if input_sms != st.session_state.input_sms:
    st.session_state.input_sms = input_sms
    st.session_state.result = None  # Clear the result when input changes

# Create two columns for Predict and Reset buttons
col1, col2 = st.columns(2)

# Predict button
with col1:
    if st.button('Predict'):
        # Check if the input is empty or contains only whitespace
        if not input_sms.strip():
            st.warning("Please enter some text to predict.")
        else:
            # Preprocess the input SMS
            transformed_sms = transform_text(input_sms)
            
            # Vectorize the transformed SMS
            vector_input = tfidf.transform([transformed_sms])
            
            # Predict using the model
            result = model.predict(vector_input[0])
            
            # Store the result in session state
            st.session_state.result = result

# Reset button
with col2:
    if st.button('Reset'):
        st.session_state.input_sms = ""  # Clear the input
        st.session_state.result = None  # Clear the result
        st.rerun()  # Force the app to rerun and update the text area

# Display the result
if st.session_state.result is not None:
    if st.session_state.result == 1:
        st.success("Spam")
    else:
        st.success("Not Spam")