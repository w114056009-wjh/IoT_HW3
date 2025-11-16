
import streamlit as st
import pickle
import re

# 載入模型和向量化器
with open('spam_classifier_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

st.title('Email Spam Classifier')
st.write('Enter a message to classify it as spam or not spam (ham).')

# 清理文本函數
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    return text

# 輸入框
user_input = st.text_area('Message:')

if st.button('Classify'):
    if user_input:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)
        result = 'Spam' if prediction[0] == 1 else 'Ham'
        st.subheader(f'Prediction: **{result}**')
    else:
        st.warning('Please enter a message.')

st.header('Model Performance')
st.image('confusion_matrix.png', caption='Confusion Matrix')
    