import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle

# 載入資料集
df = pd.read_csv('datasets/sms_spam_no_header.csv', encoding='latin-1')
df.columns = ['label', 'message']

# 清理文本
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# TF-IDF 向量化
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['cleaned_message'])
y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

# 儲存處理好的資料和向量化器
with open('processed_data.pkl', 'wb') as f:
    pickle.dump((X, y, vectorizer), f)

print('Data preprocessing complete. Processed data saved to processed_data.pkl')
    