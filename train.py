
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 載入資料
with open('processed_data.pkl', 'rb') as f:
    X, y, vectorizer = pickle.load(f)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# 評估並儲存模型
y_pred = model.predict(X_test)
print(f'Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')

with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print('Model training complete. Model saved to spam_classifier_model.pkl')
    