
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# 載入資料和模型
with open('processed_data.pkl', 'rb') as f:
    X, y, _ = pickle.load(f)
with open('spam_classifier_model.pkl', 'rb') as f:
    model, _ = pickle.load(f)

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)

# 產生分類報告
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 繪製混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

print('Confusion matrix saved to confusion_matrix.png')
    