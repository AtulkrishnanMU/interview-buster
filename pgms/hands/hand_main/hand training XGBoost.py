import csv
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib

RANDOM_SEED = 42

dataset = 'E:\project demo\pgms\hands\hand_main\model\keypoint_classifier\keypoint.csv'
model_save_path = r'E:\project demo\pgms\hands\hand_main\model\keypoint_classifier\xgboost_model.pkl'

NUM_CLASSES = 4

# Load dataset
data = np.loadtxt(dataset, delimiter=',', dtype='float32')
X_dataset = data[:, 1:]
y_dataset = data[:, 0].astype('int32')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# XGBoost model
xgboost_model = xgb.XGBClassifier(objective='multi:softmax', num_class=NUM_CLASSES, random_state=RANDOM_SEED)

# Model training
xgboost_model.fit(X_train, y_train)

# Model evaluation
y_pred = xgboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

# Save the XGBoost model
joblib.dump(xgboost_model, model_save_path)

# Inference test
test_sample = X_test[0].reshape(1, -1)
predict_result = xgboost_model.predict(test_sample)
print(predict_result)
