import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1) Load the data
df = pd.read_csv("dataset.csv")

# 2) Drop rows with missing values in features or target variable
features = [
    'Heart Rate', 'Body Temperature', 'Oxygen Saturation',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure'
]
df = df.dropna(subset=features + ['Risk Category'])

# 3) Prepare data
X = df[features]
y = df['Risk Category']

# 4) Split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5) Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6) Make predictions on the test data
y_pred = model.predict(X_test)

# 7) Evaluate the model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print("="*50)
print("📊 Model Evaluation:")
print(f"Train Accuracy: {train_acc:.2f}")
print(f"Test Accuracy: {test_acc:.2f}")
print("="*50)

print("🔎 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("="*50)

print("📄 Classification Report:\n", classification_report(y_test, y_pred))
print("="*50)

# 8) Save the trained model
joblib.dump(model, "health_risk_model.pkl")
print("✅ The model has been saved as health_risk_model.pkl")

# 9) Function to identify possible causes if the reading is abnormal
def get_possible_causes(reading):
    heart_rate, temp, spo2, systolic_bp, diastolic_bp = reading
    reasons = []

    if heart_rate < 60:
        reasons.append("Bradycardia (slow heart rate)")
    elif heart_rate > 100:
        reasons.append("Tachycardia (fast heart rate)")

    if temp > 37.5:
        reasons.append("Fever (high temperature)")
    elif temp < 36:
        reasons.append("Hypothermia (low temperature)")

    if spo2 < 95:
        reasons.append("Hypoxia (low oxygen saturation)")

    if systolic_bp > 130 or diastolic_bp > 90:
        reasons.append("Hypertension (high blood pressure)")
    elif systolic_bp < 90 or diastolic_bp < 60:
        reasons.append("Hypotension (low blood pressure)")

    return reasons

# 10) Test the function with a new reading (example)
new_reading = [110, 38.0, 92, 145, 95]  # Example of an abnormal reading

prediction = model.predict([new_reading])
if prediction[0] == 0:
    print("✅ The reading is normal")
else:
    print("⚠️ The reading is abnormal")
    causes = get_possible_causes(new_reading)
    print("🚨 Possible causes:")
    for cause in causes:
        print("-", cause)
