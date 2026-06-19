import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

print("🚀 Starting Model Training Process...")

# 1. Load the dataset
# Navigate to the Dataset folder (go up one level, then into Dataset)
try:
    df = pd.read_csv(r"C:\Users\Siddhant\OneDrive\Desktop\Employee_Attrition_Prediction\Dataset\Employee_Attrition_DATASET.csv")
    print("✅ Dataset loaded successfully!")
    print(f"   Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
except FileNotFoundError:
    print("❌ Error: Dataset file not found!")
    print("   Please check: C:/Users/Siddhant/OneDrive/Desktop/Employee_Attrition_Prediction/Dataset/")
    exit()

# 2. Data Preprocessing
# Convert target variable to binary (1 for 'Yes', 0 for 'No')
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
print(f"✅ Target variable encoded! (0 = No, 1 = Yes)")
print(f"   Attrition distribution: {df['Attrition'].value_counts().to_dict()}")

# ALL columns needed for the Streamlit dashboard (Updated!)
selected_columns = [
    'Age', 
    'MonthlyIncome', 
    'NumCompaniesWorked', 
    'YearsAtCompany', 
    'YearsSinceLastPromotion', 
    'EnvironmentSatisfaction', 
    'JobInvolvement',
    'OverTime', 
    'BusinessTravel',
    # NEW: Added missing columns from Streamlit app
    'JobSatisfaction',
    'WorkLifeBalance', 
    'TotalWorkingYears',
    'Education',
    'Department',
    'Gender'
]

# Check if all columns exist in the dataset
missing_cols = [col for col in selected_columns if col not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing columns in dataset: {missing_cols}")
    print("   Please check your dataset has all these columns.")
    exit()

X = df[selected_columns]
y = df['Attrition']

print(f"✅ Selected {len(selected_columns)} features for training!")

# Convert categorical text data into numerical values using One-Hot Encoding
X_encoded = pd.get_dummies(X, drop_first=True)
print(f"✅ After encoding: {X_encoded.shape[1]} features (from {len(selected_columns)} original)")

# 3. Save the features.pkl file in the Model folder (current directory)
feature_cols = X_encoded.columns.tolist()

# Create Model folder if it doesn't exist
os.makedirs('.', exist_ok=True)

pickle.dump(feature_cols, open('features.pkl', 'wb'))
print("✅ features.pkl saved successfully in Model folder!")

# 4. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Data split complete!")
print(f"   Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 5. Scale the features and save the scaler.pkl file
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler in Model folder
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("✅ scaler.pkl saved successfully in Model folder!")

# 6. Train the Random Forest Model
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=10,
    min_samples_split=5
)
model.fit(X_train_scaled, y_train)

# Evaluate the model
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)

print(f"✅ Model trained successfully!")
print(f"   Training Accuracy: {train_accuracy:.2%}")
print(f"   Testing Accuracy: {test_accuracy:.2%}")

# Save the model in Model folder
pickle.dump(model, open('rf_model.pkl', 'wb'))
print("✅ rf_model.pkl saved successfully in Model folder!")

# 7. Save feature importance (bonus!)
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("✅ feature_importance.csv saved!")

print("\n🎉 Model Training Complete!")
print("📁 All files ready in Model folder:")
print("   ✅ features.pkl")
print("   ✅ scaler.pkl")
print("   ✅ rf_model.pkl")
print("   ✅ feature_importance.csv")
print("\n✨ Now run your Streamlit app with: streamlit run model.py")