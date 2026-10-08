import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_churn_model():
    # Step 1: Load the raw dataset
    print("Loading dataset from local storage...")
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Step 2: Handle categorical encoding for string columns
    print("Preprocessing and encoding categorical features...")
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])
        
    # Step 3: Define feature matrix (X) and target vector (y)
    X = data.drop("Churn", axis=1)
    y = data["Churn"]
    
    # Step 4: Initialize and train the Random Forest Classifier
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Step 5: Serialize and save the trained model using pickle
    print("Saving the trained model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Model successfully trained and saved as model.pkl!")

if __name__ == "__main__":
    train_churn_model()