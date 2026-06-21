import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_loader import download_and_load_data

def train_and_save_model(model_path="parkinsons_model.pkl"):
    """
    Loads data, trains a Random Forest Classifier, and saves the model binary.
    """
    # 1. Load data
    X, y = download_and_load_data()
    
    # 2. Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    # 3. Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # 5. Save the trained model
    joblib.dump(model, model_path)
    print(f"Model successfully saved to {model_path}")
    return model

if __name__ == "__main__":
    train_and_save_model()