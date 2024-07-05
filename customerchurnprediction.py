# customer_churn_prediction.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the customer churn dataset"""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the data by handling missing values, one-hot encoding categorical variables, and scaling numerical variables"""
    df = df.dropna()  # drop rows with missing values
    df['age'] = df['age'].apply(lambda x: x / 100)  # scale age by dividing by 100
    df['tenure'] = df['tenure'].apply(lambda x: x / 100)  # scale tenure by dividing by 100
    df = pd.get_dummies(df, columns=['plan_type'])  # one-hot encode categorical variables
    scaler = StandardScaler()
    df[['age', 'tenure', 'data_usage', 'voice_minutes', 'text_messages']] = scaler.fit_transform(df[['age', 'tenure', 'data_usage', 'voice_minutes', 'text_messages']])
    return df

def train_model(X_train, y_train):
    """Train a random forest classifier on the training data"""
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    return rfc

def evaluate_model(rfc, X_test, y_test):
    """Evaluate the model on the testing data"""
    y_pred = rfc.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

def make_predictions(rfc, new_data):
    """Make predictions on new data using the trained model"""
    new_data = pd.DataFrame(new_data)
    new_data[['age', 'tenure', 'data_usage', 'voice_minutes', 'text_messages']] = StandardScaler().fit_transform(new_data[['age', 'tenure', 'data_usage', 'voice_minutes', 'text_messages']])
    new_data = pd.get_dummies(new_data, columns=['plan_type'])
    predictions = rfc.predict(new_data.drop('churned', axis=1))
    return predictions

if __name__ == "__main__":
    file_path = "customer_churn_data.csv"
    df = load_data(file_path)
    X = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X.drop('churned', axis=1), X['churned'], test_size=0.2, random_state=42)
    rfc = train_model(X_train, y_train)
    evaluate_model(rfc, X_test, y_test)
    new_data = pd.DataFrame({'age': [30], 'tenure': [2], 'plan_type': ['gold'], 'data_usage': [100], 'voice_minutes': [50], 'text_messages': [20]})
    predictions = make_predictions(rfc, new_data)
    print(predictions)
