import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report, cohen_kappa_score, matthews_corrcoef, precision_recall_curve, average_precision_score, log_loss, brier_score_loss

# Streamlit app
def main():
    st.title("Machine Learning Model Training and Evaluation")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Read data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())

        # Select target column
        target_column = df.columns[-1]

        # Select scaler type
        scaler_type = st.selectbox("Select the scaler type", ["standard", "minmax"])

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

        # Select model
        model_name = st.selectbox("Select the model", ["Logistic Regression", "SVM", "Random Forest", "XGBoost"])
        
        if model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "SVM":
            model = SVC(probability=True)
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        if st.button("Train Model"):
            # Train model
            trained_model = train_model(X_train, y_train, model, model_name)

            # Evaluate model
            metrics = evaluate_model(trained_model, X_test, y_test)

            # Display evaluation metrics
            st.write("Model Evaluation Metrics:")
            st.write(f"Accuracy: {metrics['accuracy']}")
            st.write(f"Confusion Matrix: \n{metrics['confusion_matrix']}")
            st.write(f"ROC AUC Score: {metrics['roc_auc']}")
            st.write(f"Cohen Kappa Score: {metrics['cohen_kappa']}")
            st.write(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']}")

            # Display classification report
            st.write("Classification Report:")
            classification_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.write(classification_df)

            st.write(f"Log Loss: {metrics['log_loss']}")
            st.write(f"Brier Score Loss: {metrics['brier_score_loss']}")
            st.write(f"Average Precision Score: {metrics['average_precision']}")

            # Precision-Recall Curve
            st.write("Precision-Recall Curve:")
            precision, recall = metrics['precision_recall_curve']
            st.line_chart({"Precision": precision, "Recall": recall})


# Step 2: Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if there are only numerical or categorical columns
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical_cols) == 0:
        pass
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Impute missing values for numerical columns (mean imputation)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

        # Scale the numerical features based on scaler_type
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if len(categorical_cols) == 0:
        pass
    else:
        # Impute missing values for categorical columns (mode imputation)
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

        # One-hot encode categorical features
        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
        X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test

# Step 3: Train the model
def train_model(X_train, y_train, model, model_name):
    # Train the selected model
    model.fit(X_train, y_train)

    # Ensure the directory exists
    save_dir = "trained_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the trained model
    with open(os.path.join(save_dir, f"{model_name}.pkl"), 'wb') as file:
        pickle.dump(model, file)

    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    
    # Assuming y_score is the predicted probabilities for the positive class
    y_score = model.predict_proba(X_test)[:, 1]

    # Reshape y_score for binary classification
    y_score = y_score.reshape(-1, 1)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    log_loss_value = log_loss(y_test, y_pred)
    brier_score = brier_score_loss(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    metrics = {
        "accuracy": round(accuracy, 2),
        "confusion_matrix": conf_matrix,
        "roc_auc": round(roc_auc, 2),
        "cohen_kappa": round(kappa, 2),
        "matthews_corrcoef": round(mcc, 2),
        "classification_report": class_report,
        "log_loss": round(log_loss_value, 2),
        "brier_score_loss": round(brier_score, 2),
        "average_precision": round(average_precision, 2),
        "precision_recall_curve": (precision, recall)
    }

    return metrics

if __name__ == "__main__":
    main()

