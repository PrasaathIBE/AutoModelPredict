import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc,
    log_loss
)
from sklearn.calibration import calibration_curve
import numpy as np

st.title("Automated Model Selection and Prediction App For Categorical Data")

st.sidebar.title("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    for column in X.select_dtypes(include=['object']).columns:
        X[column] = LabelEncoder().fit_transform(X[column])

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def evaluate_models(X_train, y_train, X_val, y_val):
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "LogisticRegression": LogisticRegression(),
        "GradientBoosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier()
    }

    best_model = None
    best_accuracy = 0
    model_accuracies = {}
    st.session_state.train_errors = []
    st.session_state.test_errors = []
    st.session_state.train_accuracies = []
    st.session_state.test_accuracies = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        model_accuracies[model_name] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

        # Simulate tracking loss and accuracy over epochs
        for epoch in range(1, 11):
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            val_loss = log_loss(y_val, model.predict_proba(X_val))
            train_accuracy = accuracy_score(y_train, train_predictions)
            val_accuracy = accuracy_score(y_val, val_predictions)

            st.session_state.train_errors.append(train_loss)
            st.session_state.test_errors.append(val_loss)
            st.session_state.train_accuracies.append(train_accuracy)
            st.session_state.test_accuracies.append(val_accuracy)

    return best_model, best_accuracy, model_accuracies

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.write("Data Preview:")
    st.dataframe(df.head())

if 'df' in st.session_state:
    df = st.session_state.df
    if st.button("Perform EDA"):
        st.session_state.show_eda = True
        df_encoded = df.copy()
        for column in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[column] = LabelEncoder().fit_transform(df_encoded[column])
        st.session_state.df_encoded = df_encoded

    if 'show_eda' in st.session_state and st.session_state.show_eda:
        st.write("Exploratory Data Analysis:")
        st.write(df.describe())
        fig, ax = plt.subplots()
        sns.heatmap(st.session_state.df_encoded.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    target_column = st.selectbox("Select the target column for prediction:", df.columns)

    if st.button("Preprocess Data and Select Best Model"):
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.feature_names = df.drop(columns=[target_column]).columns
        st.session_state.preprocessed = True
        best_model, best_accuracy, model_accuracies = evaluate_models(X_train, y_train, X_test, y_test)
        st.session_state.best_model = best_model
        st.session_state.best_accuracy = best_accuracy
        st.session_state.model_accuracies = model_accuracies

    if 'preprocessed' in st.session_state and st.session_state.preprocessed:
        st.write("Data Preprocessing Done!")
        st.write("Model Accuracies:")
        st.write(st.session_state.model_accuracies)
        st.write(f"Best Model Selected: {type(st.session_state.best_model).__name__} with Accuracy: {st.session_state.best_accuracy}")

        if st.button("Predict with Best Model"):
            best_model = st.session_state.best_model
            best_model.fit(st.session_state.X_train, st.session_state.y_train)
            predictions = best_model.predict(st.session_state.X_test)
            probabilities = best_model.predict_proba(st.session_state.X_test)[:, 1]

            st.session_state.predictions = predictions
            st.session_state.probabilities = probabilities

            accuracy = accuracy_score(st.session_state.y_test, predictions)
            st.session_state.predicted = True

    if 'predicted' in st.session_state and st.session_state.predicted:
        st.write(f"Model Accuracy: {accuracy_score(st.session_state.y_test, st.session_state.predictions)}")
        st.write("Predictions:")
        st.write(st.session_state.predictions)

        graph_option = st.selectbox("Select the graph to generate:", [
            "Confusion Matrix", "Precision-Recall Curve", "ROC Curve", "PR AUC",
            "ROC AUC", "Loss Curve", "Accuracy Curve", "Prediction Distribution",
            "Error Analysis", "Calibration Curve", "Feature Importance"
        ])

        if st.button("Generate"):
            predictions = st.session_state.predictions
            probabilities = st.session_state.probabilities
            model = st.session_state.best_model

            def plot_confusion_matrix():
                cm = confusion_matrix(st.session_state.y_test, predictions)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                st.pyplot(fig)

            def plot_precision_recall_curve():
                precision, recall, _ = precision_recall_curve(st.session_state.y_test, probabilities)
                fig, ax = plt.subplots()
                ax.plot(recall, precision, marker='.')
                ax.set_title('Precision-Recall Curve')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                st.pyplot(fig)

            def plot_roc_curve():
                fpr, tpr, _ = roc_curve(st.session_state.y_test, probabilities)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, marker='.')
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                st.pyplot(fig)

            def plot_pr_auc():
                precision, recall, _ = precision_recall_curve(st.session_state.y_test, probabilities)
                pr_auc = auc(recall, precision)
                st.write(f"PR AUC: {pr_auc}")

            def plot_roc_auc():
                fpr, tpr, _ = roc_curve(st.session_state.y_test, probabilities)
                roc_auc = auc(fpr, tpr)
                st.write(f"ROC AUC: {roc_auc}")

            def plot_loss_curve():
                fig, ax = plt.subplots()
                ax.plot(st.session_state.train_errors, label='Train Loss')
                ax.plot(st.session_state.test_errors, label='Test Loss')
                ax.set_title('Loss Curve')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

            def plot_accuracy_curve():
                fig, ax = plt.subplots()
                ax.plot(st.session_state.train_accuracies, label='Train Accuracy')
                ax.plot(st.session_state.test_accuracies, label='Test Accuracy')
                ax.set_title('Accuracy Curve')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                st.pyplot(fig)

            def plot_prediction_distribution():
                fig, ax = plt.subplots()
                sns.histplot(predictions, kde=True, ax=ax)
                ax.set_title('Prediction Distribution')
                ax.set_xlabel('Predictions')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            def plot_error_analysis():
                errors = predictions != st.session_state.y_test
                fig, ax = plt.subplots()
                sns.histplot(errors, kde=True, ax=ax)
                ax.set_title('Error Analysis')
                ax.set_xlabel('Errors')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

            def plot_error_box_plot():
                errors = predictions - st.session_state.y_test
                fig, ax = plt.subplots()
                sns.boxplot(data=errors, ax=ax)
                ax.set_title('Error Box Plot')
                ax.set_xlabel('Errors')
                st.pyplot(fig)

            def plot_calibration_curve():
                fig, ax = plt.subplots()
                fraction_of_positives, mean_predicted_value = calibration_curve(st.session_state.y_test, probabilities, n_bins=10)
                ax.plot(mean_predicted_value, fraction_of_positives, "s-")
                ax.set_title('Calibration Curve')
                ax.set_xlabel('Mean Predicted Value')
                ax.set_ylabel('Fraction of Positives')
                st.pyplot(fig)

            def plot_feature_importance():
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                fig, ax = plt.subplots()
                ax.bar(range(len(importances)), importances[indices], align='center')
                plt.xticks(range(len(importances)), st.session_state.feature_names[indices], rotation=90)
                ax.set_title('Feature Importance')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                st.pyplot(fig)

            if graph_option == "Confusion Matrix":
                plot_confusion_matrix()
            elif graph_option == "Precision-Recall Curve":
                plot_precision_recall_curve()
            elif graph_option == "ROC Curve":
                plot_roc_curve()
            elif graph_option == "PR AUC":
                plot_pr_auc()
            elif graph_option == "ROC AUC":
                plot_roc_auc()
            elif graph_option == "Loss Curve":
                plot_loss_curve()
            elif graph_option == "Accuracy Curve":
                plot_accuracy_curve()
            elif graph_option == "Prediction Distribution":
                plot_prediction_distribution()
            elif graph_option == "Error Analysis":
                plot_error_analysis()
            elif graph_option == "Error Box Plot":
                plot_error_box_plot()
            elif graph_option == "Calibration Curve":
                plot_calibration_curve()
            elif graph_option == "Feature Importance":
                plot_feature_importance()
