import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np
import scipy.stats as stats

st.title("Automated Model Selection and Prediction App For Regression Data")

st.sidebar.title("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    st.session_state.df = df

if 'df' in st.session_state:
    df = st.session_state.df
    st.write("Data Preview:")
    st.dataframe(df.head())

    if st.button("Perform EDA"):
        st.session_state.eda_performed = True

    if 'eda_performed' in st.session_state:
        st.write("Exploratory Data Analysis:")
        st.write(df.describe())

        # Handle categorical data for correlation matrix
        df_encoded = df.copy()
        for column in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[column] = LabelEncoder().fit_transform(df_encoded[column])

        st.write("Correlation Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(df_encoded.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    target_column = st.selectbox("Select the target column for prediction:", df.columns)

    if st.button("Preprocess Data"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical variables if necessary
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = LabelEncoder().fit_transform(X[column])

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.X = X
        st.session_state.feature_names = df.drop(columns=[target_column]).columns
        st.session_state.preprocessed = True

    if 'preprocessed' in st.session_state:
        st.write("Data Preprocessing Done!")
        
        # Show Train/Test Split
        st.write("Train/Test Split:")
        fig, ax = plt.subplots()
        ax.bar(["Train", "Test"], [len(st.session_state.X_train), len(st.session_state.X_test)], color=["blue", "orange"])
        st.pyplot(fig)

    if st.button("Compare Models and Predict"):
        st.session_state.compare_models = True

    if 'compare_models' in st.session_state:
        models = {
            "RandomForestRegressor": RandomForestRegressor(),
            "SVR": SVR(),
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet()
        }

        model_performance = []

        for name, model in models.items():
            model.fit(st.session_state.X_train, st.session_state.y_train)
            predictions = model.predict(st.session_state.X_test)

            mse = mean_squared_error(st.session_state.y_test, predictions)
            mae = mean_absolute_error(st.session_state.y_test, predictions)
            r2 = r2_score(st.session_state.y_test, predictions)

            model_performance.append({
                "Model": name,
                "MSE": mse,
                "MAE": mae,
                "R2 Score": r2
            })

        performance_df = pd.DataFrame(model_performance)
        st.write("Model Performance Comparison:")
        st.dataframe(performance_df)

        # Select the best model based on R2 Score
        best_model_name = performance_df.loc[performance_df["R2 Score"].idxmax()]["Model"]
        best_model = models[best_model_name]
        best_model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.best_model = best_model

        st.write(f"Best Model: {best_model_name}")

        predictions = best_model.predict(st.session_state.X_test)
        st.session_state.predictions = predictions

        st.write("Predictions using the best model:")
        st.write(predictions)

        st.session_state.predicted = True

    if 'predicted' in st.session_state:
        graph_option = st.selectbox("Select the graph to generate:", [
            "Scatter Plot", "Residual Plot", "Line Plot", "Histogram", "Q-Q Plot",
            "Box Plot", "Density Plot", "Cumulative Gains and Lift Charts",
            "Actual vs. Predicted Plot", "Regression Line Plot"
        ])

        if st.button("Generate"):
            predictions = st.session_state.predictions
            model = st.session_state.best_model

            def plot_scatter_plot():
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.y_test, predictions)
                ax.set_xlabel('Observed')
                ax.set_ylabel('Predicted')
                st.pyplot(fig)

            def plot_residual_plot():
                errors = predictions - st.session_state.y_test
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.y_test, errors)
                ax.hlines(y=0, xmin=min(st.session_state.y_test), xmax=max(st.session_state.y_test), colors='r')
                ax.set_xlabel('Observed')
                ax.set_ylabel('Residuals')
                st.pyplot(fig)

            def plot_line_plot():
                fig, ax = plt.subplots()
                ax.plot(st.session_state.y_test.reset_index(drop=True), label='Observed')
                ax.plot(predictions, label='Predicted')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()
                st.pyplot(fig)

            def plot_histogram():
                fig, ax = plt.subplots()
                sns.histplot(predictions, kde=True, ax=ax)
                st.pyplot(fig)

            def plot_qq_plot():
                errors = predictions - st.session_state.y_test
                fig, ax = plt.subplots()
                stats.probplot(errors, dist="norm", plot=ax)
                st.pyplot(fig)

            def plot_box_plot():
                fig, ax = plt.subplots()
                sns.boxplot(data=predictions, ax=ax)
                st.pyplot(fig)

            def plot_density_plot():
                fig, ax = plt.subplots()
                sns.kdeplot(predictions, ax=ax, fill=True)
                st.pyplot(fig)

            def plot_cumulative_gain_chart():
                # Placeholder for Cumulative Gain and Lift Chart
                st.write("Cumulative Gain and Lift Chart not implemented")

            def plot_actual_vs_predicted():
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.y_test, predictions)
                ax.plot([min(st.session_state.y_test), max(st.session_state.y_test)], 
                        [min(st.session_state.y_test), max(st.session_state.y_test)], 'r')
                ax.set_xlabel('Observed')
                ax.set_ylabel('Predicted')
                st.pyplot(fig)

            def plot_regression_line_plot():
                fig, ax = plt.subplots()
                sns.regplot(x=st.session_state.y_test, y=predictions, ax=ax)
                ax.set_xlabel('Observed')
                ax.set_ylabel('Predicted')
                st.pyplot(fig)

            if graph_option == "Scatter Plot":
                plot_scatter_plot()
            elif graph_option == "Residual Plot":
                plot_residual_plot()
            elif graph_option == "Line Plot":
                plot_line_plot()
            elif graph_option == "Histogram":
                plot_histogram()
            elif graph_option == "Q-Q Plot":
                plot_qq_plot()
            elif graph_option == "Box Plot":
                plot_box_plot()
            elif graph_option == "Density Plot":
                plot_density_plot()
            elif graph_option == "Cumulative Gains and Lift Charts":
                plot_cumulative_gain_chart()
            elif graph_option == "Actual vs. Predicted Plot":
                plot_actual_vs_predicted()
            elif graph_option == "Regression Line Plot":
                plot_regression_line_plot()
