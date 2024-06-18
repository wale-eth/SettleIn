from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('settle index.html')


@app.route('/process_data', methods=['POST'])
def process_data():
    try:
        # Retrieve data from the form
        first_name = request.form.get('first_name')
        ethnicity = request.form.get('ethnicity')

        # Get selected priorities from the form
        priority1 = request.form.get('priority1')
        priority2 = request.form.get('priority2')
        priority3 = request.form.get('priority3')
        priority4 = request.form.get('priority4')

        # Map user's ethnicity to corresponding column
        user_ethnicity_column = f"{ethnicity}"

        # Feature Engineering with Weighted Priorities
        weights = {
            priority1: 0.4,
            priority2: 0.3,
            priority3: 0.2,
            priority4: 0.1,
        }

        # Print the variables
        print("First Name:", first_name)
        print("Ethnicity:", ethnicity)
        print("Priority 1:", priority1)
        print("Priority 2:", priority2)
        print("Priority 3:", priority3)
        print("Priority 4:", priority4)

        # Initialize dictionary to build DataFrame
        prefs_dict = {}

        # Add weighted priorities
        selected_qualities = [priority1, priority2, priority3, priority4]
        for quality in selected_qualities:
            prefs_dict[quality] = weights[quality]

        # Conditionally add ethnicity
        if "High Ethnic Population" in prefs_dict:
            new_key = ethnicity
            prefs_dict[new_key] = prefs_dict.pop("High Ethnic Population")

        # Create DataFrame
        user_df = pd.DataFrame(prefs_dict, index=[0])

        print("user_df:")
        print(user_df)

        # Load and clean the data
        cleaned = pd.read_csv(r'C:\Users\HP\PycharmProjects\SettleIn\venv\10merge_final.csv')
        df = cleaned.copy()
        column_rename_dict = {
            'Crime Rate (%)': 'Low Crime Rate',
            'Unemployed (%)': 'Low Unemployment Rate',
            'COL Index': 'Low Average Cost of Living',
            'Rent Index': 'Low Average Rent',
            'IMD (Average Score)': 'High IMD Score'
        }
        df.rename(columns=column_rename_dict, inplace=True)

        # Get first 5 rows of df
        print(df.head())

        # Get columns common to both DataFrames
        common_cols = user_df.columns.intersection(df.columns)

        print("Common_cols:")
        print(common_cols)

        # Add these lines to your code before the common_cols calculation
        print("Columns in user_df:")
        print(user_df.columns)

        print("Columns in df:")
        print(df.columns)

        # Include numeric columns
        # numeric_cols = df.select_dtypes('number').columns
        selected_cols = list(common_cols) + ['Name', 'City']

        # Subset those columns + additional needed ones
        selected_df = df[selected_cols].copy()

        print(selected_df.head())
        print(selected_df.shape)
        print(selected_df.info())

        # Normalize numeric columns
        numeric_cols = selected_df.select_dtypes('number').columns
        selected_df[numeric_cols] = (selected_df[numeric_cols] - selected_df[numeric_cols].min()) / (
                selected_df[numeric_cols].max() - selected_df[numeric_cols].min())

        print("Normalized DataFrame:")
        print(selected_df[numeric_cols])
        print(selected_df[numeric_cols].info())

        # Calculate score
        scores = []

        for index, row in selected_df.iterrows():
            score = 0
            for col in common_cols:
                weight = user_df.at[0, col]
                value = row[col]
                score += weight * value

            scores.append(score)

        selected_df['Composite Score'] = scores

        print("DataFrame with Composite Score:")
        print(selected_df[['Name', 'City', 'Composite Score']])

        # Print selected_df[numeric_cols] before the train/test split
        print("Before train/test split:")
        print(selected_df[numeric_cols])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(selected_df[numeric_cols], selected_df['Composite Score'],
                                                            test_size=0.2, random_state=42)

        # Print X_train and y_train to debug
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)

        # Use the best hyperparameters obtained from grid search
        best_params = {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'alpha': 0.01,
                       'learning_rate': 'adaptive'}

        # Initialize the MLP model with the best hyperparameters
        best_mlp_model = MLPRegressor(**best_params, random_state=42)

        # Train the MLP model
        best_mlp_model.fit(X_train, y_train)

        # Make predictions
        user_recommendations = best_mlp_model.predict(selected_df[numeric_cols])

        # Normalize predictions
        user_recommendations = user_recommendations.reshape(-1, 1)
        scaler = MinMaxScaler()
        user_recommendations = scaler.fit_transform(user_recommendations)
        user_recommendations = user_recommendations.flatten()

        # Get top 5 recommendation indices
        sorted_idx = user_recommendations.argsort()[::-1][:5]

        # Get top 5 rows
        cols = ['Name', 'City', 'Composite Score']
        top_recommendations = selected_df[cols].iloc[sorted_idx]

        # Add rank and percentage match columns
        top_recommendations['Rank'] = range(1, len(top_recommendations) + 1)
        top_recommendations['Percentage Match'] = (user_recommendations[sorted_idx] * 100).round(2)

        # Sort and format
        top_recommendations = top_recommendations.sort_values('Percentage Match', ascending=False)
        top_recommendations = top_recommendations.drop(columns=['Composite Score'])

        # Create a DataFrame for priority choices
        priority_data = {'Priority': ['1st Priority', '2nd Priority', '3rd Priority', '4th Priority'],
                         'Choice': [priority1, priority2, priority3, priority4]}
        priority_df = pd.DataFrame(priority_data)

        # Create an HTML table for priority choices
        priority_table_html = priority_df.to_html(index=False, classes='table table-striped')

        # Greeting text with priority choices
        greeting_text = f"Hi {first_name},\n\nThese were your selections;\n\n{priority_table_html}\nWe are proud to share the following ward suggestions as the best fit given your entry. Congrats!\n"

        # Render tables
        priority_table_html = priority_df.to_html(index=False, classes='table table-striped')
        recommendations_table_html = top_recommendations.to_html(classes='table table-striped')

        # Pass the helpful text and tables to the template
        return render_template('settle index.html', greeting=greeting_text, priority_table=priority_table_html,
                               recommendations=recommendations_table_html)


    except Exception as e:
        return render_template('error.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
