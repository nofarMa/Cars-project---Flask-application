from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from car_data_prep import prepare_data

# Load the trained model
model = pickle.load(open('trained_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    input_data = {
        'Year': int(request.form['year']),
        'manufactor': request.form['manufacturer'],
        'model': request.form['model'],
        'Hand': int(request.form['hand']),
        'Gear': request.form['gear'],
        'capacity_Engine': int(request.form['capacity_Engine']),
        'Km': int(request.form['km']),
        'Color': request.form['color'],
        'Engine_type': request.form['engine_type']
    }

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Add the necessary columns with default values
    default_values = {
        'Prev_ownership': 'פרטית',  # ערך ברירת מחדל לבעלות קודמת
        'Curr_ownership': 'פרטית',  # ערך ברירת מחדל לבעלות נוכחית
        'City': None,
        'Pic_num': np.nan,
        'Cre_date': pd.NaT,
        'Repub_date': pd.NaT,
        'Description': None
    }

    for col, default_value in default_values.items():
        if col not in input_df.columns:
            input_df[col] = default_value

    # Preprocess the input data using prepare_data function
    processed_data = prepare_data(input_df)

    # Make prediction
    predicted_price = model.predict(processed_data)[0]

    # Check if predicted price is negative
    if predicted_price < 0:
        predicted_price = 0

    return render_template('index.html', prediction=f"The predicted car price is: {predicted_price:.2f} ₪")

if __name__ == '__main__':
    app.run(debug=True)
