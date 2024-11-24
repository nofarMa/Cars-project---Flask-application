
Overview

FlaskCarPricePredictor is a web app that predicts car prices based on user input. It uses a trained ElasticNet regression model and provides a simple interface for entering car details and viewing predictions.

Features

1.Predicts car prices based on attributes like manufacturer, year, engine capacity, and mileage.

2.Preprocesses user input to match the trained model structure.

3.Displays predictions directly on the web interface.

4.Ensures predictions are non-negative.

Tech Stack

Backend: Flask (Python)

Frontend: HTML (via Jinja templates)

Machine Learning Model: ElasticNet regression (pre-trained and saved as a .pkl file)

How It Works

User Input: Users enter car details in the form on the web interface.

Data Processing: The app preprocesses the input using the prepare_data function to align it with the model's requirements.

Prediction: The pre-trained model predicts the car price.

Display: The predicted price is displayed on the same page.
