# Polynomial Curve Fitting for Temperature Trend Prediction

## Overview
This project uses Polynomial Regression to model and predict daily temperature trends using the Delhi Daily Climate dataset.

The project studies how polynomial degree affects underfitting and overfitting and uses a day-of-year feature to model the repeating annual temperature pattern.

## Dataset
Daily Delhi Climate dataset (2013–2017).

Training data: 2013–2016  
Held-out test data: 2017

## Methodology
- Data loading and cleaning
- Exploratory Data Analysis
- Day-of-year feature engineering
- Polynomial Regression
- Bias–variance analysis
- Degree selection using validation
- Evaluation on held-out 2017 data

## Final Model
Degree-8 Polynomial Regression using the `day_of_year` feature.

## Results
- RMSE: 2.873 °C
- MAE: 2.255 °C
- R²: 0.794

## Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- HTML
- CSS
- JavaScript

## Project Structure

```text
polynomial-temperature-prediction/
├── index.html
├── style.css
├── app.js
├── data.js
├── chart.min.js
├── DailyDelhiClimateTrain.csv
├── DailyDelhiClimateTest.csv
└── Polynomial_Curve_Fitting_Temperature_Trend.ipynb
