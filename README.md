Polynomial Curve Fitting for Temperature Trend Prediction

Machine Learning – Unit 1 Project

Project Overview

This project uses **Polynomial Regression** to model and predict daily temperature trends using the **Daily Delhi Climate dataset**.

The project studies how polynomial degree affects **underfitting, overfitting, and the bias-variance trade-off**. It also uses a **day-of-year** feature to represent the repeating annual temperature cycle.

---

Problem Statement

Daily temperature follows a seasonal pattern that cannot be represented well by a simple straight-line model.

This project investigates how different polynomial degrees behave on the temperature data and develops a model that can generalize to unseen temperature data.

---

 Dataset

**Dataset:** Daily Climate Time Series Data – Delhi

* Training period:** 2013–2016
* Held-out test period:** 2017
* Training observations:** 1461
* Test observations:** 114
* Target:** Mean Temperature (`meantemp`)

---

Objectives

* Analyze Delhi temperature data using EDA.
* Apply polynomial curve fitting.
* Study underfitting and overfitting.
* Compare different polynomial degrees.
* Improve the model using the `day_of_year` feature.
* Select a suitable polynomial degree using validation.
* Evaluate the final model on unseen 2017 data.
* Provide an interactive temperature prediction website.

---

 Methodology

1. Data Loading and Cleaning

The training and test datasets are loaded and cleaned before modelling.

2. Exploratory Data Analysis

The temperature series is analyzed to identify the strong annual seasonal pattern in Delhi temperatures.

 3. Polynomial Curve Fitting

Polynomial models with different degrees are fitted to study model complexity and the bias-variance trade-off.

 4. Improved Feature Representation

Instead of using the raw time index, the project uses **`day_of_year`** to represent the repeating yearly temperature cycle.

 5. Model Selection

Different polynomial degrees are evaluated using chronological validation.

**Selected Model:** Degree-8 Polynomial Regression

 6. Final Evaluation

The final model is trained on 2013–2016 data and evaluated on the held-out 2017 test data.

---

 Results

| Metric |       Result |
| ------ | -----------: |
| RMSE   | **2.873 °C** |
| MAE    | **2.255 °C** |
| R²     |    **0.794** |

The final degree-8 seasonal polynomial explains approximately **79.4% of the variance** in the held-out 2017 temperature data.

---

Key Findings

* Low-degree polynomials underfit the temperature pattern.
* High-degree polynomials can overfit and produce unstable oscillations.
* Using `day_of_year` provides a better representation of the repeating annual cycle.
* A **degree-8 polynomial** was selected for the final seasonal model.
* The model achieved **R² = 0.794** on unseen 2017 data.

---

 Interactive Website

The project includes an interactive website with:

* Temperature trend analysis
* Polynomial degree comparison
* Bias-variance analysis
* Model exploration
* Date-based temperature prediction
* Exploratory data analysis
* Methodology and implementation

The **Date Predictor & Forecast Simulator** allows users to select a date and obtain a predicted temperature.

---

 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* HTML
* CSS
* JavaScript
* Chart.js
* Google Colab
* GitHub

---

Project Structure

text
polynomial-temperature-prediction/
├── index.html
├── style.css
├── app.js
├── data.js
├── chart.min.js
├── DailyDelhiClimateTrain.csv
├── DailyDelhiClimateTest.csv
└── Polynomial_Curve_Fitting_Temperature_Trend.ipynb

---

Google Colab

The complete machine learning implementation, analysis, validation, and final evaluation are available in the Google Colab notebook.

Notebook:
[https://colab.research.google.com/drive/13mtObermRERdTpX-myfjZ6MIfEXHa6o2#scrollTo=3bbd9025](https://colab.research.google.com/drive/13mtObermRERdTpX-myfjZ6MIfEXHa6o2#scrollTo=3bbd9025)


 Conclusion

This project demonstrates how **Polynomial Regression** can be applied to real-world temperature data and how model performance depends on both polynomial complexity and feature representation.

Using a **degree-8 polynomial with the `day_of_year` feature** provides a useful seasonal model that generalizes to the held-out 2017 data.
