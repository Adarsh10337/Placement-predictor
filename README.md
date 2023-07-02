# Placement-predictor
This project serves as a valuable tool for students, educational institutions, and recruiters to gain insights into the factors that influence campus placements. It empowers students to make informed decisions, helps institutions optimize their placement programs, and enables recruiters to identify potential candidates effectively.

# Campus Placement Prediction

This repository contains code for a machine learning model to predict campus placement based on student data. The model is built using Python and scikit-learn library.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Campus Placement Prediction model is designed to predict whether a student will be placed or not based on various factors such as academic performance, work experience, and specialization. The model is trained on a dataset of student records and uses a Random Forest Classifier algorithm.

## Features

The model takes the following features as input for prediction:
- Secondary School Percentage (ssc_p)
- Secondary School Board (ssc_b)
- Higher Secondary School Percentage (hsc_p)
- Higher Secondary School Board (hsc_b)
- Higher Secondary School Stream (hsc_s)
- Undergraduate Degree Percentage (degree_p)
- Undergraduate Degree Type (degree_t)
- Employability Test Percentage (etest_p)
- MBA Specialization (specialisation)
- MBA Percentage (mba_p)

The target variable is the placement status (status), which can be "Placed" or "Not Placed".

## Requirements

To run the model locally, you need the following dependencies:
- Python (version 3.x)
- scikit-learn library
- pandas library
- Flask library

## Installation

1. Clone the repository:

git clone https://github.com/Adarsh10337/Placement-predictor.git


2. Install the required dependencies:


## Usage

1. Prepare your data:
- Create a CSV file containing student data with the required features.
- Make sure the CSV file follows the same format as the provided `train.csv` file.

2. Train the model:
- Run the `train_model.py` script to train the Random Forest Classifier model on your data.

3. Run the Flask app:
- Start the Flask app by running the `app.py` script.
- Access the app in your browser at `http://localhost:5000`.

4. Predict the placement status:
- Enter the student data in the provided form on the web page.
- Click the "Predict" button to see the predicted placement status.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
