
![FrontEnd_Personality Prediction](https://github.com/nikhil-3112/Personality-Prediction/assets/98270496/75488b3a-23e7-4759-a1cc-e00a202c9565)

# Streamlit Machine Learning Dashboard

This project is a machine learning dashboard built with Streamlit, aimed at predicting the personality of individuals based on the Big Five personality traits using data mining techniques and advanced artificial neural network algorithms.

## Dependencies

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/) (version 2.5.1)
- [Keras](https://keras.io/) (nightly build, version 2.5.0.dev2021032900)
- [Seaborn](https://seaborn.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/your_project.git
cd your_project
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Data Collection

The dataset used for this project has been imported from the Kaggle website. Data collection is the crucial first step, where accurate insights are collected, measured, and analyzed using standard validated techniques. The dataset contains attributes such as age, gender, and the Big Five personality traits: Openness, Neuroticism, Extraversion, Agreeableness, and Conscientiousness.

## Attribute Selection

Attributes of the dataset play a significant role in the accuracy and efficiency of the machine learning model. In this project, attributes such as age, gender, and the Big Five personality traits were selected for personality prediction. Feature selection is essential for reducing dimensionality and avoiding overfitting.

## Data Preprocessing

Preprocessing of the dataset is essential for achieving optimal results from machine learning algorithms. This involves sampling to select a subset of the dataset, handling missing values, encoding categorical variables, and scaling numerical features. Preprocessing ensures that the data is in a suitable format for training the machine learning model.

## Problem Statement

Personality assessment has become a widely used method for hiring employees and evaluating candidates. Classifying personality based on the Big Five traits using data mining techniques provides a convenient way to judge individuals. The goal of this project is to build a model that accurately predicts the personality of individuals based on their responses. An advanced artificial neural network algorithm is employed for classification, and the predicted personality type is displayed to the user.

## Usage

Once the Streamlit application is running, users can interactively explore the dataset, preprocess the data, train machine learning models, and visualize results. The dashboard provides an intuitive interface for predicting personality types based on input data.

## Contributing

Contributions to this project are welcome! If you have suggestions for new features, bug fixes, or improvements, please open an issue or submit a pull request.
