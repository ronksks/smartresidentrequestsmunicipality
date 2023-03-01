# Smart Resident Requests Municipality
This is a machine learning system designed to classify complaints according to their department. It is built using Python 3 and the Scikit-Learn and Pandas libraries.

## Usage
### To use the system, you need to have Python 3 installed, as well as the Scikit-Learn and Pandas libraries. Once you have those, you can run the main Python file, main.py, by typing the following command in your terminal:

python main.py
This will run the system, which will perform the following steps:

Read the data from an Excel file (complaints.xlsx by default).<br />
Split the data into training, validation, and testing sets.<br />
Pre-process the data by removing stop words and punctuation, and performing stemming (if using the default pre-processing method).<br />
Extract features from the pre-processed data using either TF-IDF or bag-of-words.<br />
Train several machine learning models on the training data.<br />
Evaluate the models on the validation data and select the best one.<br />
Save the best model to a file.<br />
Test the best model on the testing data and print the results.<br />
## Files
The project consists of two main Python files:<br />

main.py: This is the main file that you should run to use the system. It imports the Classification class from the classification.py file and uses it to perform the classification.<br />
classification.py: This file contains the Classification class, which is responsible for performing the classification. It has several methods for reading the data, splitting it into sets, pre-processing it, extracting features, training and evaluating models, and saving the final model.<br />
