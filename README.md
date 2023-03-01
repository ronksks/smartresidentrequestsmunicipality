# Smart Resident Requests Municipality
### This is a machine learning system designed to classify complaints according to their department. <br />It is built using Python 3 and the Scikit-Learn and Pandas libraries.

## Usage in briefly
### To use the system, you need to have Python 3 installed, as well as the Scikit-Learn and Pandas libraries. <br />Once you have those, you can run the main Python file, main.py, by typing the following command in your terminal:

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

main.py: This is the main file that you should run to use the system. <br />
It imports the Classification class from the classification.py file and uses it to perform the classification.<br />
classification.py: This file contains the Classification class, which is responsible for performing the classification. <br />
It has several methods for reading the data, splitting it into sets, pre-processing it, extracting features, training and evaluating models, and saving the final model.<br />

## Requirements
This project requires the following Python libraries to be installed:

pyodbc, 
wtforms, 
matplotlib, 
seaborn, 
scikit-learn, 
numpy, 
pandas, 
stanfordnlp, 
Flask==1.1.1

## File Structure
main.py - the main Python file that runs the Flask web application.<br />
classification.py - a file containing functions for text classification.<br />
version.py - a Python file containing the version number of the project.<br />
templates/ - a directory containing the HTML templates for the web application<br />.
static/ - a directory containing the CSS and JavaScript files for the web application.<br />
## Classes
The following Python classes are defined in version.py:<br />

Person: A class representing a person.<br />
Department: A class representing a department.<br />
Complaint: A class representing a complaint.<br />

## Running the Application
To run the application, run python main.py in the terminal.

## Usage in elaboration
When the application is running, navigate to http://localhost:5000 in a web browser to access the web application.<br />
Firstly, the application is designed to be run in a Python environment, so you will need to have Python installed on your machine.<br />
Additionally, you will need to install the required Python packages specified in the requirements.txt file. <br />
You can install these packages by running the command "pip install -r requirements.txt" in your terminal.<br />

Once you have set up the necessary environment and installed the required packages, you can use the application by running the main.py file.<br />
This will launch the Flask web application and open it in your web browser.<br />

The home page of the application provides options for uploading a CSV file containing complaint data or manually entering complaint data. <br />
If you choose to upload a CSV file, make sure the file is formatted correctly and follows the schema specified in the project documentation. <br />
<br />
The application will validate the file and display any errors if they exist. <br />
If the file is valid, the application will store the complaint data in a database and display a summary of the complaints.<br />

If you choose to manually enter complaint data, you will be prompted to fill out a form with the required fields. <br />
The application will validate the form and display any errors if they exist. <br />
If the form is valid, the application will store the complaint data in the database and display a summary of the complaints.
<br />
The complaint summary page displays a table of all the complaints that have been entered into the system,<br />
along with some summary statistics.<br />
<br />
You can click on the headers of the table to sort the data by different columns. <br />
Additionally, you can filter the table by department, status, or subject using the dropdown menus at the top of the page.

Finally, the application includes a classification feature that allows you to classify the complaints based on their details. <br />
To use this feature, you will need to first train a classifier using the train_classifier.py file. <br />
<br />
This file reads in a labeled CSV file of complaints and their categories, trains a machine learning model on the data, <br />
and saves the trained model to a file.<br /> Once you have trained the classifier, you can use the classify_complaint.py file to classify new complaints.<br />
This file reads in a complaint's details, loads the trained classifier, and predicts the category for the complaint.

## Credits
This project was developed by Ron Yalensky and Adi Dinner.


