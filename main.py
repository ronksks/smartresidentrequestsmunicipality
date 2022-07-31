"""
Ron Yalensky
Adi Dinner

---------
Details
---------



TODO List:
-------------
DONE: 0) Create SQL database in Azure that will contains the next columns: S/N (Primary Key), (Municipality Class - 3 Tables)
TODO: 1) Add pre-processing and classify it for new complaint - main.py
DONE: 2) Add SQL query to insert the new complaint - sql_sql_communication.py
TODO: 3) Add handling form POST - new complaint - index.html + main.py
TODO: 4) Add handling from POST - compliant status - index.html + main.py
TODO: 5) Add in the index.html - Handling complaint status - frontend and backend - index.html
TODO: 6) Fix the method training_model - according to our book - classification.py
TODO: 7) Fix the method model_evaluation - according to our book - classification.py
TODO: 8) Fix the method confusion_matrix - according to our book - classification.py
TODO: 9) Adding a function that process the new complaint - classification.py
DONE: 10) Adding SQL queries as defined in the class - sql_sql_communication.py

TODO: 99) Frontend website - adding information / changing / improving / adding features to the website.

"""

from flask import Flask, render_template, request

from classification import Classification
import municipality
from sql_communication import SqlCommunication

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dodo'

# TODO: Before deploy to the web the boolean value training_process_only must be false.
classification_process = Classification(training_process_only=True)
sql_connection = SqlCommunication()
sql_connection.connect_server()

subject_list = sql_connection.get_department(from_local=False)


def process_new_complaint(new_complaint):
    """
    Processing the new complaint and insert into the database.
    :param new_complaint:
    :return:
    """

    # Getting the department after the classification process.
    department = classification_process.process_new_data()

    # Save the new complaint into the database.

    pass


def get_compliant_status(complaint_id):
    """
    Getting from the database the complaint status if the complaint exist.
    :param complaint_id:
    :return:
    """
    pass


@app.route('/', methods=['GET', 'POST'])
def index():
    with_preload = True
    if request.method == 'POST':
        request_form = request.form
        if 'new_complaint' in request_form:
            person = municipality.Person()
            complaint = municipality.Complaint()
            pass
        elif "complaint_status" in request_form:
            person = municipality.Person()
            complaint = municipality.Complaint()
            pass
        with_preload = False
        user = request.form['contact_name']
        print(user)
    else:
        pass
    return render_template('index.html', with_preload=with_preload, subject_list=subject_list)


if __name__ == '__main__':
    """
    IMPORTANT:
    -----
    Don't insert any code below except for the app.run().
    It will not run any other code except it.
    """
    app.run(debug=True)
