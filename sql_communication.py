"""

"""
import string
import municipality
import pyodbc

__version__ = "0.1.1"


class SqlQueries:

    def get_complaint_status(self, complaint_id):
        query = 'SELECT * FROM dbo.complaints WHERE SerialNumber=' + complaint_id
        return query

    def get_department_list(self):
        query = 'SELECT Name FROM dbo.department'
        return query

    def send_new_complaint(self, person: municipality.Person, complaint: municipality.Complaint):
        queries = ['INSERT INTO dbo.persons ("Id", "FirstName", "LastName", "FullName", "Phone", "Email") '
                   'VALUES (?,?,?,?,?,?,?)',
                   person.Id, person.First_Name, person.Last_Name, person.Full_Name, person.Phone, person.Email,
                   'INSERT INTO dbo.complaints ("Department", "Subject", "Status", "Details", "Person") '
                   'VALUES (?,?,?,?,?)',
                   complaint.Department, complaint.Subject, complaint.Status, complaint.Details, complaint.Person]
        return queries


class SqlCommunication(SqlQueries):

    def __init__(self, server_name="leadsbezeq.database.windows.net", user="ilcattivo", password="zaq!2wsx",
                 database="leadsBezeq"):
        super().__init__()
        self.server = server_name
        self.username = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect_server(self):
        """

        :return:
        """
        try:
            self.connection = pyodbc.connect(
                'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + self.server + ';DATABASE=' + self.database + ';UID=' + self.username + ';PWD=' + self.password)
            self.cursor = self.connection.cursor()

        except Exception as e:
            print(e)
            return False
        return True

    def check_complaint_status(self, complaint_id):
        """

        :param complaint_id:
        :return:
        """
        self.cursor.execute(self.get_complaint_status(complaint_id))
        return self.cursor.fetchone().toarray()

    def get_department(self, from_local=True):
        """

        :param from_local:
        :return:
        """
        department_list = []
        try:
            if from_local:
                file = open('static/classification/department_list.txt', 'r', encoding='utf8')
                department_list = file.readlines()
                department_list = list(map(str.strip, department_list))
                department_list = [str(dep).translate(str.maketrans('', '', string.punctuation)) for dep in
                                   department_list]
            else:
                self.cursor.execute(self.get_department_list())
                print(self.get_department_list())
                rows = self.cursor.fetchall()
                department_list = [row.Name for row in rows]
        except Exception as e:
            print(e)
        return department_list

    def insert_departments(self, department_list):
        """

        :param department_list:
        :return:
        """
        try:
            for dep in department_list:
                self.cursor.execute("INSERT INTO dbo.department(Name) VALUES (?)", dep)
                self.cursor.commit()
        except Exception as e:
            print(e)
        pass
