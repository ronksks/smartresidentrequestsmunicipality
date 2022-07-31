__version__ = '0.21.3'


class Person:
    def __init__(self):
        self.SerialNumber = None
        self.Id = None
        self.First_Name = None
        self.Last_Name = None
        self.Phone = None
        self.Email = None
        self.Full_Name = None

    def person(self, serial_number, first_name, last_name=None, full_name=None, phone=None, email=None, person_id=None):
        self.SerialNumber = serial_number
        self.Id = person_id
        self.First_Name = first_name
        self.Last_Name = last_name
        self.Full_Name = full_name
        self.Phone = phone
        self.Email = email


class Department:
    def __init__(self):
        self.SerialNumber = None
        self.Name = None
        pass

    def department(self, serial_number, name):
        self.SerialNumber = serial_number
        self.Name = name


class Complaint:
    def __init__(self):
        self.SerialNumber = None
        self.Department = None
        self.Subject = None
        self.Status = None
        self.Details = None
        self.Person = None

    def complaint(self, serial_number, department: Department, subject, status, details, person: Person):
        self.SerialNumber = serial_number
        self.Department = department
        self.Subject = subject
        self.Status = status
        self.Details = details
        self.Person = person
