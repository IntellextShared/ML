import os
import uuid
from user import User
from message import Message

class LocalStorageDB:

    def __init__(self):
        self.NumUsers = 0
        self.users = []
        self.userFiles = {}
        self.BuildLocal()

    def BuildLocal(self):
        # build local file structure
        try:
            os.mkdir(os.path.join(os.getcwd(), "LocalStorageDB"))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(os.getcwd(), "LocalStorageDB/Users"))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(os.getcwd(), "LocalStorageDB/Messages"))
        except FileExistsError:
            pass

    def insertUser(self, NewUser):
        try:
            self.userFiles[NewUser]
            return print("Err user already exists")
        except KeyError:
            self.users.append(User(NewUser))
            self.userFiles[NewUser] = [self.initUserFile(self.users[len(self.users) - 1]), NewUser, 0]

    # Returns location of user's directory
    def initUserFile(self, user):
        userDirectory = user.getUsername() + str(user.getUserID())[4:8]
        try:
            os.mkdir(os.path.join(os.getcwd(), "LocalStorageDB/Users/" + userDirectory))
        except FileExistsError:
            return print("User already exists and already has a directory initialized")

        UserFileInit = open("./LocalStorageDB/Users/" + userDirectory + "/" + user.getUsername() + ".txt", "w+") 
        UserFileInit.write(user.getUsername() + "\n" + str(user.getUserID()) + "\n")

        try:
            os.mkdir(os.path.join(os.getcwd(), "LocalStorageDB/Users/" + userDirectory + "/messages"))
        except FileExistsError:
            return print("Error attempting to create directory for message storage")
        
        return os.getcwd() + "\LocalStorageDB\\Users\\" + userDirectory

    def writeMessage(self, user, message):
        try:
            userValue = self.userFiles[user]
            fileLocation = userValue[0] + "\messages\\" + userValue[1] + str(userValue[2]) + ".txt"
            self.userFiles[user][2] += 1
        except KeyError:
            return print("User does not exist in database")
        
        messageFile = open(fileLocation, "w+")
        messageFile.write(message.writeMessage())
        messageFile.close()
        messageFile = open("./LocalStorageDB/Messages/" + userValue[1] + str(userValue[2]) + ".txt", "w+")
        messageFile.write(message.writeMessage())
        messageFile.close()

    def printUsers(self):
        for user in self.users:
            user.printUser()

myDB = LocalStorageDB()
myDB.insertUser("a")
myDB.insertUser("b")
myDB.insertUser("a")

myDB.printUsers()

myDB.writeMessage("a", Message("Hey there", "TOPIC: item SENTIMENT: NEUTRAL, SCORE: 0.8699129223823547\nTOPIC: quantity SENTIMENT: POSITIVE, SCORE: 0.6045356392860413\nTOPIC: delivery date SENTIMENT: POSITIVE, SCORE: 0.6045356392860413\nTOPIC: price SENTIMENT: NEUTRAL, SCORE: 0.4512723684310913"))