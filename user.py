from message import Message
import uuid


class User:
    counter = 0

    def __init__(self, UserName):
        User.counter += 1
        self.UserId = uuid.uuid1()
        self.Name = UserName
        self.Messages = []

    def insertMessage(self, message):
        if(isinstance(message, Message)):
            self.Messages.append(message)

    def getUsername(self):
        return self.Name

    def getUserID(self):
        return self.UserId

    def compareUsername(self, toCompare):
        return toCompare == self.Name

    def getMessages(self):
        return self.Messages

    def printUser(self):
        print("Name: " + self.Name)
        print("UUID:", self.UserId, "\n")
        for message in self.Messages:
            message.printMessage()
