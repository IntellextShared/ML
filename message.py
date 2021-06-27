class SentimentTopic:
    def __init__(self, topic, sentiment, score):
        self.Topic = topic
        self.Sentiment = sentiment
        self.Score = score

    def printTopic(self):
        print(self.Topic + "\n" + self.Sentiment + "\n" + self.Score + "\n")

    def getTopic(self):
        return self.Topic + "\n" + self.Sentiment + "\n" + self.Score + "\n"


class Message:

    # content -> content of original message
    # sentiment is message containing sentiment by topic made by SentimentAnalysisMLBot
    def __init__(self, content, sentiment):
        self.Content = content
        self.Topics = []

        try:
            topics = sentiment.split("\n")
        except:
            print("Sentiment passed in incorrect format")
            return

        for topic in topics:
            topicEnd = topic.find("SENTIMENT")
            TOPIC = topic[0:topicEnd]
            sentimentEnd = topic.find("SCORE")
            SENTIMENT = topic[topicEnd:sentimentEnd - 2]
            SCORE = topic[sentimentEnd:]

            self.Topics.append(SentimentTopic(TOPIC, SENTIMENT, SCORE))

    def printMessage(self):
        for topic in self.Topics:
            topic.printTopic()

    def writeMessage(self):
        fullMessage = self.Content + "\n"
        for topic in self.Topics:
            fullMessage += "\n" + topic.getTopic()
        return fullMessage
