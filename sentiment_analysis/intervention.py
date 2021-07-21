class Intervention:
    def __init__(self, msg_constructor):
        self.msg_constructor = msg_constructor

    def trigger(self, sentiment_message):
        sentiments = sentiment_message.topic_sentiments
        return None

class NegativeSentimentIntervention(Intervention):
    """
    An Intervention that is triggered when the sentiment for any topic becomes overly negative.
    """

    def __init__(self, msg_constructor, threshold, min_length):
        super(NegativeSentimentIntervention, self).__init__(msg_constructor)
        self.threshold = threshold
        self.min_length = min_length

    def trigger(self, sentiment_message):
        sentiments = sentiment_message.topic_sentiments
        negative_topics = {}
        for topic, sentiment_out in sentiments.items():
            label, score, sentences = sentiment_out
            if label == "NEGATIVE" and score > self.threshold and len(sentences) > self.min_length:
                negative_topics[topic] = (score, sentences)
        if len(negative_topics) > 0:
            return self.msg_constructor(negative_topics)
        else:
            return None