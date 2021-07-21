import matplotlib.pyplot as plt
import numpy as np
class BaseMessageBuilder:
    DIVIDER_BLOCK = {"type": "divider"}
    STRUCTURE_BLOCK = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
        },
    }

    def __init__(self, channel):
        self.channel = channel
        self.username = "sentiment_analysis_bot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.last_msg = "start"
        self.userid = 'U024BUC0WAY'

class MessageBuilder(BaseMessageBuilder):
    """Constructs the sentiment message and stores the state of what the current sentiment of the conversation is"""

    STRUCTURE_BLOCK = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                "Sentiment by topic: \n\n"
            ),
        },
    }

    def __init__(self, channel, app):
        super(MessageBuilder, self).__init__(channel)

        self.last_msg = "start"
        self.last_msg_sentiments = None
        self.topic_sentiments = {"default": ("default", 1.0)}
        self.all_topic_sentiments = []
        self.app = app
    
    def set_topic_sentiments(self, topic_sentiments):
        self.topic_sentiments = topic_sentiments
        self.all_topic_sentiments.append(topic_sentiments)

    def get_message_payload(self):
        sentiment_block = self._get_msg_block(self.topic_sentiments)
        plot_block = self._get_img_block()
        self.last_msg_sentiments = sentiment_block[0]["text"]["text"]
        payload = {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                self.STRUCTURE_BLOCK,
                self.DIVIDER_BLOCK,
                *sentiment_block,
                self.DIVIDER_BLOCK,
                *plot_block
            ],
        }
                
        return payload

    def _get_img_block(self):
        template = lambda image_url: {
                    "type": "image",
                    "title": {
                        "type": "plain_text",
                        "text": "Please enjoy this photo of a kitten"
                    },
                    "image_url": image_url,
                    "alt_text": "An incredibly cute kitten."
                    }
        graphs = {}
        fig, axs = plt.subplots(2, 2)
        for topic_sentiments in self.all_topic_sentiments:
            for topic, sent_out in topic_sentiments.items():
                sentiment, score, sentences = sent_out
                if sentiment == "POSITIVE":
                    sign = 1
                elif sentiment == "NEUTRAL":
                    sign = 0
                elif sentiment == "NEGATIVE":
                    sign = -1
                if topic not in graphs:
                    graphs[topic] = []
                else:
                    graphs[topic].append(sign * score)
        block = []
        if len(graphs) > 0:
            for i, (topic, sentiments) in enumerate(graphs.items()):
                sentiments = [sentiment.detach().cpu().numpy() for sentiment in sentiments]
                ax = axs[i//2, i%2]
                ax.plot(np.arange(len(sentiments)), sentiments)
                ax.set_title(topic)
            plt.savefig("sent_graph{}.png".format(i))
            response = self.app.client.files_upload(file="sent_graph{}.png".format(i), channels='general')
            
            #link = response["file"]["url_private_download"]
            #block.append(template(link))
        return block


    def _get_msg_block(self, topic_sentiments):
        msg = ''.join(["TOPIC: {} SENTIMENT: {}, SCORE: {} \n".format(topic, topic_out[0], topic_out[1]) for topic, topic_out in topic_sentiments.items()])
        return [{"type": "section", "text": {"type": "mrkdwn", "text": msg}}]

class NegInterventionMessageBuilder(BaseMessageBuilder):

    STRUCTURE_BLOCK = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                "Hi, we noticed that this conversation seems to be overheating a little regarding the following topics: \n\n"
            ),
        },
    }
    def __init__(self, channel):
        super().__init__(channel)

    def get_message_payload(self, negative_topics):
        sentiment_block = self._get_msg_block(negative_topics)
        payload = {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                self.STRUCTURE_BLOCK,
                self.DIVIDER_BLOCK,
                *sentiment_block
            ],
        }
        
        return payload

    def _get_msg_block(self, negative_topics):
        msg = "".join(["TOPIC: {}, SCORE: {} \n".format(topic, topic_out[0]) for topic, topic_out in negative_topics.items()])
        return [{"type": "section", "text": {"type": "mrkdwn", "text": msg}}]
