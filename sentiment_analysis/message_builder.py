class MessageBuilder:
    """Constructs the sentiment message and stores the state of what the current sentiment of the conversation is"""

    WELCOME_BLOCK = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                "Sentiment by topic: \n\n"
            ),
        },
    }
    DIVIDER_BLOCK = {"type": "divider"}

    def __init__(self, channel):
        self.channel = channel
        self.username = "sentiment_analysis_bot"
        self.icon_emoji = ":robot_face:"
        self.timestamp = ""
        self.reaction_task_completed = False
        self.pin_task_completed = False
        self.last_msg = "start"
        self.topic_sentiments = {"default": ("default", 1.0)}
        self.userid = 'U024BUC0WAY'

    def get_message_payload(self):
        return {
            "ts": self.timestamp,
            "channel": self.channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "blocks": [
                self.WELCOME_BLOCK,
                self.DIVIDER_BLOCK,
                *self._get_msg_block(self.topic_sentiments)
            ],
        }


    def _get_msg_block(self, topic_sentiments):
        msg = ''.join(["TOPIC: {} SENTIMENT: {}, SCORE: {} \n".format(topic, topic_out[0], topic_out[1]) for topic, topic_out in topic_sentiments.items()])
        return [{"type": "section", "text": {"type": "mrkdwn", "text": msg}}]
