import os
import logging
from message_builder import NegInterventionMessageBuilder
from intervention import NegativeSentimentIntervention
from db import LocalStorageDB
from flask import Flask
from slack_sdk.web import WebClient
from slack_bolt import App
#from slackeventsapi import SlackEventAdapter
from message import Message
from message_builder import MessageBuilder
from main import add_args_to_parser, SentimentAnalyzer

import argparse

def add_slack_args_to_parser(parser):
    parser.add_argument("--neg_intervention",
                        help="Whether to use the negative intervention if conditions are met",
                        action="store_true",
                        default=False,
                        )
    parser.add_argument("--neg_intervention_threshold",
                        help="Threshold activation for intervention",
                        type=float,
                        default=0.9,
                        )
    parser.add_argument("--neg_intervention_min_length",
                        help="Minimum number of messages for negative intervention",
                        type=int,
                        default=1,
                        )
    return parser

# Initialize a Flask app to host the events adapter
app = App(token="xoxb-2142005292307-2147964030372-7aS2R4vRwmJykYqRtfkHiECp", signing_secret="b05f018ae0e9fec0d039837c1c188a62")
#app = Flask(__name__)
#slack_events_adapter = SlackEventAdapter(os.environ["SLACK_SIGNING_SECRET"], "/slack/events", app)

# Initialize a Web API client
#slack_web_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

# For simplicity we'll store our app data in-memory with the following data structure.
# instantiated_channels = {"channel": sentiment_message}
instantiated_channels = {}
messages = []

parser = argparse.ArgumentParser()
parser = add_args_to_parser(parser)
parser = add_slack_args_to_parser(parser)
args = parser.parse_args()
SA = SentimentAnalyzer(args)
database = LocalStorageDB()
database.BuildLocal()

def start_bot(user_id: str, channel: str, client):
    # Create a new onboarding tutorial.
    sentiment_message = MessageBuilder(channel, app)

    # Get the onboarding message payload
    message = sentiment_message.get_message_payload()

    # Post the onboarding message in Slack
    response = client.chat_postMessage(**message)

    # Capture the timestamp of the message we've just posted so
    # we can use it to update the message after a user
    # has completed an onboarding task.
    sentiment_message.timestamp = response["ts"]
    interventions = []
    if args.neg_intervention:
        neg_intervention_message = NegInterventionMessageBuilder(channel)

        neg_intervention = NegativeSentimentIntervention(neg_intervention_message.get_message_payload, args.neg_intervention_threshold, args.neg_intervention_min_length)
        interventions.append((neg_intervention_message, neg_intervention))

    # Store the message sent in onboarding_tutorials_sent
    if channel not in instantiated_channels:
        instantiated_channels[channel] = (sentiment_message, interventions)



# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@app.event("message")
def message(event, client):
    """Display the sentiment of the conversation in a channel after receiving a message
    that contains "start".
    """

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")

    if text and text.lower() == "start":
        print("Starting bot...")
        return start_bot(user_id, channel_id, client)
    elif text and channel_id in instantiated_channels:
        try:
            database.insertUser(user_id)
        except:
            pass
        # Get the original sentiment message sent.
        sentiment_message, interventions = instantiated_channels[channel_id]
        if sentiment_message.userid == user_id:
            return

        # Mark the pin task as completed.
        sentiment_message.last_msg = text

        messages.append(text)
        SA.set_text(messages)
        print("RUNNING ANALYSIS...")
        topic_sentiments = SA.run_analysis()
        print("ANALYSIS COMPLETE")
        sentiment_message.set_topic_sentiments(topic_sentiments)
        # Get the new message payload
        message = sentiment_message.get_message_payload()

        # Post the updated message in Slack
        updated_message = client.chat_postMessage(**message)
        database.writeMessage(user_id, Message(text, sentiment_message.last_msg_sentiments))

        for intervention_msg, intervention in interventions:
            payload = intervention.trigger(sentiment_message)
            if payload is not None:
                client.chat_postMessage(**payload)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.start(port=3000)
    #app.run(port=3000)