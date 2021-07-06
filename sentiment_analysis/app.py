import os
import logging
from flask import Flask
from slack_sdk.web import WebClient
from slackeventsapi import SlackEventAdapter
from message_builder import MessageBuilder
from main import add_args_to_parser, SentimentAnalyzer
import argparse
# Initialize a Flask app to host the events adapter
app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(os.environ["SLACK_SIGNING_SECRET"], "/slack/events", app)

# Initialize a Web API client
slack_web_client = WebClient(token=os.environ['SLACK_BOT_TOKEN'])

# For simplicity we'll store our app data in-memory with the following data structure.
# onboarding_tutorials_sent = {"channel": {"user_id": OnboardingTutorial}}
instantiated_channels = {}
messages = []

parser = argparse.ArgumentParser()
parser = add_args_to_parser(parser)
analysis_args = parser.parse_args()
SA = SentimentAnalyzer(analysis_args)

def start_bot(user_id: str, channel: str):
    # Create a new onboarding tutorial.
    sentiment_message = MessageBuilder(channel)

    # Get the onboarding message payload
    message = sentiment_message.get_message_payload()

    # Post the onboarding message in Slack
    response = slack_web_client.chat_postMessage(**message)

    # Capture the timestamp of the message we've just posted so
    # we can use it to update the message after a user
    # has completed an onboarding task.
    sentiment_message.timestamp = response["ts"]

    # Store the message sent in onboarding_tutorials_sent
    if channel not in instantiated_channels:
        instantiated_channels[channel] = sentiment_message



# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@slack_events_adapter.on("message")
def message(payload):
    """Display the sentiment of the conversation in a channel after receiving a message
    that contains "start".
    """
    event = payload.get("event", {})

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")


    if text and text.lower() == "start":
        print("Starting bot...")
        return start_bot(user_id, channel_id)
    elif text and channel_id in instantiated_channels:
        # Get the original sentiment message sent.
        sentiment_message = instantiated_channels[channel_id]
        if sentiment_message.userid == user_id:
            return

        # Mark the pin task as completed.
        sentiment_message.last_msg = text

        messages.append(text)
        SA.set_text(messages)
        print("RUNNING ANALYSIS...")
        topic_sentiments = SA.run_analysis()
        print("ANALYSIS COMPLETE")
        sentiment_message.topic_sentiments = topic_sentiments
        # Get the new message payload
        message = sentiment_message.get_message_payload()

        # Post the updated message in Slack
        updated_message = slack_web_client.chat_postMessage(**message)


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.run(port=3000)