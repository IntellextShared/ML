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
onboarding_tutorials_sent = {}
messages = []

parser = argparse.ArgumentParser()
parser = add_args_to_parser(parser)
analysis_args = parser.parse_args()
SA = SentimentAnalyzer(analysis_args)

def start_onboarding(user_id: str, channel: str):
    # Create a new onboarding tutorial.
    onboarding_tutorial = MessageBuilder(channel)

    # Get the onboarding message payload
    message = onboarding_tutorial.get_message_payload()

    # Post the onboarding message in Slack
    response = slack_web_client.chat_postMessage(**message)

    # Capture the timestamp of the message we've just posted so
    # we can use it to update the message after a user
    # has completed an onboarding task.
    onboarding_tutorial.timestamp = response["ts"]

    # Store the message sent in onboarding_tutorials_sent
    if channel not in onboarding_tutorials_sent:
        onboarding_tutorials_sent[channel] = onboarding_tutorial



# ============== Message Events ============= #
# When a user sends a DM, the event type will be 'message'.
# Here we'll link the message callback to the 'message' event.
@slack_events_adapter.on("message")
def message(payload):
    """Display the onboarding welcome message after receiving a message
    that contains "start".
    """
    event = payload.get("event", {})

    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")


    if text and text.lower() == "start":
        print("Starting onboarding...")
        return start_onboarding(user_id, channel_id)
    elif text and channel_id in onboarding_tutorials_sent:
        # Get the original tutorial sent.
        onboarding_tutorial = onboarding_tutorials_sent[channel_id]
        if onboarding_tutorial.userid == user_id:
            return

        # Mark the pin task as completed.
        onboarding_tutorial.last_msg = text

        messages.append(text)
        SA.set_text(messages)
        print("RUNNING ANALYSIS...")
        topic_sentiments = SA.run_analysis()
        print("ANALYSIS COMPLETE")
        onboarding_tutorial.topic_sentiments = topic_sentiments
        # Get the new message payload
        message = onboarding_tutorial.get_message_payload()

        # Post the updated message in Slack
        updated_message = slack_web_client.chat_postMessage(**message)



if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    app.run(port=3000)