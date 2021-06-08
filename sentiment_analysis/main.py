#import libraries
from os import read
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import pickle
from interpret import InterpretableSentimentClassifier
LABEL_MAP = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE"
    }

def add_args_to_parser(parser):
    
    parser.add_argument("--text_file", 
                        type=str,
                        help="File with text to do sentiment classification on",
                        default=None)
    parser.add_argument("--text",
                        type=list,
                        help="List of strings to do topic and sentiment analysis on (to be set in app.py, not directly from terminal)",
                        default=None)
    parser.add_argument("--save_path", 
                        type=str,
                        help="Path to save visualization to",
                        default="viz.pkl")
    parser.add_argument("--topics",
                        nargs='+',
                        help="list of topics to classify",
                        type=str,
                        default=["item", "quantity", "price", "delivery date"])
    parser.add_argument("--interpret",
                        help="Whether to compute the integrated gradients attributions for the sentiment classification",
                        action="store_true",
                        default=False,
                        )
    return parser

def read_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    return lines

def sort_by_topic(text, topic_labels, topics):
    assert len(text) == len(topic_labels)
    num_sentences = len(text)
    num_topics = len(topics)
    sorted_text = {topic: "" for topic in topics}
    for i in range(num_sentences):
        for label in topic_labels[i]:
            sorted_text[label] += text[i] + " "
    return sorted_text

class SentimentAnalyzer():
    def __init__(self, args):
        self.args = args
        #Initialize sentiment classifier
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.sentiment_classifier = InterpretableSentimentClassifier()

        #Initialize zero-shot topic classifier
        self.topic_classifier = pipeline("zero-shot-classification",
                            model="valhalla/distilbart-mnli-12-9")    

    def set_text(self, text):
        self.args.text = text

    def run_analysis(self):
        assert self.args.text_file or self.args.text
        if self.args.text_file:
            text = read_file(self.args.text_file)
        elif self.args.text:
            text = self.args.text

        #Classify topic of text from command line
        topic_out = self.topic_classifier(text, self.args.topics, multi_label=True)
        topics = topic_out[0]["labels"]

        topic_labels = [[topic_out[i]["labels"][j] for j in range(len(topics)) if topic_out[i]["scores"][j] > 0.7] for i in range(len(topic_out))]
        

        sorted_text = sort_by_topic(text, topic_labels, topics)
        topic_sentiments = {}
        with open("out.txt", "w") as file:
            # Process sentiment for each topic
            for topic, sentences in sorted_text.items():
                if self.args.interpret:
                    #Classify sentiment of text from command line and compute attributions
                    score, label, atts = self.sentiment_classifier.interpret_sentence(self.model, sentences, self.tokenizer)
                    print('Visualize attributions based on Integrated Gradients')
                    self.sentiment_classifier.save_visualization(self.args.save_path)
                else:
                    score, label, _ = self.sentiment_classifier.classify_sentiment(self.model, sentences, self.tokenizer)
                topic_sentiments[topic] = (label, score, sentences)

                #Output results
                file.write("{} SENTIMENT: {} SCORE {} TOPIC: {} \n".format(sentences, label, score, topic))
        return topic_sentiments

    #print("Your sentence had a sentiment of {} with a score of {:.3f} and a topic of {}".format(label, score, topic))

if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser = add_args_to_parser(parser)
    
    args = parser.parse_args()
    SA = SentimentAnalyzer(args)
    SA.run_analysis()

    
