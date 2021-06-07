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
    parser.add_argument("-text_file", 
                        type=str,
                        help="File with text to do sentiment classification on")
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

if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser = add_args_to_parser(parser)
    
    args = parser.parse_args()

    text = read_file(args.text_file)

    #Initialize sentiment classifier
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_classifier = InterpretableSentimentClassifier()

    #Initialize zero-shot topic classifier
    topic_classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")    

    #Classify topic of text from command line
    topic_out = topic_classifier(text, args.topics, multi_label=True)
    topics = topic_out[0]["labels"]

    topic_labels = [[topic_out[i]["labels"][j] for j in range(len(topics)) if topic_out[i]["scores"][j] > 0.7] for i in range(len(topic_out))]
    

    sorted_text = sort_by_topic(text, topic_labels, topics)

    with open("out.txt", "w") as file:
        # Process sentiment for each topic
        for topic, sentences in sorted_text.items():
            if args.interpret:
                #Classify sentiment of text from command line and compute attributions
                score, label, atts = sentiment_classifier.interpret_sentence(model, sentences, tokenizer)
                print('Visualize attributions based on Integrated Gradients')
                sentiment_classifier.save_visualization(args.save_path)
            else:
                score, label, _ = sentiment_classifier.classify_sentiment(model, sentences, tokenizer)

            #Output results
            file.write("{} SENTIMENT: {} SCORE {} TOPIC: {} \n".format(sentences, label, score, topic))

    #print("Your sentence had a sentiment of {} with a score of {:.3f} and a topic of {}".format(label, score, topic))
