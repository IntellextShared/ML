#import libraries
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import pickle
from interpret import Visualizer
LABEL_MAP = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE"
    }

def add_args_to_parser(parser):
    parser.add_argument("-text", 
                        type=str,
                        help="Text to do sentiment classification on")
    parser.add_argument("--save_path", 
                        type=str,
                        help="Path to save visualization to",
                        default="viz.pkl")
    parser.add_argument("--topics",
                        nargs='+',
                        help="list of topics to classify",
                        type=str,
                        default=["indemnification", "loss", "liability"])
    return parser

if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser = add_args_to_parser(parser)
    
    args = parser.parse_args()

    #Initialize sentiment classifier
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    visualizer = Visualizer()

    #Initialize zero-shot topic classifier
    topic_classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")

    #Classify sentiment of text from command line and compute attributions
    score, label, atts = visualizer.interpret_sentence(model, args.text, tokenizer)
    print('Visualize attributions based on Integrated Gradients')
    visualizer.save_visualization(args.save_path)

    #Classify topic of text from command line
    topic_out = topic_classifier(args.text, args.topics)
    id = np.argmax(topic_out["scores"])
    topic = topic_out["labels"][id]
    #Output results
    print("Your sentence had a sentiment of {} with a score of {:.3f} and a topic of {}".format(label, score, topic))
