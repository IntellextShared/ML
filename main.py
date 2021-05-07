#import libraries
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import pickle
from interpret import Visualizer
LABEL_MAP = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE"
    }


if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", 
                        type=str,
                        help="Text to do sentiment classification on")
    parser.add_argument("--save_path", 
                        type=str,
                        help="Path to save visualization to",
                        default="viz.pkl")
    args = parser.parse_args()

    #Initialize sentiment classifier
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    visualizer = Visualizer()

    #Classify text from command line and compute attributions
    score, label, atts = visualizer.interpret_sentence(model, args.text, tokenizer)
    print('Visualize attributions based on Integrated Gradients')
    visualizer.save_visualization(args.save_path)
    
    #Output results
    print("Your sentence had a sentiment of {} with a score of {:.3f}".format(label, score))
