#import libraries
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
import torch.nn as nn
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization

import pickle
LABEL_MAP = {
        0: "NEGATIVE",
        1: "NEUTRAL",
        2: "POSITIVE"
    }
class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model
        self.device=model.device

    def forward(self, *input, **kwargs):
        out = self.model(*input, **kwargs)
        return out.logits

class InterpretableSentimentClassifier():
    """Uses Integrated Gradients to compute attributions for a sentiment classification"""
    def __init__(self):
        # accumalate couple samples in this array for visualization purposes
        self.vis_data_records_ig = []

    def classify_sentiment(self, model, sentence, tokenizer, min_len=32):
        """Uses model to predict the sentiment of a sentence, then uses Integrated Gradients to compute attributions
            INPUTS:
            model: sentiment classification model
            sentence (str): sentence to classify the sentiment of
            tokenizer: tokenizer associated with our model to convert the sentence into tokens
            min_len: the length to pad our sentence to

        """
        model = WrapperModel(model)
        PAD_IND = tokenizer.pad_token_id
        indexed = tokenizer([sentence],
                                padding="max_length",
                                truncation=True,
                                max_length=32,
                                return_tensors="pt")
        text = tokenizer.convert_ids_to_tokens(indexed['input_ids'][0])

        if len(text) < min_len:
            text += ['pad'] * (min_len - len(text))

        model.zero_grad()


        # predict
        preds = F.softmax(model(**indexed), dim=-1)
        pred_ind = torch.argmax(preds.squeeze()).item()
        pred = torch.max(preds)
        return pred, LABEL_MAP[pred_ind], pred_ind

    def interpret_sentence(self, model, sentence, tokenizer, min_len = 32):
        """Uses model to predict the sentiment of a sentence, then uses Integrated Gradients to compute attributions
            INPUTS:
            model: sentiment classification model
            sentence (str): sentence to classify the sentiment of
            tokenizer: tokenizer associated with our model to convert the sentence into tokens
            min_len: the length to pad our sentence to

        """
        model = WrapperModel(model)
        lig = LayerIntegratedGradients(model, model.model.roberta.embeddings)
        PAD_IND = tokenizer.pad_token_id
        token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
        indexed = tokenizer([sentence],
                                padding="max_length",
                                truncation=True,
                                max_length=32,
                                return_tensors="pt")
        text = tokenizer.convert_ids_to_tokens(indexed['input_ids'][0])

        if len(text) < min_len:
            text += ['pad'] * (min_len - len(text))

        model.zero_grad()

        input_indices = indexed['input_ids']
        
        # input_indices dim: [sequence_length]
        seq_length = min_len

        # predict
        preds = F.softmax(model(**indexed), dim=-1)
        pred_ind = torch.argmax(preds.squeeze()).item()
        pred = torch.max(preds)
        label = pred_ind
        # generate reference indices for each sample
        reference_indices = token_reference.generate_reference(seq_length, device=model.device).unsqueeze(0)

        # compute attributions and approximation delta using layer integrated gradients
        attributions_ig, delta = lig.attribute(input_indices, reference_indices, \
                                            n_steps=50, return_convergence_delta=True, target=pred_ind)

        #print('pred: ', pred.item(), ', delta: ', abs(delta))

        self.add_attributions_to_visualizer(attributions_ig, text, pred, pred_ind, label, delta, self.vis_data_records_ig)
        return pred, LABEL_MAP[pred_ind], attributions_ig
        
    def add_attributions_to_visualizer(self, attributions, text, pred, pred_ind, label, delta, vis_data_records):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()
        
        # storing couple samples in an array for visualization purposes
        self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
                                attributions,
                                pred.detach().cpu().numpy(),
                                LABEL_MAP[pred_ind],
                                LABEL_MAP[label],
                                LABEL_MAP[label],
                                attributions.sum(),       
                                text,
                                delta))
    
    def save_visualization(self, path):
        html = visualization.visualize_text(self.vis_data_records_ig)
        with open(path, "wb") as file:
            pickle.dump(html, file)
