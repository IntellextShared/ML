import transformers

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-text", 
                            type=str,
                            help="Text to do sentiment classification on")
    args = parser.parse_args()
    classifier = transformers.pipeline('sentiment-analysis')
    sentiment = classifier(args.text)[0]
    print("Your sentence had a sentiment of {} with a score of {:.3f}".format(sentiment["label"], sentiment["score"]))