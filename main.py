#import libraries
import transformers
import argparse

if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
                            "-text", 
                            type=str,
                            help="Text to do sentiment classification on")
    args = parser.parse_args()
    #Initialize sentiment classifier
    classifier = transformers.pipeline('sentiment-analysis')
    #Classify text from command line
    sentiment = classifier(args.text)[0]
    #Output results
    print("Your sentence had a sentiment of {} with a score of {:.3f}".format(sentiment["label"], sentiment["score"]))
