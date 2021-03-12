import os
import re
import nltk
import pandas as pd

nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA


def lambda_handler(event, context):
    text = event['text']

    # Load tickers
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), './cleaned_tickers.csv'))
    tickers = df['ticker'].tolist()
    emojis = {
        'rocket': 1.0,
        'gem': 1.0,
    }

    # Split sentences
    text = re.sub(r'\sand\s|\sor\s', '. ', text)
    tokenized_str = sent_tokenize(text)

    # Apply a sentiment analyzer
    sia = SIA()
    sia.lexicon.update(emojis)
    result = dict()

    for sentence in tokenized_str:
        pol_score = sia.polarity_scores(sentence)
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
        words = tokenizer.tokenize(sentence)
        ticker = None

        for word in words:
            if word in tickers:
                ticker = word

        if not ticker:
            continue

        if ticker in result:
            result[ticker] = pol_score['compound'] if pol_score['compound'] > result[ticker] else result[ticker]
        else:
            result[ticker] = pol_score['compound']

    data = []
    for ticker, sentiment_score in result.items():
        data.append({
            'ticker': ticker,
            'sentiment_score': sentiment_score
        })

    return {
        'statusCode': 200,
        'body': data,
    }
