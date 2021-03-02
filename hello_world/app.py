import re
import nltk
import en_core_web_sm
import pandas as pd
from emoji import get_emoji_regexp

nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def lambda_handler(event, context):
    text = event['text']

    # Load tickers
    df = pd.read_csv('./cleaned_tickers.csv')
    tickers = df['ticker'].tolist()

    # Remove emojis if exists
    text = get_emoji_regexp().sub(u'', text)
    text = re.sub(r'and|or', '.', text)
    tokenized_str = sent_tokenize(text)

    # Remove stop words
    nlp = en_core_web_sm.load()
    all_stopwords = nlp.Defaults.stop_words
    tokens_without_sw = [word for word in tokenized_str if word not in all_stopwords]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = ([lemmatizer.lemmatize(word) for word in tokens_without_sw])
    cleaned_output = lemmatized_tokens

    # Apply a sentiment analyzer
    sia = SIA()
    result = dict()

    for sentence in cleaned_output:
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
        "statusCode": 200,
        "body": data,
    }
