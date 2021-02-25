import json
import re
import spacy
import en_core_web_sm
from emoji import get_emoji_regexp
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


def lambda_handler(event, context):
    text = event['text']

    # Remove emojis if exists
    text = get_emoji_regexp().sub(u'', text)

    # Break apart every word in the string into an individual word
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|http\S+')
    tokenized_str = tokenizer.tokenize(text)

    # Convert tokens into lowercase
    lower_str_tokenized = [word.lower() for word in tokenized_str]

    # Remove stop words
    nlp = en_core_web_sm.load()
    all_stopwords = nlp.Defaults.stop_words
    tokens_without_sw = [word for word in lower_str_tokenized if word not in all_stopwords]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = ([lemmatizer.lemmatize(word) for word in tokens_without_sw])
    stemmer = PorterStemmer()
    stem_tokens = ([stemmer.stem(word) for word in tokens_without_sw])
    cleaned_output = lemmatized_tokens

    # Apply a sentiment analyzer
    sia = SIA()
    results = []

    for sentences in cleaned_output:
        pol_score = sia.polarity_scores(sentences)
        pol_score['words'] = sentences
        results.append(pol_score)

    print(results)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "hello world",
        }),
    }
