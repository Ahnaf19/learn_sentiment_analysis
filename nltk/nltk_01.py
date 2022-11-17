from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents
from nltk import word_tokenize

# following https://www.digitalocean.com/community/tutorials/how-to-work-with-language-data-in-python-3-using-the-natural-language-toolkit-nltk

# used nltk twitter data samples and POS tagger by using these two commands
# python - m nltk.downloader twitter_samples
# python -m nltk.downloader averaged_perceptron_tagger

print(twitter_samples.fileids())
# prints ['negative_tweets.json', 'positive_tweets.json', 'tweets.20150430-223406.json']

# using twitter_samples.strings() pass into any of the json file and returns list of tweets
# print(twitter_samples.strings('tweets.20150430-223406.json'))

tweets = twitter_samples.strings('positive_tweets.json')  # list of tweets
# print(type(tweets)) # list

# TOKENIZATION
# tokenization is the act of breaking up a sequence of strings into pieces such as words, keywords, phrases, symbols and other elements, which are called tokens.
# list where each element is a list of tokens.

# before using tokenizer in NLTK, use this additional resource, punkt. The punkt module is a pre-trained model that helps you tokenize words and sentences.
#  For instance, this model knows that a name may contain a period (like “S. Daityari”) and the presence of this period in a sentence does not necessarily end it.

# nltk.download('punkt') # import nltk first. also, needs to run once
tweets_tokens = twitter_samples.tokenized(
    'positive_tweets.json')  # list in a list
# so we have tokens of each tweets as a list in a list. we can tag the tokens with the appropriate Part of Speech (POS) tags.

# single sentence from json tokenizing
single_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
print(single_tweet_tokens)

# single sentence tokenizing
# sentece = "Hi there! My name is Khan and I am happy to be here!"
# single_tokens = word_tokenize(sentece)
# print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
# print(single_tokens)

# .tokenized() method returns special characters such as @ and _. these characters can be removed through regular expression RE.

# TAGGING SENCTENCES
# import nltk's post tagger at the beginning
# pos_tag_sents(list of tokenized_str) and pos_tag(tokenized_str)
tweets_tagged = pos_tag_sents(tweets_tokens)

# print(tweets[0]) # first tweet
# print(tweets_tokens[0]) # list of tokens of first tweet
# print(tweets_tagged[0]) # list of tuples (token, tagged pos) of first tweet

JJ_count = 0  # adjective base form, comparative: JJR and superlative: JJS
NN_count = 0  # noun, common, singular or mass and NNP: noun, proper, singular
# https://www.guru99.com/pos-tagging-chunking-nltk.html and https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html for nltk pos tags and their meanings
for tweet in tweets_tagged:
    for pair in tweet:
        tag = pair[1]
        if tag == 'JJ':
            JJ_count += 1
        elif tag == 'NN':
            NN_count += 1

print(f'JJ: {JJ_count}, NN: {NN_count}')
