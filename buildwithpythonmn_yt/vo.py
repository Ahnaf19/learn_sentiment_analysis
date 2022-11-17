# https://github.com/attreyabhatt/Sentiment-Analysis

import string
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt


def remove_stop_words(tokenized_words, stop_words):
    final_words = list()
    for word in tokenized_words:
        if word not in stop_words:
            final_words.append(word)

    return final_words


with open('read.txt', encoding='utf-8') as f:
    text = f.read()


# LOWER & REMOVE PUNCTUATION

text = text.lower()
# print(string.punctuation) # shows all the punctuations in python which are: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# print(text)

# str1: specifies the list of characters that need to be replaced
# str2: specifies the list of characters with which the characters need to be replaced
# str3: specifies the list of characters that need to be deleted
cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
print(cleaned_text)  # removes punctuations from text


# TOKENIZATION

tokenized_words = cleaned_text.split()
print(tokenized_words)

stop_words = stopwords.words('english')
final_words = remove_stop_words(tokenized_words, stop_words)

print(final_words)

# NLP Emotion Algorithm

# 1) Check if the word in the final word list is also present in emotion.txt
#  - open the emotion file
#  - Loop through each line and clear it
#  - Extract the word and emotion using split

# 2) If word is present -> Add the emotion to emotion_list
# 3) Finally count each emotion in the emotion list

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(
            ",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)
w = Counter(emotion_list)
print(w)

# Plotting the emotions on the graph

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
