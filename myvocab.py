import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    text = text.replace('\\','')
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

folderName = 'split_1'
train_filename = folderName + '/' + 'train.tsv'

alldata = pd.read_csv(train_filename,sep='\t', header=0)
review_corpus = alldata['review'].apply(remove_html_tags)

# print(review_corpus.head(5))
review_corpus = review_corpus.to_numpy()
# print(review_corpus[:5])
# print(f'length:{review_corpus.shape}')
# exit()

class Vocabulary:
    def __init__(self, name):
        
        self.name = name
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        self.stop_words = list(STOP_WORDS)
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):

        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in word_tokenize(sentence):
            word = word.lower()

            sentence_len += 1
            if word not in self.stop_words:
                self.add_word(word)
            else:
                continue
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    def fine_tune(self):
        for word in self.word2count.keys():
            if self.word2count[word] <= 5:
                break

        


# voc = Vocabulary('test')
# corpus = ['This is the first sentence.',
#           'This is the second.',
#           'There is no sentence in this corpus longer than this one.',
#           'My dog is named Patrick.',
#           'This is the first sentence. This is the second. There is no sentence in this corpus longer than this one. My dog is named Patrick.']
# for test in corpus:
#     new_test = word_tokenize(test)
#     print(f'test:{new_test}')

# for sent in corpus:
#     voc.add_sentence(sent)

# for word in range(voc.num_words):
#     print(voc.to_word(word))
voc = Vocabulary('movie')
alldata["review"].replace( { r'[^a-zA-Z0-9 ]' : '' }, inplace= True, regex = True)
review_corpus = alldata["review"]

tokenized_sents = [word_tokenize(i) for i in review_corpus]
print(len(tokenized_sents))

flattened = []
for sublist in tokenized_sents:
    for val in sublist:
        flattened.append(val)
print(len(flattened))

Vocab=[]
for item in flattened:
    if item not in Vocab and item not in STOP_WORDS:
        Vocab.append(item)

Vocab = np.array(Vocab)
print(len(Vocab))
filename = 'myvocab.txt'
np.savetxt(filename,Vocab)
print('My vocab is saved')



# test_corpus = review_corpus[:2]
# for doc in test_corpus:
#     print(f'doc:{doc}')
#     print('doc:',doc.split('.'))
#     # for sent in doc:
#     #     print(f'sent:{sent}')
# print(f'test_corpus:{test_corpus}')

# for sent in review_corpus:
#     voc.add_sentence(sent)
# length_voc = voc.num_words
# print(f'length_voc:{length_voc}')
# for word in range(voc.num_words):
#     print(voc.to_word(word))

