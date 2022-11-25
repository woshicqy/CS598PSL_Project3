import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


stop_wrods = ['around', 'only', 'were', 'nevertheless', 'she', 
              'even', 'must', 'ourselves', 'whereupon', 'done', 
              'whether', 'below', 'empty', 'many', 'without', 
              'nor', 'top', 'anything', 'further', 'somehow', 
              '’ll', 'ten', 'might', 'move', 'thereafter', 
              'yet', 'made', 'former', 'much', 'few', 
              'my', 'for', 'herself', 'afterwards', 'until', 
              '’d', 'any', 'who', 'in', 'please', 'become', 
              'another', 'else', 'had', 'cannot', 'off', 
              'seemed', 'every', 'again', 'myself', 'as', 
              'namely', 'could', 'your', 'himself', 'what', 
              'these', 'whenever', "'m", 'would', 'together', 
              'most', 'perhaps', 'behind', 'it', 'keep', 
              'fifty', 'am', 'into', 'always', 'same', 
              'each', 'make', 'that', 'ca', 'one', 
              '’re', 'never', 'more', 'us', 'four', 
              'anyhow', 'during', 'beside', 'whereafter', 
              'sixty', 'hence', 'by', 'seeming', 'besides', 
              'yourself', 'whereby', '’ve', '‘re', 'are', 
              'just', "'d", 'beyond', '‘d', 'whence', 
              'either', 'nobody', 'then', 'though', 'otherwise', 
              'them', 'there', 'five', 'they', 'name', 
              'per', 'yourselves', 'n‘t', 'because', 'whose', 
              'thence', 'hereupon', 'something', 'hereby', 
              'anyway', 'should', 'back', 'eight', 'and', 
              'here', 'can', 'among', 'amongst', 'next', 
              'nowhere', 'nothing', 'me', 'neither', 'indeed', 
              'six', 'seems', 'various', 'latter', 'whom', 
              'you', 'becomes', 'after', 'have', 'thus', 
              'everything', '‘m', 'less', 'someone', 'about', 
              'hundred', 'therefore', 'to', 'down', 'several', 
              'through', 'put', 'ever', 'where', 'when', 
              'elsewhere', 'get', 'why', 'but', 'those', 
              'whole', 'regarding', 'yours', 'part', 'herein', 
              'too', 'their', 'our', 'if', 'will', 'towards', 
              "'re", 'moreover', 'somewhere', 'amount', 'mine', 
              'still', 'full', 'other', "'s", 'such', 'does', 
              'nine', 'unless', 'wherever', 'wherein', 'sometimes', 
              'between', "'ve", 'alone', 'him', 'of', 'whoever', 
              'none', 'twenty', 'front', 'least', 'across', 'first', 
              'last', 'almost', 'be', 'often', 'serious', 'seem', 'others', 
              'forty', 'at', 'three', 'we', 'which', 'against', 'therein', 
              'doing', 'his', 'thereupon', 'beforehand', 'now', 'from', 
              '’m', 'two', 'however', 'call', 'becoming', 'a', 'well', 
              'everywhere', 'really', 'with', 'used', 'or', 'although', 
              'show', 'its', 'via', 'no', 'out', 'whatever', 'before', 
              'up', 'so', 'this', 'whither', 'formerly', 'twelve', 'since', 
              'while', 'eleven', 'anyone', 'do', '‘ve', 'toward', 'latterly', 
              'upon', 'her', 'how', 'above', 'the', 'some', 'noone', '’s', 
              'thru', 'hereafter', 'was', 'mostly', 'throughout', 'n’t', 
              'has', 'onto', 'been', "n't", 'see', 'over', 'along', 're', 
              '‘s', 'quite', 'give', 'own', 'on', 'anywhere', 'rather', "'s",
              'within', 'fifteen', 'due', 'not', 'everyone', 'take', 'whereas', 
              'than', 'bottom', 'under', 'ours', 'already', 'also', 'itself', 
              "'ll", 'meanwhile', '‘ll',"\'ll", 'hers', 'third', 'all', 'he', 'being', 
              'sometime', 'say', 'using', 'did', 'once', 'may', 'is', 'an', 'Mr',
              'thereby', 'enough', 'very', 'i', 'except', 'side', 'themselves', 'both', 'go', 'became']

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    text = text.replace('\\','')
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

folderName = 'split_1'
train_filename = folderName + '/' + 'train.tsv'

alldata = pd.read_csv(train_filename,sep='\t', header=0)
# review_corpus = alldata['review'].apply(remove_html_tags)

alldata["review"].replace( { r'[^a-zA-Z ]' : '' }, inplace= True, regex = True)
review_corpus = alldata['review']
# print(review_corpus.head(5))
review_corpus = review_corpus.to_numpy()
# print(review_corpus[:5])
# print(f'length:{review_corpus.shape}')
# exit()

class Vocabulary:
    def __init__(self, name,stopWords):
        
        self.name = name
        PAD_token = 0   # Used for padding short sentences
        SOS_token = 1   # Start-of-sentence token
        EOS_token = 2   # End-of-sentence token
        self.stop_words = stopWords
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
        ps = PorterStemmer()
        wl = WordNetLemmatizer()
        for word in word_tokenize(sentence):
            word = word.lower()
            word = wl.lemmatize(word)
            word = ps.stem(word)

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
    
    def getKeys(self,freq):
        delete = [key for key in self.word2count if self.word2count[key] <= freq]
        return delete


        return self.word2count.keys()

    def fine_tune(self,freq):
        keys = self.getKeys(freq)
        for word in keys:
            index = self.word2index[word]
            del self.word2count[word]
            del self.word2index[word]
            del self.index2word[index]
            self.num_words -= 1


voc = Vocabulary('movie',stop_wrods)
for sent in review_corpus:
    voc.add_sentence(sent)

print('Size (Before):',voc.num_words)
voc.fine_tune(20)
print('Size (After):',voc.num_words)
vocab = []
for word in (voc.word2count.keys()):
    vocab.append(word)

filename = 'myvocab.txt'
np.savetxt(filename,vocab,fmt='%s')
print('My vocab is saved')

# alldata["review"].replace( { r'[^a-zA-Z0-9 ]' : '' }, inplace= True, regex = True)
# review_corpus = alldata["review"]

# tokenized_sents = [word_tokenize(i) for i in review_corpus]
# print(len(tokenized_sents))

# flattened = []
# for sublist in tokenized_sents:
#     for val in sublist:
#         flattened.append(val)
# print(len(flattened))

# Vocab=[]
# for item in flattened:
#     if item not in Vocab and item not in STOP_WORDS:
#         Vocab.append(item)

# Vocab = np.array(Vocab)
# print(len(Vocab))

