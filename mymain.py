import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

def training_function():
    start = datetime.now()
    start_time = start.strftime("%H:%M:%S")
    res = []
    with open("final_vocab.txt") as f:
        new_vocab = f.readlines()
        new_vocab = [v[:-1].replace("_"," ") for v in new_vocab]
    print('>>>>> Vocab loading is done <<<<<')
    print('>>>>> Start Training <<<<<')
    for i in range(1,6):
        print(f'>>>>> Fold:{i} <<<<<')
        folderName = 'split_' + str(i)
        train_filename = folderName + '/' + 'train.tsv'
        test_filename = folderName + '/' + 'test.tsv'
        test_y_filename = folderName + '/' + 'test_y.tsv'

        train_data = pd.read_csv(train_filename,sep='\t', header=0)
        train_y = train_data['sentiment']
        train_features = train_data.copy()
        train_features = train_features.drop(['sentiment'],axis=1)

        test_data = pd.read_csv(test_filename,sep='\t', header=0)
        test_features = test_data['review']
        test_y_data = pd.read_csv(test_y_filename,sep='\t', header=0)

        # Basic function to clean the text 
        def clean_text(text):     
            return text.strip().lower()

        vectorizer = CountVectorizer(ngram_range=(1, 4),vocabulary=new_vocab)
        tfvectorizer = TfidfVectorizer()


        classifier = LogisticRegression(penalty='l2')
        LRmodel = Pipeline([
                        ('vectorizer', vectorizer),
                        ('classifier', classifier)])

        test_Y = test_y_data['sentiment']

        LRmodel.fit(train_features['review'],train_y)
        LRpred = LRmodel.predict_proba(test_features)
        LRpred = LRpred[:,1]
        result = LRmodel.predict(test_features)
        print(f'Accuracy: {accuracy_score(test_Y,result)*100}%')
        auc = roc_auc_score(test_Y, LRpred,average='micro')
        print(f'AUC: {auc*100}%')
        res.append(auc)
    print('>>>>> All folds are done <<<<<')
    mean_auc = np.mean(res)

    print(f'mean AUC:{mean_auc}')

    end = datetime.now()
    end_time = end.strftime("%H:%M:%S")
    used_time = end-start
    print(f'Runding time:{used_time}')

def myprediction(trainfile,testfile,myvocab):
    ### load train ###
    train_data = pd.read_csv(trainfile,sep='\t', header=0)
    train_y = train_data['sentiment']
    train_features = train_data.copy()
    train_features = train_features.drop(['sentiment'],axis=1)

    ### load test ###
    test_data = pd.read_csv(testfile,sep='\t', header=0)
    test_features = test_data['review']

    with open(myvocab) as f:
        new_vocab = f.readlines()
        new_vocab = [v[:-1].replace("_"," ") for v in new_vocab]
    print('>>>>> Data is loaded <<<<<')
    def clean_text(text):     
            return text.strip().lower()
    
    vectorizer = CountVectorizer(ngram_range=(1, 4),vocabulary=new_vocab)
    tfvectorizer = TfidfVectorizer()

    ### define the model ###
    classifier = LogisticRegression(penalty='l2')
    LRmodel = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)])

    LRmodel.fit(train_features['review'],train_y)
    LRpred = LRmodel.predict_proba(test_features)
    LRpred = LRpred[:,1]
    Predict_prob = pd.DataFrame(LRpred, columns=['prob'])

    pid = test_data["id"]
    res = pd.concat([pid,Predict_prob], axis = 1)

    res.to_csv("mysubmission.txt",index=None, sep='\t', mode='w')
    print('mysubmission.txt saving is done!')



if __name__ == '__main__':
    # training_function()

    trainfile = 'train.tsv'
    testfile = 'test.tsv'
    myvocab = "myvocab.txt"
    myprediction(trainfile,testfile,myvocab)


