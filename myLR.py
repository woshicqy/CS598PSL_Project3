import numpy as np
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
res = []
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
    # print(test_data.head(5))
    # print(test_data.shape)
    test_y_data = pd.read_csv(test_y_filename,sep='\t', header=0)

    def tokenizer(sentence):
        nlp = English()
        stopwords = list(STOP_WORDS)
        punctuations = '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
        mytokens = nlp(sentence)
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
        mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
        return mytokens

    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [clean_text(text) for text in X]
        def fit(self, X, y, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}

    # Basic function to clean the text 
    def clean_text(text):     
        return text.strip().lower()

    vectorizer = CountVectorizer() 
    tfvectorizer = TfidfVectorizer()


    classifier = LogisticRegression()
    LRmodel = Pipeline([("cleaner", predictors()),
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)])

    # Train the Model
    print('>>>>> Start Training <<<<<')
    from datetime import datetime

    now = datetime.now()
    test_Y = test_y_data['sentiment']
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    LRmodel.fit(train_features['review'],train_y)
    print('>>>>> Training is done <<<<<')
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Finish Training Time =", current_time)
    LRpred = LRmodel.predict(test_features)
    # print('prediction shape:',LRpred.shape)
    # print('prediction:',LRpred)
    # print('test Y shape:',test_Y.shape)
    # exit()
    print(f'Accuracy: {accuracy_score(test_Y,LRpred)*100}%')
    now = datetime.now()
    auc = roc_auc_score(test_Y, LRpred,average='micro')

    current_time = now.strftime("%H:%M:%S")
    print("All done Time =", current_time)
    print(f'AUC: {auc*100}%')
    res.append(auc)

print('>>>>> All folds are done <<<<<')
mean_auc = np.mean(res)

print(f'mean AUC:{mean_auc}')