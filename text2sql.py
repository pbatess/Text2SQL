import json
from collections import Counter
import numpy as np
import pandas as pd
import re
import nltk
from nltk.data import find
import gensim
import sklearn
from sympy.parsing.sympy_parser import parse_expr
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

np.random.seed(0)
nltk.download('word2vec_sample')

class Text2SQLParser:
    def __init__(self):
        """
        Basic Text2SQL Parser. This module just attempts to classify the user queries into different "categories" of SQL queries.
        """
        self.parser_files = "data/semantic-parser"
        self.word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_sample, binary=False)

        self.train_file = "sql_train.tsv"
        self.test_file = "sql_val.tsv"

    def load_data(self):
        """
        Load the data from file.

        """
        self.train_df = pd.read_csv(self.parser_files + "/" + self.train_file, sep="\t")
        self.test_df = pd.read_csv(self.parser_files + "/" + self.test_file, sep="\t")

        self.ls_labels = list(self.train_df["Label"].unique())

    def predict_label_using_keywords(self, question):
        """
        Predicts the label for the question using custom-defined keywords.

        """

        label= ""
        keywords = {

            'comparison': {'greater', 'less', 'equal', 'between', 'compare', 'difference', 'versus', 'match'},
            'grouping': {'group', 'count', 'sum', 'average', 'having', 'aggregate', 'total', 'each', 'correspond'},
            'ordering': {'order', 'sort', 'rank', 'top', 'limit', 'ascending', 'descending'},
            'multi_table': {'join', 'foreign key', 'inner', 'outer', 'left', 'right', 'cross', 'combine', 'dataset', 'merge', 'relation', 'physician', 'order_id', 'address_id', 'product_id', 'customer_id', 'patient', 'dept_code', 'other_details'}
        }
        

        question_lower = question.lower()
        
        multi_table_matches = [word for word in keywords['multi_table'] if word in question_lower]
        if len(multi_table_matches) >= 2:
            return 'multi_table'
        match_counts = {label: sum(word in question_lower for word in words) for label, words in keywords.items()}

        label = max(match_counts, key=match_counts.get, default='comparison')

            
        return label
    
    def evaluate_accuracy(self, prediction_function_name):
        """
        Gives label wise accuracy of the model.

        """
        correct = Counter()
        total = Counter()
        main_acc = 0
        main_cnt = 0
        
        for i in range(len(self.test_df)):
            q = self.test_df.loc[i]["Question"].split(":")[1].split("|")[0].strip()
            gold_label = self.test_df.loc[i]['Label']
            if prediction_function_name(q) == gold_label:
                correct[gold_label] += 1
                main_acc += 1
            total[gold_label] += 1
            main_cnt += 1
        accs = {}
        for label in self.ls_labels:
            accs[label] = (correct[label]/total[label])*100
            
        return accs, 100*main_acc/main_cnt
        

    def get_sentence_representation(self, sentence):
        """
        Gives the average word2vec representation of a sentence.

        """
        words = sentence.lower().split()
        vectors = [self.word2vec_model[word] for word in words if word in self.word2vec_model]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(300)


    def init_ml_classifier(self):
        """
        Initializes the ML classifier.

        """
        self.classifier = sklearn.linear_model.LogisticRegression(max_iter=1000)
        


    def train_label_ml_classifier(self):
        """
        Train the classifier.
        
        """
        X_train = [self.get_sentence_representation(q) for q in self.train_df['Question']]
        y_train = self.train_df['Label']
        
        self.classifier.fit(X_train, y_train)


    
    def predict_label_using_ml_classifier(self, question):
        """
        Predicts the label of the question using the classifier.

        """
        sentence_vector = self.get_sentence_representation(question).reshape(1, -1)
        return self.classifier.predict(sentence_vector)[0]
    
print("======================================================================")
print("Checking Text2SQL Parser")
print("======================================================================")

# Define text2sql parser object
sql_parser = Text2SQLParser()

# Load the data files
sql_parser.load_data()

# Initialize the ML classifier
sql_parser.init_ml_classifier()

# Train the classifier
sql_parser.train_label_ml_classifier()

# Evaluating the keyword-based label classifier. 
print("------------- Evaluating keyword-based label classifier -------------")
accs, _ = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_keywords)
for label in accs:
	print(label + ": " + str(accs[label]))

# Evaluate the ML classifier
print("------------- Evaluating ML classifier -------------")
sql_parser.train_label_ml_classifier()
_, overall_acc = sql_parser.evaluate_accuracy(sql_parser.predict_label_using_ml_classifier)
print("Overall accuracy: ", str(overall_acc))

print()
print()
