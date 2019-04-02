import sys;

import sqlalchemy;
from sqlalchemy import create_engine;
import pandas as pd;
import re;
import numpy as np;
import pandas as pd;

import nltk;
from nltk.corpus import stopwords;
from nltk.tokenize import word_tokenize;
from nltk.stem import WordNetLemmatizer;
nltk.download('stopwords');
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger']);

from sklearn.model_selection import GridSearchCV;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.pipeline import Pipeline;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.multioutput import MultiOutputClassifier;
from sklearn.metrics import classification_report;
from sklearn.externals import joblib;


def load_data(database_filepath):
    '''
    load data from database
    Input:
        database_filepath: a path to the database
    Output:
        X: input Features to the model 
        Y: output/Target feature to the model
        category_names: Column names for categories ie. Target feature Y
    '''
    #create db engine
    print("create engine");
    engine = create_engine('sqlite:///'+database_filepath);
    #read sql table messages into df dataframe
    print("load table into df");
    df = pd.read_sql_table('messages', engine);
    #create variable X & Y for Input Feature and Output features for Model
    print("create input output features and category columns names");
    X = df.message;
    Y = df.iloc[:, 4:];
    #save columns of Y as list  
    category_names=Y.columns.values.tolist();
    #return X,Y and category_names
    print("input,output,category names:");
    return X, Y, category_names;
    


def tokenize(text):
    '''
    Tokenize the input text and return a clean tokens as to process the text data.
    Do normalization,then removing stopwords and finally lemmatization
    Input:
        text: provide text
    Output:
        clean_tokens: get list of clean tokens
    '''
    # First we will do normalization of data
    #print("normalization");
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower());
    tokens = word_tokenize(text);
   
    # Next will remove stopwords
    #print("stopwords removal");
    tokens = [w for w in tokens if w not in stopwords.words("english")];

    # then will apply lemmatizer to all of our tokens
    #print("apply lemmatizer");
    lemmatizer = WordNetLemmatizer();

    #create clean_tokens list to save all the clean tokens
    clean_tokens = [];
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip();
        clean_tokens.append(clean_tok);

    #return clean token list
    #print("return tokens");
    return clean_tokens;
    #pass
    

def build_model():
    '''
    Creates a multi-output Random Forest classifier machine learning pipeline for
    natural language processing with CountVectorizer, tf-idf, ultiOutput RandomForestClassifier Classifier,
    and GridSearchCV features. 
    
    Output:
        cv:Returns Machine Learning pipeline with best parameter for model ie. CV
    '''
    #creating pipeline
    print("crate pipline");
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    
    #finding best parameter using grid serach
    print("set parameters for gridseachcv");
    parameters = {'clf__estimator__n_estimators': [50,100,150],'clf__estimator__min_samples_split': [2, 3, 5]};
    
    print("return pipline and paramenter")
    
    cv = GridSearchCV(pipeline,parameters);
    
    #return cv
    print("return cv")
    return cv;
    #pass


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print precision, recall, fscore for all the categories for a multi-output 
    classification
    
    Input:
        model : trained model
        X_test: test set input features 
        Y_test: test set target features
        category_names: names of different target features categories
    output:
        print category, precision, recall, fscore
        
    '''
    #predict on test data
    print("predict the model");
    y_pred = model.predict(X_test);
    print("print results")
    results_cv_dict = {};

    for pred, label, col in zip(y_pred.transpose(), Y_test.values.transpose(), category_names):
        print(category_names);
        print(classification_report(label, pred));
        results_cv_dict[col] = classification_report(label, pred, output_dict=True);
    
def save_model(model, model_filepath):
    '''
    Save the model 
    Input:
        model: model which is built
        model_filepath: file path location where model will be saved 
    '''
	#save model
    print("save model");
    joblib.dump(model, model_filepath)
	#pass


def main():
    if len(sys.argv) == 3:
        print(len(sys.argv));
        print(sys.argv);
        database_filepath, model_filepath = sys.argv[1:]
        print(database_filepath,model_filepath);
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        print(len(X_train), len(X_test), len(Y_train), len(Y_test))
        
        print('Building model...')
        model = build_model();
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()