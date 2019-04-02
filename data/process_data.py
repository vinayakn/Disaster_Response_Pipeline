##Vinayak .V. Navale

#import all the required libraries
import sys;
import pandas as pd;
import numpy as np;
from sqlalchemy import create_engine;

def load_data(messages_filepath, categories_filepath):
    '''
    Load CSV Data files Into DataFrames:messages & categories
    and Merge above dataframe and create DataFrame :df
        
    Input:
        messages_filepath: Contains Message Dataset
        categories_filepath:Contains Categories Dataset 
    Output:
        df:created Dataframe 'DF' by merging both messages & categories dataframe
    
    '''
    #loading files to dataframe
    messages=pd.read_csv(messages_filepath);
    categories = pd.read_csv(categories_filepath);
    
    #merging both above dataframe and creating df dataframe
    df = pd.merge(messages, categories, on='id');
    return df;    
#    pass


def clean_data(df):
    '''
    Cleaning data: Load df Dataframe and Clean it 
    
    Input:
        Merged dataframe:df
    Output:
        Cleaned DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True));
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1];
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int);
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True);
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1);
    # check number of duplicates
    print("Count of Duplicates: ",len(df[df.duplicated() == True]));
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    print("Count of Duplicates: ",len(df[df.duplicated() == True]));
    #return clean df
    return df;
#    pass


def save_data(df, database_filename):
    '''
    Load Cleaned DataFrame df and save it to database
    '''
    #Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///'+database_filename);
    df.to_sql('messages', engine, index=False, chunksize=1000, if_exists='replace'); 

    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()