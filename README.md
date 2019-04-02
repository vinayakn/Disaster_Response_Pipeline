# Disaster Response Pipeline Project


### Table Contents:

|S.NO| Description | README |
|--| ------ | ------ |
|1|Summary|[GOTO]()|
|2|Installation | [GOTO]()|
|3|Instructions | [GOTO]() |
|4|The file structure | [GOTO]() |

# Summary

The project used a data from Figure Eight that contained labeled disaster messages received by an aid organization. 
A multi-output Random Forrest classifier is trained using a natural language processing (NLP).
This model is then used as back-end for a web app where a user can search any message and see the categories it belongs to.
The web app also displays useful visualizations about the training data available.

Data is Imbalanced due to which classifer Accuracy is high whoever we notice f1-score is not that great!!.
Note: Takes long time to classfy the model(almost 6 to 7 hours).

##### **Steps Followod:**
* An ETL pipeline was created, extracting data from csv files, cleaning and loaded into an SQL database.
* A machine learning pipeline was created to extract the NLP features and then optimize the algorithm using grid search.
* A web app was then developed that extracts the initial data from the database and provides some interactive visual summaries.
* Users are also able to search their own message to be classified one of the cetegoires of the algorithm.

# Installation
#### **Libraries:** sys,pandas,numpy,re,sqlite3, sqlalchemy,sklearn, nltk, plotly, flask,json
#### **HTML:** Bootstrap


# The file structure: 

-**app**
|- template
|	|- master.html 	# main page of web app
|	|- go.html 	  	# classification result page of web app
|- run.py 			# Flask file that runs app

-**data**
|- disaster_categories.csv # data to process
|- disaster_messages.csv   # data to process
|- process_data.py         # code to process
|- disaster_messages.db    # cleaned data

-**models**
|- train_classifier.py  #Model builder
|- model.pkl 			#saved model

-**README.md**

# Instructions:
1. **Run the following commands in the project's root directory to set up your database and model.**

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/disaster_messages.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3002/
