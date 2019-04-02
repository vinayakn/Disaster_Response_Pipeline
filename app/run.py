import json;
import plotly;
import pandas as pd;

from nltk.stem import WordNetLemmatizer;
from nltk.tokenize import word_tokenize;

from flask import Flask;
from flask import render_template, request, jsonify;
from plotly.graph_objs import Bar,Histogram;
from sklearn.externals import joblib;
from sqlalchemy import create_engine;


app = Flask(__name__);

def tokenize(text):
    print("running tokenize funciton");
    tokens = word_tokenize(text);
    lemmatizer = WordNetLemmatizer();

    clean_tokens = [];
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip();
        clean_tokens.append(clean_tok);

    return clean_tokens;

# load data
print("loading data");
engine = create_engine('sqlite:///../data/disaster_messages.db');
df = pd.read_sql_table('messages', engine);

# load model
print("loading model");
model = joblib.load("../models/model.pkl");


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    print("running index function")
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message'];
    genre_names = list(genre_counts.index);
    
	# Graph2 Frequency of  Message's  by category
    categories = df.columns[4:].tolist()
    messgs_count = df.iloc[:, 4:].sum().tolist()
	
	## Graph3 heatmap

	
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
					marker=dict(color='orange')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
		#Graph2:Frequency of message count Histogram
		{
            'data': [
                Histogram(
                    x = categories,
                    y = messgs_count,
                    histfunc='sum',
                    marker=dict(color='purple')
                )
            ],

            'layout': {
                'title': 'Frequency distribution Messages by Categories',
                'yaxis': {
                    'title': "Frequency of Message by Category"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }		

    ];
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)];
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder);
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON);


# web page that handles user query and displays model results
@app.route('/go')
def go():
    print("running go function")
    # save user input in query
    query = request.args.get('query', '') ;

    # use model to predict classification for query
    classification_labels = model.predict([query])[0];
    classification_results = dict(zip(df.columns[4:], classification_labels));

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    );


def main():
    print("started running app")
    #app.run(host='0.0.0.0', port=3001, debug=True);
    app.run(host='127.0.0.1', port=3002, debug=True);


if __name__ == '__main__':
    main();