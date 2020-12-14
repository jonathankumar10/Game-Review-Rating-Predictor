from flask import Flask
from flask import Flask, session, url_for, redirect, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import pickle
import re
import nltk
import string
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
import pandas as pd
from pandas import DataFrame

with open('./datasets/pickle_data/final_model_multinomial.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
with open('./datasets/pickle_data/vectorizer.pkl', 'rb') as pickle_file:
    vectorizer = pickle.load(pickle_file)
print("Model and Vectorizer rrun from file")

app = Flask(__name__)

app.config['SECRET_KEY'] = 'jonathan'
 
def clean_comments(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'re", " are", text)

    text = re.sub(r"[0-9]+", ' ', text)
    text = re.sub(r"-", ' ', text)
    
    
    text = text.strip().lower()
    

    default_stop_words = set(stopwords.words('english'))
    default_stop_words.difference_update({'no', 'not', 'nor', 'too', 'any'})
    stop_words = default_stop_words.union({"'m", "n't", "'d", "'re", "'s",
                                           'would','must',"'ve","'ll",'may'})

    word_list = word_tokenize(text)
    filtered_list = [w for w in word_list if not w in stop_words]
    text = ' '.join(filtered_list)
    
    text = re.sub(r"'", ' ', text)
    
   
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((i, " ") for i in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    

    text = ' '.join([w for w in text.split() if len(w)>1])

    # Replace multiple space with one space
    text = re.sub(' +', ' ', text)
    
    text = ''.join(text)

    return text


def NormalizeWithPOS(text):
    # Lemmatization & Stemming according to POS tagging

    word_list = word_tokenize(text)
    rev = []
    lemmatizer = WordNetLemmatizer() 
    stemmer = PorterStemmer() 
    for word, tag in pos_tag(word_list):
        if tag.startswith('J'):
            w = lemmatizer.lemmatize(word, pos='a')
        elif tag.startswith('V'):
            w = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('N'):
            w = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('R'):
            w = lemmatizer.lemmatize(word, pos='r')
        else:
            w = word
        w = stemmer.stem(w)
        rev.append(w)
    review = ' '.join(rev)
    return review

class InfoForm(FlaskForm):

    comment = StringField('Enter your Comment/Review :')
    submit = SubmitField('Submit')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/question.html', methods =['GET','POST'])
def question():
    comment = False
    form = InfoForm()
    if form.validate_on_submit():
        comment = form.comment.data
        form.comment.data = ''
        Input = [comment]
        input_df = DataFrame(Input,columns=['comment'])
        input_df['clean_comment'] = input_df['comment'].apply(clean_comments)
        input_df['clean_comment'] = input_df['clean_comment'].apply(NormalizeWithPOS)
        input_testing_features = vectorizer.transform(input_df['clean_comment'])
        predict = model.predict(input_testing_features)
        answer = predict[0]
        return render_template('answer.html', answer = answer, form = form)

    return render_template('question.html', form= form, comment = comment)

if __name__ == '__main__':
    app.run(debug = True)
