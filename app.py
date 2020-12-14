from flask import Flask
from flask import Flask, session, url_for, redirect, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import pickle
from pandas import DataFrame

with open('./datasets/pickle_data/final_model_multinomial.pkl', 'rb') as pickle_file:
    model = pickle.load(pickle_file)
with open('./datasets/pickle_data/vectorizer.pkl', 'rb') as pickle_file:
    vectorizer = pickle.load(pickle_file)
print("Model and Vectorizer rrun from file")

app = Flask(__name__)

app.config['SECRET_KEY'] = 'jonathan'
 

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
        input_testing_features = vectorizer.transform(input_df['comment'])
        predict = model.predict(input_testing_features)
        answer = predict[0]
        return render_template('answer.html', answer = answer, form = form)

    return render_template('question.html', form= form, comment = comment)

if __name__ == '__main__':
    app.run(debug = True)
