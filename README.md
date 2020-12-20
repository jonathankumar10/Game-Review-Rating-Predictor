# Game_Review_Rating_Predctor

Introduction:
The goal of the term project is building a classifier a rating given out as the output given a particular input comment/review.
We had to work on Kaggle dataset and train a model of our choosing and also host this model on a website so that we could predict the rating of a particular review/comment.
We also had to submit this project on Kaggle so that we could participate in the Kaggle challenge.
Coding the right solution was only half of the output. Apart from that as a primary objective we had to create a well written report and also create a blog on our portfolio webpage.

Data:
We used the board game geek review data. This data was present in the following link:   
•	https://www.kaggle.com/jvanelteren/boardgamegeek-reviews
The dataset provided had 3 csv files, out of which I had to work only on one csv file making my work a little bit easier. The file name being: 
•	bgg-15m-reviews
This csv file consisted of User, Rating, comment, ID and name columns.

Data Preprocessing:
For data preprocessing I used a number of methods to clean and convert the data into a format which was required for my model to predict the ratings.
•	Firstly, I deleted unwanted columns. The names of the columns being Unnamed: 0 User, ID and Name.
•	Secondly, all the rows containing null values were dropped. 
•	Post that the rating column needed to be worked on as it contained float values ranging between 1 to 10. These values were rounded off to ensure efficiency of the model.
•	Next, we needed to clean the comments section of the csv file. To do that we cleaned the data using various helper libraries such as nltk for removal of stop words, lemmatization, stemming, parts of speech tagging, removal of punctuations, removal of html tags and also removal of least frequently occurring words.
•	Post that the data was then split into train and test using sklearn’s train_test_split function/library.
•	Next came conversion of these words post cleaning to be mapped to numbers. For this purpose, Count vectorizer was used.


Model Creation and Hyperparameter Tuning:
•	The model chosen for this project was Multinomial Naive Bayes and the accuracy metric chosen was mean squared error.
•	Multinomial Naïve Bayes was one of the models which gave  out the least mean squared error for the accuracy metric. Some of the other models that came close were Random Forest and KNN.
•	Multinomial Naïve Bayes was imported from sklearns machine learning library.
•	The training data was then fitted onto the model and was tested for all of the test data
•	The base model gave a good Mean squared error score of 3.30
•	The best model gave a better mean squared error score of 3.11

The most challenging part of this projectwas dealing with a huge dataset. To handle this problem I had to reduce the dataset by removing the null values attrubutes. Also, Reduced the 10 least frequency words out of the dataset.

Link to my website:
https://jonathankumar10.github.io/
