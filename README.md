# Disaster-Tweets
Predict.dat - temporary file created when KNN runs
Predict.csv - the file that is created for submission to leaderboard. Currently, this predict file is my highest F1 score on the leaderboard
sample_submission.csv - a kaggle submission example
test.csv - kaggle testing data
train.csv - kaggle training data


Prediciting.py

      gatherData(file1, file2) 
- This function uses the training and testing file names as arguments. This functions reads and gathers both the training data and testing data into a Dataframe. Removes stopwords and uses TFIDF on the text data. Returns the training data, training labels, and testing data, the training data frame. 
<br>
      
      Remove(df) 
- This function reads the text data and does the following to each line:  <br>
a.	Changed to lower case <br>
b.	Removed punctuation <br>
c.	Lemmatization/Stemming <br>
d.	Removed URLS <br>
e.	Removed # symbol <br>
f.  It then returns the Dataframe. <br>
<br>

    NearestNeighbor(train, test, k, labels, df) 
- This function: <br>
a.	Loop through and use cosine similarity on the testing array value and training array value <br>
b.	Add this consine similarity to data list <br>
c.	Reverse the sorted array to decreasing order <br>
d.	Choose the best up to k items from data list <br>
e.   Count the positive and negatives, whatever wins: <br>
>>1. 1 for positive wins <br>
>>2. 0 for negative wins <br>

&nbsp; &nbsp; &nbsp; &nbsp; f.  Decided not to choose even k values so I didn’t fix the the 3rd rule: <br>
>> &nbsp; &nbsp; &nbsp; &nbsp; 	1. If there is a tie between positive and negative, it is suppose to pick the closest one (as of right now, it does not work for even k values) <br>

&nbsp; &nbsp; &nbsp; &nbsp;e.	Add to predict array (for getting to write to a csv file) and prediction file (backup in case something happens to the predict array) <br>
<br>

      Predict_file(file1, c, y_test) 
- This function uses the sample submission file to get the Ids, and creates a dataframe. It then adds the targeted predictions into a column. Then makes a predict.csv file. 
<br>

      Main 
- It tries 5 sklearn classifiers, gathers the F1 scores of each classifier and times it. A table of F1 scores is then printing. It then creates a predict file for BernoullinNB, which has the highest F1 score on the data. The random state is 0. Separately, tries KNN and creates a predict file. KNN has been commented out. Since, it takes a long time. 
