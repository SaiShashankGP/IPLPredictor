# IPLPredictor
## Introduction
Inspired from Nomura Quant Challenge 2022 Problem Statement, I created an IPL score predictor for first innings. Using the data given, I have created some features for the model to train on and I have used Linear Regression from sklearn library as the model to predict scores.
## Data
The data consists of each ball as a data point each ball has the following features, 
- Match ID of each match(integer)
- Batting Team Name(object)
- Bowling Team Name(object)
- Over(integer)
- Ball(integer)
- Batsman(object)
- Non Striker(object)
- Bowler(object)
- Wide Runs(integer)
- No Ball Runs(integer)
- Bye Runs(integer)
- Legbye Runs(integer)
- Penalty Runs(integer)
- Batsman Runs(integer)
- Extra Runs(integer)
- Total Runs(integer)
- Dismissed Player Name(object)
- Dismissal Kind(object)
- Fielder Name(object)<br>
The training data consists of 300 matches with all the first inning's data. The testing data consists of 100 matches with only some overs from each match. So, we have to somehow draw conclusions from the given data about the whole innings. So, we need somehow sum up the innings into a single vector/row of numbers which describes the match better instead of having a lot of datapoints together describing the match.
## Approach
Since our goal is to predict final score of the match. I planned on using Linear Regression to predict the score. But I had many data points for a single match which has only one final score. So, I planned on deriving some important features from the given features. The features I decided upon were, 
- Avg. Current Run Rate
- Avg. Strike Rate of Batsman
- Avg. Economy of Bowler
- Maximum Score of a Batsman
- Expected Extra Runs per ball<br>
I arrived at these datapoints from the experience of watching many IPL matches. And these features also describe how the batting team and bowling team contributed to the score.<br>
After preparing the training features from the given data, I split them into training and validation with the ratio of 22:3, which I obtained by trail and error, and trained the linear regression model to fit to the data.<br>
After validating the model using the validation set, I obtained the measure of 41.61 sq runs/sample with mean squared error as metric. Following is the scatter plot showing the predictions and actual score.<br>

