"""

Name                : Sai Shashank GP
Date                : 25-06-2022
Problem Statement   : Predict the 1st innings score of IPL matches
Approach            : Using Linear Regression and creating new parameters out of given ones
Libraries used      : Pandas, Numpy, Matplotlib, Scikit-learn 

"""

# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


# Data Pre-Processing

ipl_train = pd.read_csv('IPL_train.csv')
train_data = pd.DataFrame(ipl_train)

matches = train_data.groupby(train_data.match_id)

class Batsman:
    def __init__(self, match, name):
        self.match = match.copy()
        self.name = name
        self.stats = self.match.groupby(self.match.batsman).get_group(self.name)
        self.scorechart = self.stats.batsman_runs
        self.balls_played = self.stats.shape[0]
        self.finalSR = np.sum(self.scorechart)*100/self.balls_played
        self.isDismissed = self.stats.player_dismissed.isnull().values.all()

    def calcInstSR(self, over, ball):
        score = 0
        sr = 0
        balls = 0
        for _, row in self.stats.iterrows():
            if row['over'] == over and row['ball'] == ball:
                return sr
            score += row['batsman_runs']
            balls += 1
            sr = score*100/balls

class Bowler:
    def __init__(self, match, name):
        self.match = match.copy()
        self.name = name
        self.stats = self.match.groupby(self.match.bowler).get_group(self.name)
        self.runschart = self.stats.total_runs
        self.balls_played = self.stats.shape[0]
        self.finalEco = np.sum(self.runschart)*6/self.balls_played

    def calcInstEco(self, over, ball):
        score = 0
        eco = 0
        balls = 0
        for _, row in self.stats.iterrows():
            if row['over'] == over and row['ball'] == ball:
                return eco
            score += row['total_runs']
            if row['wide_runs'] == 0 or row['noball_runs'] == 0:
                balls += 1
            eco = score*6/balls

class Match:
    def __init__(self, match):
        self.match = match.copy()
        self.match_id = self.match.match_id.unique()[0]
        self.match.fillna(0, inplace=True)
        self.batsmen = dict([(i, Batsman(self.match, i)) for i in self.match.batsman.unique()])
        self.bowlers = dict([(i, Bowler(self.match, i)) for i in self.match.bowler.unique()])

    def createCRRcol(self):
        match_copy = self.match.copy()
        score = 0
        balls_played = 0
        runrate = [0]
        for i in range(self.match.index[0]+1, self.match.index[-1]+1):
            score += match_copy.total_runs[i-1]
            if match_copy.wide_runs[i] == 0 or match_copy.noball_runs[i] == 0:
                balls_played += 1
            crr = score*6/balls_played
            runrate.append(crr)
        self.match['CRR'] = runrate

    def createExpExtra(self):
        match_copy = self.match.copy()
        extracol = [0]
        extras = 0
        balls_played = 0
        for i in range(self.match.index[0]+1, self.match.index[-1]+1):
            extras += match_copy.extra_runs[i-1]
            if match_copy.extra_runs[i] != 0:
                balls_played += 1
            if balls_played != 0:
                expextra = extras/balls_played
            else:
                expextra = 0
            extracol.append(expextra)
        self.match['ExpExtra'] = extracol

    def createScoreCol(self):
        match_copy = self.match.copy()
        scorecol = []
        score = 0
        for i in range(self.match.index[0], self.match.index[-1]+1):
            score += match_copy.total_runs[i]
            scorecol.append(score)
        self.match['Score'] = scorecol

def compressMatch(match):
    match_obj = Match(match)

    match_obj.createCRRcol()
    avgRR = np.mean(match_obj.match.CRR)

    avgSR = np.mean(np.array([i.finalSR for i in match_obj.batsmen.values()]))

    Score1 = np.max(np.array([np.sum(i.scorechart) for i in match_obj.batsmen.values()]))    

    avgEco = np.mean(np.array([i.finalEco for i in match_obj.bowlers.values()]))

    match_obj.createExpExtra()
    avgExtra = match_obj.match.ExpExtra.tail(1).item()

    match_id = match_obj.match_id

    match_obj.createScoreCol()
    score = np.max(match_obj.match.Score)

    return [match_id, avgRR, avgSR/100, Score1/100, avgEco, avgExtra, score/10]

def createX(matches):
    rows = []
    for _, match in matches:
        row = compressMatch(match)
        rows.append(row)
    col_names = ['match_id', 'avgRR', 'SRscaled', 'Score1scaled', 'Ecoscaled', 'Extras', 'Scorescaled']
    df = pd.DataFrame(rows, columns=col_names)
    return df

df = createX(matches)

Y_data = df['Scorescaled']
df.drop(['match_id', 'Scorescaled'], axis=1, inplace=True)
X_data = df

# Training the model

f = 0.12

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=f, random_state=0)

lr = LinearRegression(fit_intercept=False)
lr.fit(X_train, Y_train)

# Validating the model

Y_pred = lr.predict(X_test)

# print(mse(Y_pred*10, Y_test*10))              Obtained value: 41.614

# plt.plot(Y_test, Y_pred, 'r.')
# plt.plot(Y_test, Y_test)
# plt.show()

# Testing the model

ipl_test = pd.read_csv('IPL_test.csv')
test_data = pd.DataFrame(ipl_test)

test_matches = test_data.groupby(test_data.match_id)

test_df = createX(test_matches)
test_df.drop(['match_id', 'Scorescaled'], axis=1, inplace=True)
x_data = test_df

test_pred = lr.predict(x_data)

final_predictions = test_pred*10

final_df = pd.DataFrame(final_predictions, columns=['predictions'])
final_df.to_csv('IPL_test_predictions.csv')

