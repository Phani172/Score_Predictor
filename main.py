import pandas as pd
import numpy as np
#Importing dataset
df = pd.read_csv('./score_data.csv')
# print(f"Dataset successfully Imported of Shape : {df.shape}")

import seaborn as sns
import matplotlib.pyplot as plt

notrequired = ['mid', 'date', 'venue','batsman', 'bowler', 'striker', 'non-striker']
# print(f'Before Removing Irrelevant Columns : {df.shape}')
ipl_df = df.drop(notrequired, axis=1) # Drop Irrelevant Columns
# print(f'After Removing Irrelevant Columns : {ipl_df.shape}')
# Define Consistent Teams
const_teams = [ 'Royal Challengers Bangalore', 'Chennai Super Kings',
              'Mumbai Indians','Kolkata Knight Riders',
               'Sunrisers Hyderabad']
# print(f'Before Removing Inconsistent Teams : {ipl_df.shape}')
ipl_df = ipl_df[(ipl_df['bat_team'].isin(const_teams)) & (ipl_df['bowl_team'].isin(const_teams))]
# print(f'After Removing Irrelevant Columns : {ipl_df.shape}')

# print(f'Before Removing Overs : {ipl_df.shape}')
ipl_df = ipl_df[ipl_df['overs'] >= 5.0]
# print(f'After Removing Overs : {ipl_df.shape}')
# Exclude non-numeric columns from the correlation matrix
numeric_cols = ipl_df.select_dtypes(include=[np.number])


# Plot the correlation matrix using seaborn's heatmap
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for col in ['bat_team', 'bowl_team']:
  ipl_df[col] = le.fit_transform(ipl_df[col])
ipl_df.shape
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder',
                                        OneHotEncoder(),
                                        [0, 1])],
                                      remainder='passthrough')
ipl_df = np.array(columnTransformer.fit_transform(ipl_df))
ipl_df.shape
cols = ['batting_team_Chennai Super Kings',
              'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians',
              'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
              'bowling_team_Chennai Super Kings',
              'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians',
              'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad', 'runs', 'wickets', 'overs',
       'runs_last_5', 'wickets_last_5', 'total']
df = pd.DataFrame(ipl_df, columns=cols)
final_df=df
features = final_df.drop(['total'], axis=1)
labels = final_df['total']
from sklearn.model_selection import train_test_split
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.20, shuffle=True)
# print(f"Training Set : {train_features.shape}\nTesting Set : {test_features.shape}")
models = dict()
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
# Train Model
tree.fit(train_features, train_labels)
# Evaluate Model
train_score_tree = str(tree.score(train_features, train_labels) * 100)
test_score_tree = str(tree.score(test_features, test_labels) * 100)
# print(f'Train Score : {train_score_tree[:5]}%\nTest Score : {test_score_tree[:5]}%')
models["tree"] = test_score_tree
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
# print("---- Decision Tree Regressor - Model Evaluation ----")
# print("Mean Absolute Error (MAE): {}".format(mae(test_labels, tree.predict(test_features))))
# print("Mean Squared Error (MSE): {}".format(mse(test_labels, tree.predict(test_features))))
# print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, tree.predict(test_features)))))
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
# Train Model
forest.fit(train_features, train_labels)
# Evaluate Model
train_score_forest = str(forest.score(train_features, train_labels)*100)
test_score_forest = str(forest.score(test_features, test_labels)*100)
# print(f'Train Score : {train_score_forest[:5]}%\nTest Score : {test_score_forest[:5]}%')
models["forest"] = test_score_forest
# print("---- Random Forest Regression - Model Evaluation ----")
# print("Mean Absolute Error (MAE): {}".format(mae(test_labels, forest.predict(test_features))))
# print("Mean Squared Error (MSE): {}".format(mse(test_labels, forest.predict(test_features))))
# print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(test_labels, forest.predict(test_features)))))
import matplotlib.pyplot as plt
model_names = list(models.keys())
accuracy = list(map(float, models.values()))
# creating the bar plot
plt.bar(model_names, accuracy)
def score_predict(batting_team, bowling_team, runs, wickets, overs, runs_last_5, wickets_last_5, model=forest):
  prediction_array = []
  # Batting Team
  if batting_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0]
  elif batting_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,1,0,0,0]
  elif batting_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,1,0,0]
  elif batting_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,1,0]
  elif batting_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,1]
  # Bowling Team
  if bowling_team == 'Chennai Super Kings':
    prediction_array = prediction_array + [1,0,0,0,0]
  elif bowling_team == 'Kolkata Knight Riders':
    prediction_array = prediction_array + [0,1,0,0,0]
  elif bowling_team == 'Mumbai Indians':
    prediction_array = prediction_array + [0,0,1,0,0]
  elif bowling_team == 'Royal Challengers Bangalore':
    prediction_array = prediction_array + [0,0,0,1,0]
  elif bowling_team == 'Sunrisers Hyderabad':
    prediction_array = prediction_array + [0,0,0,0,1]
  prediction_array = prediction_array + [runs, wickets, overs, runs_last_5, wickets_last_5]
  prediction_array = np.array([prediction_array])
  pred = model.predict(prediction_array)
  return int(round(pred[0]))