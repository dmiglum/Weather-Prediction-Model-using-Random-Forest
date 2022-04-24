#Random Forest Example

''' Machine Learning Creating a Model Steps
1) State the question and determine required data
2) Acquire the data in an accessible format
3) Identify and correct missing data points/anomalies as required
4) Prepare the data for the machine learning model
5) Establish a baseline model that you aim to exceed
6) Train the model on the training data
7) Make predictions on the test data
8) Compare predictions to the known test set targets and calculate performance metrics
9) If performance is not satisfactory, adjust the model, acquire more data, or try a different modeling technique
10) Interpret model and report results visually and numerically
'''
#about 80% of the time spent in data analysis is cleaning and retrieving data
import pandas as pd

features = pd.read_csv("temps.csv")

#checking for outliers
features.describe() #all variables look reasonable

#Use datetime for dealing with dates
import datetime

#get years, months, and days
years= features['year']
months = features['month']
days = features['day']

#list dates as string and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

import matplotlib.pyplot as plt
%matplotlib inline
# Set up the plotting layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
fig.autofmt_xdate(rotation = 45)

# Actual max temperature measurement
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

# Temperature from 1 day ago
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

# Temperature from 2 days ago
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

# Friend Estimate
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)

### One-hot encoding days of the week
features = pd.get_dummies(features)
features.iloc[1:4, 5:12] #testing rows and columns

#Converting data to numpy arrays and separating the target (known as label) from features
# Use numpy to convert to arrays
import numpy as np
# Labels are the values we want to predict
labels = np.array(features['actual'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

### Splitting data into training and testing sets
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets - we use 75% of data for training and 25% for testing
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
########################################
'''
It looks as if everything is in order! Just to recap, to get the data
into a form acceptable for machine learning we:
1) One-hot encoded categorical variables
2) Split data into features and labels
3) Converted to arrays
4) Split data into training and testing sets
    
Depending on the initial data set, there may be extra work involved such as
removing outliers, imputing missing values, or converting temporal variables into cyclical representations. 
These steps may seem arbitrary at first, but once you get the basic workflow, 
it will be generally the same for any machine learning problem. It’s all about taking 
human-readable data and putting it into a form that can be understood by a machine learning model.
'''
### Establish a baseline -> our baseline are historical averages for the day
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))
#We need to beat an error of about 5 degrees

###### Train model
'''
After all the work of data preparation, creating and training the model is pretty
simple using Scikit-learn. We import the random forest regression model from
skicit-learn, instantiate the model, and fit (scikit-learn’s name for training) 
the model on the training data. (Again setting the random state for reproducible results). 
'''
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Make Predictions on the Test Set
'''
Our model has now been trained to learn the relationships between the features
and the targets. The next step is figuring out how good the model is! To do this
we make predictions on the test features (the model is never allowed to see the
test answers). We then compare the predictions to the known answers. When performing
regression, we need to make sure to use the absolute error because we expect some of 
our answers to be low and some to be high.
'''
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#Our average estimate is off by 3.83, which is more than 1 degree better than a baseline

# Performance metrics - Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%')

# Improving the model
'''
In the usual machine learning workflow, this would be when start hyperparameter tuning. 
This is a complicated phrase that means “adjust the settings to improve performance” (The 
settings are known as hyperparameters to distinguish them from model parameters learned 
during training). The most common way to do this is simply make a bunch of models with 
different settings, evaluate them all on the same validation set, and see which one does 
best. Of course, this would be a tedious process to do by hand, and there are automated 
methods to do this process in Skicit-learn. 
'''

# Interpret Model and Report Results
'''
At this point, we know our model is good, but it’s pretty much a black box. 
We feed in some Numpy arrays for training, ask it to make a prediction, evaluate 
the predictions, and see that they are reasonable. The question is: how does this 
model arrive at the values? There are two approaches to get under the hood of the 
random forest: first, we can look at a single tree in the forest, and second, we 
can look at the feature importances of our explanatory variables.
'''
# Visualizing a single tree
# Import tools needed for visualization
''' 
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
test_tree = rf.estimators_[5] #taking a single tree
# Export the image to a dot file
export_graphviz(test_tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree.png') #doesn't work because of path dependency issues between pydot and graphviz
'''

#Alternative way to create an image
from sklearn import tree
test_tree = rf.estimators_[5] #taking a single tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(test_tree, feature_names = feature_list, rounded = True, precision = 1)
fig.savefig('tree.png')
# Tree from above is expansive with 15 layers, so we create a example tree with 3 only layers to make it easier to see

# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state = 42)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(tree_small, feature_names = feature_list, rounded = True, precision = 1)
fig.savefig('small_tree.png')


''' Observations from our small tree (how to interpret this particular tree)
Based solely on this tree, we can make a prediction for any new data point. Let’s take an example of making a prediction for Wednesday, December 27, 2017. The (actual) variables are: temp_2 = 39, temp_1 = 35, average = 44, and friend = 30. We start at the root node and the first answer is True because temp_1 ≤ 59.5. We move to the left and encounter the second question, which is also True as average ≤ 46.8. Move down to the left and on to the third and final question which is True as well because temp_1 ≤ 44.5. Therefore, we conclude that our estimate for the maximum temperature is 41.0 degrees as indicated by the value in the leaf node. An interesting observation is that in the root node, there are only 162 samples despite there being 261 training data points. This is because each tree in the forest is trained on a random subset of the data points with replacement (called bagging, short for bootstrap aggregating). (We can turn off the sampling with replacement and use all the data points by setting bootstrap = False when making the forest). Random sampling of data points, combined with random sampling of a subset of the features at each node of the tree, is why the model is called a ‘random’ forest.
Furthermore, notice that in our tree, there are only 2 variables we actually used to make a prediction! According to this particular decision tree, the rest of the features are not important for making a prediction. Month of the year, day of the month, and our friend’s prediction are utterly useless for predicting the maximum temperature tomorrow! The only important information according to our simple tree is the temperature 1 day prior and the historical average. Visualizing the tree has increased our domain knowledge of the problem, and we now know what data to look for if we are asked to make a prediction!
'''

### Variable importance
'''
In order to quantify the usefulness of all the variables in the entire random forest, 
we can look at the relative importances of the variables. The importances returned in 
Skicit-learn represent how much including a particular variable improves the prediction. 
The actual calculation of the importance is beyond the scope of this post, but we can 
use the numbers to make relative comparisons between variables.
'''

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
'''
At the top of the list is temp_1, the max temperature of the day before. This tells 
us the best predictor of the max temperature for a day is the max temperature of the 
day before, a rather intuitive finding. The second most important factor is the 
historical average max temperature, also not that surprising. Your friend turns out to 
not be very helpful, along with the day of the week, the year, the month, and the temperature 2 days prior.
'''

# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators= 1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('temp_1'), feature_list.index('average')]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, train_labels)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
#Results are very similar tho those with more variables, which tells us that
#we don't need other extra variables
'''
Knowing how to find the right balance between performance and cost is an essential
skill for a machine learning engineer and will ultimately depend on the problem!
At this point we have covered pretty much everything there is to know for a basic 
implementation of the random forest for a supervised regression problem. We can 
feel confident that our model can predict the maximum temperature tomorrow with 
94% accuracy from one year of historical data.
'''

# Visualizations
#Plotting importances
import matplotlib.pyplot as plt
%matplotlib inline
# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

# Plotting dataset with predictions highlighted
# Use datetime for creating date objects for plotting
import datetime
# Dates of training values
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})
# Dates of predictions
months = test_features[:, feature_list.index('month')]
days = test_features[:, feature_list.index('day')]
years = test_features[:, feature_list.index('year')]
# Column of dates
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions})
# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
'''
To further diagnose the model, we can plot residuals (the errors) to see if our model has a tendency 
to over-predict or under-predict, and we can also see if the residuals are normally distributed.
'''
# Final chart showing the actual values, the temperature one day previous, 
# the historical average, and our friend’s prediction
# Make the data accessible for plotting
true_data['temp_1'] = features[:, feature_list.index('temp_1')]
true_data['average'] = features[:, feature_list.index('average')]
true_data['friend'] = features[:, feature_list.index('friend')]
# Plot all the data as lines
plt.plot(true_data['date'], true_data['actual'], 'b-', label  = 'actual', alpha = 1.0)
plt.plot(true_data['date'], true_data['temp_1'], 'y-', label  = 'temp_1', alpha = 1.0)
plt.plot(true_data['date'], true_data['average'], 'k-', label = 'average', alpha = 0.8)
plt.plot(true_data['date'], true_data['friend'], 'r-', label = 'friend', alpha = 0.3)
# Formatting plot
plt.legend(); plt.xticks(rotation = '60');
# Lables and title
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual Max Temp and Variables');
'''
At this point, if we want to improve our model, we could try different hyperparameters (settings) 
try a different algorithm, or the best approach of all, gather more data! The performance of any 
model is directly proportional to the amount of valid data it can learn from, and we were using 
a very limited amount of information for training.
'''