import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BeautifyGraph as btyg

train = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\train.csv")
test = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\test.csv")

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

# Initial Data Exploration
train_df.head(10)
test_df.head(10)

train_duplicates = train_df.duplicated() 

print(train_duplicates.sum())

train_df.shape

train_df.columns

survived = train_df.loc[train_df.Survived == 1]
not_survived = train_df.loc[train_df.Survived == 0]

# Age & Fare Comparisson Between those who Died (R) and Survived (B)
plt.scatter(survived['Age'], survived['Fare'], c='blue', alpha=0.50)
plt.scatter(not_survived['Age'], not_survived['Fare'], c='red', alpha=0.50)
plt.title('Age & Fare Comparisson Between those who Died (R) and Survived (B)')
plt.xlabel('Age')
plt.ylabel('Fare Price')
plt.show()

women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

survived_class = survived['Pclass']
not_survived_class = not_survived['Pclass']
# Class Comparisson Between those who Survived
Passenger_classes = np.arange(1,4)

Passenger_classes_array_survived = []

for i in range(4): 
    if i == 0: 
        continue
    Passenger_classes_array_survived.append(survived.groupby('Pclass').size()[i])

fig, ax = plt.subplots()

ax.bar(Passenger_classes,Passenger_classes_array_survived, width=1, edgecolor="white", linewidth=0.7, color = 'blue')

ax.set(xlim=(0, 4), xticks=np.arange(1, 4),
       ylim=(0, 200), yticks=np.linspace(0, 400, 21)
       )
plt.title('Class Comparisson Between those who Survived in different classes')
plt.xlabel('Class')
plt.ylabel('Total')
plt.show()

# Class Comparisson Between those who Died 
Passenger_classes_array_not_survived = []

for i in range(4): 
    if i == 0: 
        continue
    Passenger_classes_array_not_survived.append(not_survived.groupby('Pclass').size()[i])
    
fig, ax = plt.subplots()

ax.bar(Passenger_classes,Passenger_classes_array_not_survived, width=1, edgecolor="white", linewidth=0.7, color = 'red')

ax.set(xlim=(0, 4), xticks=np.arange(1, 4),
       ylim=(0, 200), yticks=np.linspace(0, 400, 21)
       )
plt.title('Class Comparisson Between those who Died in different classes')
plt.xlabel('Class')
plt.ylabel('Total')
plt.show()

fig, ax = plt.subplots()


# Class Comparisson Between those who Died and survived visualised
ax.bar(Passenger_classes,Passenger_classes_array_not_survived, width=1, edgecolor="white", linewidth=0.7, color = 'red', alpha = 0.5)
ax.bar(Passenger_classes,Passenger_classes_array_survived, width=1, edgecolor="white", linewidth=0.7, color = 'blue', alpha = 0.5)

ax.set(xlim=(0, 4), xticks=np.arange(1, 4),
       ylim=(0, 200), yticks=np.linspace(0, 400, 21)
       )
plt.title('Class Comparisson Between those who Died (R) and Survived (B) in different classes')
plt.xlabel('Class')
plt.ylabel('Total')
plt.show()


# Survival Rate of men and women
print("Percentage of men who survived: {:.2f}%".format(rate_men*100))
print("Percentage of women who survived: {:.2f}%".format(rate_women*100))

# Data Cleaning
train_df_cleaned = train_df.drop(['PassengerId', 'Survived', 'Name', 'SibSp', 'Ticket', 'Cabin',  'Embarked'], axis='columns')
test_df_cleaned = test_df.drop(['PassengerId', 'Name', 'SibSp', 'Ticket', 'Cabin',  'Embarked'], axis='columns')

# Turning categorical data to numerical Data
train_df_cleaned['Sex'].replace(['female', 'male'],[1, 0], inplace=True) # Could have just done Label Encoding... 

    # Embarked and Cabin can potentially lead us to more accurate prediction but dropping for now to reduce complexity and dimensionality.

    # Embarked feature could indicate that the stop prior to the disaster could of meant passengers would be taught health and safety guidelines and would have it fresh in memory. 
    # Passengers who have already been on the cruise would of been taught on their embarking and could have forgotten health and safety procedure: 
    # e.g. knowing key escape routes and where to evacuate in case of emergency. 

    # Cabin feature could indicate where the passenger was staying during the disaster. If the passengers room was closer to deck and or escape routes,
    # they have a higher likelihood of surviving. 

# train_df_cleaned = train_df_cleaned['Fare'].round(2)
# test_df_cleaned = test_df_cleaned['Fare'].round(2)


# Train Test Split
from sklearn.model_selection import train_test_split

X = train_df_cleaned

y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 11, test_size = 0.3)

X_train.describe().round(2)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier

    # rf = RandomForestClassifier()

    # rf.fit(X_train, y_train)

    # y_pred = rf.predict(X_test)

    # rf.score(X_test, y_test)

# finding out more metrics on model i.e. accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# finding out which features had the most influence on the prediction 
features = pd.DataFrame(rf.feature_importances_, index = X.columns)

features.head()


# Random Forest 2 - Tuning Hyper Parameters
    # rf2 = RandomForestClassifier(
    #     n_estimators = 100, # Number of Trees in the forest
    #     max_features = None, # Sets limit on number of features used - in this case, there is no limit
    #     max_depth = 50, # Sets the depth of tree
    #     max_leaf_nodes = 50, # Sets the number of leaf nodes to branch out 
    #     min_samples_split = 2 # Sets the number of samples required to split internal node
    # )

    # rf2.fit(X_train, y_train)

    # y_pred = rf2.predict(X_test)

    # rf2.score(X_test, y_test)

# Random Forest 3- Finding optimum hyperparameters using GridSearchCV
from sklearn.model_selection import GridSearchCV

    # param_grid = {
    #     'n_estimators': [100, 500, 1000, 1500],
    #     'max_depth': [None, 10, 50],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'bootstrap': [True, False] # Random rows of data are selected with replacement to form training datasets for each tree
    # }

    # grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5 )
    # grid_search.fit(X_train, y_train)

    # print("Best Parameters: ", grid_search.best_params_)
    # print("Best Parameters: ", grid_search.best_estimator_)
    
    
    
    # From Gridsearch, these where the best parameters discovered: 
    # Best Parameters:  {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}
    # Best Parameters:  RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=500)
    
    

rf3 = RandomForestClassifier(
    bootstrap = True, 
    max_depth = 10, 
    min_samples_leaf = 1, 
    min_samples_split = 5,
    n_estimators = 500
)

rf3.fit(X_train, y_train)

y_pred = rf3.predict(X_test)

rf3.score(X_test, y_test)
