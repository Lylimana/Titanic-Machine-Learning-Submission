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

ages_survived = train_df.loc[train_df.Survived == 1]["Age"]

ages_range = [i for i in range(100)]

ages_survived_array = [age for age in ages_survived]
    
btyg.ScatterGraph(ages_survived_array, ages_survived_array)

survived = train_df.loc[train_df.Survived == 1]
not_survived = train_df.loc[train_df.Survived == 0]

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

print("Percentage of women who survived: {:.2f}%".format(rate_women*100))
print("Percentage of men who survived: {:.2f}%".format(rate_men*100))

# Data Cleaning

train_df_cleaned = train_df.drop(['Name', 'SibSp', 'Ticket', 'Cabin',  'Embarked'], axis='columns')

# Embarked and Cabin can potentially lead us to more accurate prediction but dropping for now to reduce complexity and dimensionality.

# Embarked feature could indicate that the stop prior to the disaster could of meant passengers would be taught health and safety guidelines and would have it fresh in memory. 
# Passengers who have already been on the cruise would of been taught on their embarking and could have forgotten health and safety procedure: 
# e.g. knowing key escape routes and where to evacuate in case of emergency. 

# Cabin feature could indicate where the passenger was staying during the disaster. If the passengers room was closer to deck and or escape routes,
# they have a higher likelihood of surviving. 
