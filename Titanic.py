import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BeautifyGraph as bfyg

train = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\train.csv")
test = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\test.csv")

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

# Data Exploration
train_df.head(10)
test_df.head(10)

train_duplicates = train_df.duplicated() 

print(train_duplicates.sum())

train_df.shape

train_df.columns

# ages_survived = train_df.loc[train_df.Survived == 1]["Age"]

# ages_range = [i for i in range(100)]

# ages_survived_array = [age for age in ages_survived]
    
# plt.scatter(ages_survived_array, ages_survived_array)
# plt.show()

survived = train_df.loc[train_df.Survived == 1]
not_survived = train_df.loc[train_df.Survived == 0]

plt.scatter(survived['Age'], survived['Fare'], c='red', alpha=0.50)
plt.scatter(not_survived['Age'], not_survived['Fare'], c='blue', alpha=0.50)
plt.show()

women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("Percentage of women who survived: {:.2f}%".format(rate_women*100))
print("Percentage of men who survived: {:.2f}%".format(rate_men*100))



