import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import BeautifyGraph as bfyg

train = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\train.csv")
test = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\test.csv")

train_df = pd.DataFrame(train)
test_df = pd.DataFrame(test)

train_df.head(10)
test_df.head(10)

women = train_df.loc[train_df.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_df.loc[train_df.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("Percentage of women who survived: {:.2f}%".format(rate_women*100))
print("Percentage of men who survived: {:.2f}%".format(rate_men*100))

train_duplicates = train_df.duplicated() 

print(train_duplicates.sum())

train_df.shape

train_df.columns

# age_sorted = train_df.sort_values('Age')

ages_survived = train_df.loc[train_df.Survived == 1]["Age"]

ages_range = [i for i in range(100)]

ages_survived_array = [age for age in ages_survived]
    
print(ages_range)
print(ages_survived_array)
    
plt.scatter(ages_range, ages_survived_array)
plt.show()




