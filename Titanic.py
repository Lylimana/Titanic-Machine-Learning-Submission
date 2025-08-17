import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

train_duplicates.sum()
