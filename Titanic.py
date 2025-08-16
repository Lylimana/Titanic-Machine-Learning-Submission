import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\train.csv")
test = pd.read_csv(r"C:\Users\manal\Desktop\Titanic ML Submission\titanic\test.csv")

train_df = pd.DataFrame(train)
