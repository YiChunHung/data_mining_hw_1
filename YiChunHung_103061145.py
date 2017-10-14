import os
import glob
import pandas as pd
import numpy as np
import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.utils
import plotly.plotly as py
import plotly.graph_objs as go
import math
import matplotlib.pyplot as plt
# matplotlib inline

# my functions
import helpers.data_mining_helpers as dmh
import helpers.text_analysis as ta

path = "./sentiment labelled sentences"
data_array = []
categories = []
for filepath in glob.glob(os.path.join(path, '*labelled.txt')):
    filename = os.path.splitext(filepath.split("/")[-1])[0]
    categories.append(filename)
    with open(filepath, "r") as f:
        for line in f:
            l = line.rstrip().split("\t")
            l.append(filename)
            data_array.append(l)

df = pd.DataFrame(data_array, columns=['text', 'label', 'category_name'])

# Shuffle the entire dataframe
df = sklearn.utils.shuffle(df).reset_index(drop=True)

# Format of df
#   text label from
# 1  a     1    c
# 2  b     0    a
# 3
# 4

# Print the length of df
print("Data length is {}".format(len(df)))

# Check for the missing value
print(df.isnull().apply(lambda x: dmh.check_missing_values(x)))

# Check for duplicated data
print("Data duplicates number : {}".format(sum(df.duplicated("text"))))
if sum(df.duplicated("text")) > 0:
    df.drop_duplicates(keep='first', inplace=True)

print("Data length(drop duplicates): {}".format(len(df)))

# Sample the data
df_sample = df.sample(frac=0.25)
print("length of sample df : {}".format(len(df_sample)))

# Visulize the sample data
df_category_counts = ta.get_tokens_and_frequency(list(df.category_name))
df_sample_category_counts = ta.get_tokens_and_frequency(
    list(df_sample.category_name))

# plt.subplot(211, title="Whole data")
df.category_name.value_counts().plot(kind="bar", rot=0)

# plt.subplot(212, title="Sample Data")
df_sample.category_name.value_counts().plot(kind="bar", rot=0)
# plt.show()

# Feature creation
df["unigrams"] = df["text"].apply(lambda x: dmh.tokenize_text(x))

# Feature subset selection
count_vect = CountVectorizer()
df_counts = count_vect.fit_transform(df.text)
analyze = count_vect.build_analyzer()
analyze(" ".join(list(df[4:5].text)))
# Todo : Visulize data.

# Dimensionality Reduction

from sklearn.decomposition import PCA
df_reduced = PCA(n_components=3).fit_transform(df_counts.toarray())

# Calculate the
trace1 = ta.get_trace(
    X_reduced, X["category_name"], "alt.atheism", "rgb(71,233,163)")
trace2 = ta.get_trace(
    X_reduced, X["category_name"], "soc.religion.christian", "rgb(52,133,252)")
trace3 = ta.get_trace(
    X_reduced, X["category_name"], "comp.graphics", "rgb(229,65,136)")
trace4 = ta.get_trace(
    X_reduced, X["category_name"], "sci.med", "rgb(99,151,68)")
