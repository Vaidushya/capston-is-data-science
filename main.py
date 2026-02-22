import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('penguin.csv')
sns.get_dataset_names()
df = sns.load_dataset('penguins')
df.head(10)
df.shape
df.tail()
df.isnull().sum()
df.describe().T
df.describe()
df.dtypes
df.info()
df.isnull().sum()
df.describe(include="all")
df.corr()
sns.heatmap(df.corr(), cmap="Wistia", annot=True)
df.hist(figsize=(12, 8))
plt.show()
df.plot(kind="box", subplots=True, layout=(3, 2), sharex=False, sharey=False, figsize=(8, 12))
plt.show()
df.sex.value_counts()
df.island.value_counts()
df.species.value_counts()
sns.countplot(data=df, x="sex", palette="summer")
sns.countplot(data=df, x="island", palette="RdPu")
sns.countplot(data=df, x="species", palette="YlOrBr")
sns.countplot(data=df, x="sex", palette="rocket", hue="species")
sns.countplot(data=df, x="island", palette="husl", hue="species")
sns.countplot(data=df, x="island", palette="spring", hue="sex")
sns.pairplot(data=df, hue="species", palette="mako")