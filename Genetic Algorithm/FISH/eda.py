#%%
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from yellowbrick.classifier import ConfusionMatrix

#%%
df_fish = pd.read_csv(r'Fish.csv')
# df_fish.describe()
# ? 0 weight is found on the dataset, remove invalid record.
df_fish = df_fish[df_fish['Weight'] != 0]
# df_fish.describe()

# %%
sns.countplot(data=df_fish, x='Species')

#%%
# Distribution
sns.pairplot(df_fish, hue='Species')

# Correlation with diff parameters.
# option 1
sns.heatmap(df_fish.corr(), annot=True)

# option 2
corr = df_fish.corr()
corr.style.background_gradient(cmap='coolwarm')


#%%
# To resign the plot into 1 plot
fig, ([ax1, ax2], [ax3, ax4], [ax5, ax6]) = plt.subplots(3, 2, figsize=(20, 20))
sns.boxplot(x="Species", y="Weight", data=df_fish, ax=ax1)
sns.boxplot(x="Species", y="Length1", data=df_fish, ax=ax2)
sns.boxplot(x="Species", y="Length2", data=df_fish, ax=ax3)
sns.boxplot(x="Species", y="Length3", data=df_fish, ax=ax4)
sns.boxplot(x="Species", y="Height", data=df_fish, ax=ax5)
sns.boxplot(x="Species", y="Width", data=df_fish, ax=ax6)
ax1.tick_params(labelrotation=0, labelsize=20)
ax2.tick_params(labelrotation=0, labelsize=20)
ax3.tick_params(labelrotation=0, labelsize=20)
ax4.tick_params(labelrotation=0, labelsize=20)
ax5.tick_params(labelrotation=0, labelsize=20)
ax6.tick_params(labelrotation=0, labelsize=20)
fig.tight_layout()

#%%

#%%

df_fish_dummies = pd.get_dummies(df_fish, columns=['Species'])
x = df_fish_dummies.iloc[:, 1:13].values
y = df_fish_dummies["Weight"].values

#%%
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

x = df_fish.iloc[:, 1:]
y = df_fish.iloc[:, 0:1]

for train_index, test_index in sss.split(df_fish, df_fish["Species"]):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# %%

rfc = RandomForestClassifier(n_estimators=1000).fit(x_train, y_train)
confusion_matrix(y_true=y_test, y_pred=rfc.predict(x_test))

# %%
cm = ConfusionMatrix(rfc)
cm.score(x_test, y_test)

#%%
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values[6], x_test)

# %%
shap.initjs()
i = 31
j = 0
shap.force_plot(explainer.expected_value[j], shap_values[j][i], x_test.values[i], feature_names=x_test.columns)


#%%
df_fish.iloc[25]
x_test.values[25]

#%%
explainer = shap.Explainer(rfc, x_test)
shap_values = explainer(x_test)
shap.waterfall_plot(shap_values[0])

# %%
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(x_test)
expected_value = explainer.expected_value

i = 0
shap.decision_plot(expected_value[i], shap_values[i][0], x_test.columns)

# %%
shap.waterfall_plot(shap_values[0])
