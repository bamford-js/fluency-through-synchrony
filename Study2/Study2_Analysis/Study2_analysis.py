# -*- coding: utf-8 -*-
"""
 -- Study 2 Data Analysis --

Statistical tests for Study 2

Created on Sun Apr 18 09:21:23 2021
@author: bamfo
"""

import pandas as pd 
import os
import seaborn as sns
import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#%% Load data from CSV
directory = "C:/Users/bamfo/OneDrive/Documents/DPhil/Study 2/Analysis"
os.chdir(directory)
data = pd.read_csv('data_2022-04-18.csv')
# quest = pd.read_csv('QualtricsData_processed.csv')

data = data.rename(columns={'Beat Variance':'IOI Variance','Beat Mean':'IOI Mean (s)'})

#%% Filter
# apply this to remove outliers

# remove trials shorter than 1000 frames. Implies a lot of excluded frames
# remove statistical outliers on all variables
# particularly outlandish pupil sizes
# remove outliers on beat, but not missing recordings

data1 = data
data1 = data1[(np.abs(stats.zscore(data1['Performance (%/ms)'])) < 2.5)]
beatOutliers = (np.abs(stats.zscore(data1['IOI Variance'].dropna())) < 2.5)
for i in beatOutliers.index:
    if beatOutliers.loc[i] == False:
        data1 = data1.drop(i, axis=0)
beatMOutliers = (np.abs(stats.zscore(data1['IOI Mean (s)'].dropna())) < 2.5)
for i in beatMOutliers.index:
    if beatMOutliers.loc[i] == False:
        data1 = data1.drop(i, axis=0)
pupilOutliers = (np.abs(stats.zscore(data1['Pupil Size (mm)'].dropna())) < 2.5)
for i in pupilOutliers.index:
    if pupilOutliers.loc[i] == False:
        data1 = data1.drop(i, axis=0)
ipaOutliers = (np.abs(stats.zscore(data1['LHIPA'].dropna())) < 2.5)
for i in ipaOutliers.index:
    if ipaOutliers.loc[i] == False:
        data1 = data1.drop(i, axis=0)

data1 = data1[(data1['Recording Length'] > 1000000)]

#%% Export filtered data
directory = "C:/Users/bamfo/OneDrive/Documents/DPhil/Study 2/Analysis"
os.chdir(directory)
data1.to_csv('data_filtered_2022-04-20.csv')

#%% rename conditions
data = data.replace({'sync':'Synchrony','async':'Non-synchrony','single':'Control'})
data = data.rename(columns={'Performance':'Performance (%/ms)','Pupil Size':'Pupil Size (mm)'})

#%% Describe sample
descriptives = data[['Participant','Sex','Age','Handedness','Perceptual Ability']].drop_duplicates(subset='Participant').reset_index()
genders = descriptives.groupby(['Sex']).count()
hands = descriptives.groupby(['Handedness']).count()
age = descriptives['Age'].astype(float).mean()
age_SD = descriptives['Age'].astype(float).std()
age_range = [ descriptives['Age'].astype(float).min(), descriptives['Age'].astype(float).max() ]
percAbil = descriptives['Perceptual Ability'].astype(float).mean()
percAbilSD = descriptives['Perceptual Ability'].astype(float).std()
n = len(descriptives)

#%%
print(pg.corr(quest['Participant'],quest['Openness']))
print(pg.corr(quest['Participant'],quest['Conscientiousness']))
print(pg.corr(quest['Participant'],quest['Extraversion']))
print(pg.corr(quest['Participant'],quest['Agreeableness']))
print(pg.corr(quest['Participant'],quest['Neuroticism']))
sns.regplot(x="Participant", y="Extraversion", data=quest);

#%% Regression plot
ax = sns.lmplot(x='IOI Variance', y='Performance (%/ms)', hue='Condition',data=data1, scatter_kws={"s": 5})


#%% Parametric analysis
analyses = {'Performance (%/ms)':0,'Pupil Size (mm)':0,'LHIPA':0,'Fixations':0,'IOI Variance':0,'IOI Mean (s)':0}
desc = {'Performance (%/ms)':0,'Pupil Size (mm)':0,'LHIPA':0,'Fixations':0,'IOI Variance':0,'IOI Mean (s)':0}

# Descriptive stats
for i in desc:
    desc[i] = data.groupby(['Condition']).describe()[i]
desc = pd.concat(desc)

pairs = [('Non-synchrony','Control'),('Synchrony','Non-synchrony'),('Control','Synchrony')]
order = ['Non-synchrony','Control','Synchrony']


sns.set_theme(style='whitegrid')
dotSize = 3.5
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axisLoc = [[0,0],[1,0],[0,1],[1,1],[2,0],[2,1]]
j=0
for i in analyses:
    # if i != 'Performance':
    #     data2 = data.dropna(subset=['Fixations'])
    # else:
    #     data2 = data
    # data2 = data1[(np.abs(stats.zscore(data1[[i]])) < 2.5).all(axis=1)].reset_index()
    # ax = sns.catplot(x='Condition', y=i, kind='violin', data=data1)
    a,b = axisLoc[j]
    sns.violinplot(ax=axes[a,b], x='Condition', y=i, data=data1, cut=0, order=order, color='lightgrey')
    sns.stripplot(ax=axes[a,b], x='Condition', y=i, data=data1, dodge=True, size=dotSize, jitter=.25, order=order)
    j=j+1
    #annotator = Annotator(ax,pairs,data=data1,x='Condition',y=i, order=order, loc='outside')
    #annotator.configure(test='Mann-Whitney',comparisons_correction='bonferroni')
    #annotator.apply_and_annotate()
    #plt.show()  
    
    analyses[i] = pg.rm_anova(dv=i, within='Condition', subject='Participant', data=data, correction=True)
analyses = pd.concat(analyses)
plt.show()  

#%% Export analysis
desc.to_csv('Study2_descriptives.csv')
analyses.to_csv('Study2_analyses.csv')

#%% 
post_hocs = pg.pairwise_ttests(dv='Beat Mean',within='Condition',subject='Participant',data=data1, effsize='cohen', padjust='bonf',alternative='two-sided')

#%% 
post_hocs = pg.pairwise_ttests(dv='Pupil Size (mm)',within='Condition',subject='Participant',data=data, effsize='cohen', padjust='bonf',alternative='two-sided')

#%%
post_hocs = pg.pairwise_ttests(dv='Performance (%/ms)',within='Condition',subject='Participant',data=data, effsize='cohen', padjust='bonf')

#%%
post_hocs = pg.pairwise_ttests(dv='Fixations',within='Condition',subject='Participant',data=data, effsize='cohen', padjust='bonf')
    
#%% Non-parametric analysis
analyses = {'Pupil Size':0,'LHIPA':0,'Fixations':0,'Performance':0}
data2 = data
for i in analyses:
    if i != 'Performance':
        data2 = data.dropna(subset=['Fixations'])
    # data2 = data1[(np.abs(stats.zscore(data1[[i]])) < 2).all(axis=1)].reset_index()
    ax = sns.catplot(x='Condition', y=i, kind='violin', split=True, data=data2)
    analyses[i] = pg.friedman(dv=i, within='Condition', subject='Participant', data=data2)
analyses = pd.concat(analyses)

#%% Effect of perceptual ability??
# copied from study 3 needs adjusting
sns.lmplot(x='Perceptual Ability', y='Bonding', hue='Tempo Ratio',data=data1, scatter_kws={"s": 5})
print(pg.corr(data1['Perceptual Ability'], data1['Difficulty']))