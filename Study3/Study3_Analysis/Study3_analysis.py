# -*- coding: utf-8 -*-
"""
-- Study 3 Analysis --


Created on Sun Apr 18 09:21:23 2021

@author: bamfo
"""

import pandas as pd 
import os
import seaborn as sns
import pingouin as pg
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# from statannotations.Annotator import Annotator

# set some parameters
directory = "C:/Users/bamfo/OneDrive/Documents/DPhil/Study 3/Analysis" #"E:/Study3"
os.chdir(directory)

setDiameter = 'diameter_3d'
trials = 6+1 # number of trials, plus practice

#%% Load from CSV
data = pd.read_csv('data_2022-05-30.csv')
quest = pd.read_csv('QualtricsData_processed.csv')

#%% Load filtered data from CSV
data1 = pd.read_csv('data_filtered_2022-05-30.csv')

#%% Set functions
# all required functions are here, to be called on later

def analysis(df):
    df = df.rename(columns={'Beat Mean':'IOI Mean (s)','Beat Variance':'IOI Variance','Pupil Size':'Pupil Size (mm)'})
    df['Tempo Ratio'] = df['Tempo Ratio'].replace({'120-120':'Synchrony','120-080':'Complementary','120-113':'Non-synchrony'})
    analyses = {'Difficulty':0,'Bonding':0,'Pupil Size (mm)':0,'LHIPA':0,'IOI Variance':0,'IOI Mean (s)':0}
    pairs = [('Synchrony','Complementary'),('Synchrony','Non-synchrony'),('Complementary','Non-synchrony')]
    order = ['Synchrony','Complementary','Non-synchrony']
    sns.set_theme(style='whitegrid')
    dotSize = 3.5
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axisLoc = [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]
    j=0
    for i in analyses:
        a,b = axisLoc[j]
        sns.violinplot(ax=axes[a,b], x='Tempo Ratio', y=i, data=df, cut=0, order=order, color='lightgrey')
        sns.stripplot(ax=axes[a,b], x='Tempo Ratio', y=i, data=df, dodge=True, size=dotSize, jitter=.25, order=order)
        j=j+1
        #annotator = Annotator(ax,pairs,data=df,x='Tempo Ratio',y=i, order=order)
        #annotator.configure(test='t-test_paired',comparisons_correction='bonferroni')
        #annotator.apply_and_annotate()
              
        analyses[i] = pg.rm_anova(dv=i, within='Tempo Ratio', subject='Participant', data=df, correction=True)
    plt.show()  
    return pd.concat(analyses)

#%% Identify outliers
# use this for playing around with identifying outliers of different thresholds
pupilOutliers = data[(np.abs(stats.zscore(data['Pupil Size'])) >= 2.5)] 
beatOutliers = data[(np.abs(stats.zscore(data['Beat Variance'])) >= 2.5)]
likeOutliers = data[(np.abs(stats.zscore(data['Liking'])) >= 2.5)]
lengthOutliers = data[(data['Trial Length'] <= 4000)]
IPAOutliers = data[(np.abs(stats.zscore(data['LHIPA'])) >= 2.5)]
connOutliers = data[(np.abs(stats.zscore(data['Connection'])) >= 2.5)]
diffOutliers = data[(np.abs(stats.zscore(data['Difficulty'])) >= 2.5)]

outliers = pd.concat([pupilOutliers, beatOutliers, likeOutliers, lengthOutliers, IPAOutliers, connOutliers, diffOutliers])

#%% Filter
# remove trials shorter than 4000 frames. Implies a lot of excluded frames
# remove statistical outliers on all variables
# particularly rediculous pupil sizes
# remove outliers on beat, but not missing recordings
data1 = data
#data1 = data1[(data1['Recording Length'] > 1000000)].reset_index().drop(['index'],axis=1)
data1 = data1[(data1['Trial Length'] > 4000)]
#%% 
# optional outliers. Not done in paper.


data1 = data1[(np.abs(stats.zscore(data1['Pupil Size'])) < 2.5)]
data1 = data1[(np.abs(stats.zscore(data1['LHIPA'])) < 2.5)]
data1 = data1[(np.abs(stats.zscore(data1['Liking'])) < 2.5)]
data1 = data1[(np.abs(stats.zscore(data1['IOS'])) < 2.5)]
data1 = data1[(np.abs(stats.zscore(data1['Connection'])) < 2.5)]
data1 = data1[(np.abs(stats.zscore(data1['Difficulty'])) < 2.5)]
beatOutliers = (np.abs(stats.zscore(data1['Beat Variance'].dropna())) < 2.5)
for i in beatOutliers.index:
    if beatOutliers.loc[i] == False:
        data1 = data1.drop(i, axis=0)
#exclusions = [1,17,40,41,43,44,45,46,47,48,49,50,51,52,53,54,57,73,75,76,78,15,20,62,63,60,3,22,23,38,39,14,24]
#data1 = data1.loc[~data1['Participant'].isin(exclusions)]

#%% Filter Beat Mean
for i in range(len(data)):
    if data.loc[i,'Beat Mean'] == 0:
        data.loc[i,'Beat Mean'] = None 

#%% Export filtered data
directory = "E:/Study3"
os.chdir(directory)
#%%
data1.to_csv('data_filtered_2022-05-30.csv')

#%% RM ANOVAs
results1 = analysis(data1)


#%% full post-hoc tests (including drums)

posthocs = pg.pairwise_ttests(dv='Difficulty', within='Tempo Ratio', subject='Participant',
                             padjust='bonf', data=data1) #between='Drums'
# maybe try some raincloud plots?

#%% Participant description

partStats = quest.describe()
print(quest['Sex'].value_counts())
print(quest['Handedness'].value_counts())


#%% Export ANOVA results
results1.to_csv('Study3_analysis.csv')

#%% Correlation matrix
sns.pairplot(data[['Liking','Connection','IOS','Difficulty','Pupil Size']])

#%% Correlation between Bonding and Difficulty
sns.scatterplot(data['Bonding'], data['Difficulty'])
print(pg.corr(data['Bonding'], data['Difficulty']))

#%% Correlation between Pupil Size and Difficulty
sns.scatterplot(data['Pupil Size'], data['Difficulty'])
print(pg.corr(data['Pupil Size'], data['Difficulty']))

#%% Correlation between Perceptual Ability and Difficulty by condition
# could be interesting to investigate in the paper
df = data1
df = df.rename(columns={'Beat Mean':'IOI Mean (s)','Beat Variance':'IOI Variance','Pupil Size':'Pupil Size (mm)'})
df['Tempo Ratio'] = df['Tempo Ratio'].replace({'120-120':'Synchrony','120-080':'Complementary','120-113':'Non-synchrony'})

sns.lmplot(x='Perceptual Ability', y='Difficulty', hue='Tempo Ratio',data=df, scatter_kws={"s": 5})
print(pg.corr(df['Perceptual Ability'], df['Difficulty']))

#%% line of best fit
sns.lmplot(x='Perceptual Ability', y='Bonding',data=data1, scatter_kws={"s": 5})
print(pg.corr(data1['Perceptual Ability'], data1['Bonding']))

#%% Cronbach's Alpha
# for the bonding scale
print(pg.cronbach_alpha(data1[['Liking','Connection','IOS']]))



#%% Regression plot

ax = sns.lmplot(x='Pupil Size', y='Bonding', hue='Difficulty',data=data1, scatter_kws={"s": 5})

#%% 
df = data1
sns.set_theme(style='whitegrid')
df['Tempo Ratio'] = df['Tempo Ratio'].replace({'120-120':'Synchrony','120-080':'Complementary','120-113':'Non-synchrony'})
df['Drums'] = df['Drums'].replace({'same':'Matched','diff':'Unmatched'})
sns.violinplot(x='Tempo Ratio', y='Bonding', hue='Drums',data=df, color='lightgrey')
sns.stripplot(x='Tempo Ratio', y='Bonding', hue='Drums',data=df)
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0)
