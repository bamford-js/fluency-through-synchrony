# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:41:26 2021

@author: bamfo
"""

# Part 1
import pandas as pd
import json
from io import StringIO
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Import raw Qualtrics data
rawData = pd.read_csv('Study1a_data.csv')

dataFiltered = rawData[['jsPsych-data','Q10','Q9','Q7','Duration (in seconds)']].dropna().drop([0,1]).reset_index()
df = dataFiltered

# Import JSON data
c = 0
data = []
for i in range(len(df)):
    c = c+1
    x = df.loc[i,'jsPsych-data']
    try:
        y = json.loads(x)
        d = y['filedata']
        d = StringIO(d)
        d = pd.read_csv(d, sep=",")
        d[['ID']] = c
        d[['Gender']] = df.loc[i,'Q10']
        d[['Country']] = df.loc[i,'Q9']
        d[['Age']] = df.loc[i,'Q7']
        d[['Duration (s)']] = d.iloc[-1,3]/1000
        data.append(d)
        print(i)
    except:
        print('error in '+str(i))
for i in range(len(data)):
    if i == 0:
        df = data[i]
    else:
        df = pd.concat([df,data[i]])

# Filter
df = df.dropna(subset=['correct']).reset_index() #drop missing values from correct column
df = df[['ID','Gender','Country','Age','trial_index','test_part','direction','stim_type','rt','correct','Duration (s)']]
#df = df[df.test_part == 'test'].dropna(subset=['rt'])

# Add block numbers
df[['block']] = 0
exclusions = [] # 23,37,47,50,68
for i in df.index:
    if df.loc[i,'ID'] in exclusions:
        continue
    elif df.loc[i,'ID'] < 11: # pilot participants
        if df.loc[i,'trial_index'] > 13 and df.loc[i,'trial_index'] < 54:
            df.loc[i,'block'] = 1
        if df.loc[i,'trial_index'] > 54 and df.loc[i,'trial_index'] < 95:
            df.loc[i,'block'] = 2
        if df.loc[i,'trial_index'] > 95 and df.loc[i,'trial_index'] < 136:
            df.loc[i,'block'] = 3
    elif df.loc[i,'ID'] == 14: # pilot participant
        if df.loc[i,'trial_index'] > 13 and df.loc[i,'trial_index'] < 54:
            df.loc[i,'block'] = 1
        if df.loc[i,'trial_index'] > 54 and df.loc[i,'trial_index'] < 95:
            df.loc[i,'block'] = 2
        if df.loc[i,'trial_index'] > 95 and df.loc[i,'trial_index'] < 136:
            df.loc[i,'block'] = 3
    else:
        if df.loc[i,'trial_index'] > 13 and df.loc[i,'trial_index'] < 74:
            df.loc[i,'block'] = 1
        if df.loc[i,'trial_index'] > 74 and df.loc[i,'trial_index'] < 135:
            df.loc[i,'block'] = 2
        if df.loc[i,'trial_index'] > 135 and df.loc[i,'trial_index'] < 196:
            df.loc[i,'block'] = 3
    print(i)
df = df[df.block != 0]

dfCor = df[df.correct == True]

# Create summary table
summary = dfCor[['ID','block','stim_type','rt']].groupby(['ID','block','stim_type']).mean()
summary[['correct']] = df[['ID','block','stim_type','correct']].groupby(['ID','block','stim_type']).sum()
summary[['trialCount']] = df[['ID','block','stim_type','trial_index']].groupby(['ID','block','stim_type']).count()
summary = summary.reset_index()
summary[['Accuracy (%)']] = 0
summary[['Performance (%/ms)']] = 0
for i in range(len(summary)):
    summary.loc[i,'Accuracy (%)'] = (summary.loc[i,'correct']/summary.loc[i,'trialCount'])*100
    summary.loc[i,'Performance (%/ms)'] = (summary.loc[i,'Accuracy (%)']/summary.loc[i,'rt'])*100
summary = summary.rename(columns={'block':'Block','rt':'Reaction Time (ms)','stim_type':'Stimulus'})

#%% Identify outliers
# summary[['Z score']] = summary[(np.abs(stats.zscore(summary[['Performance (%/ms)']])) < 2.5).all(axis=1)]

summary[['Z score']] = np.abs(stats.zscore(summary[['Performance (%/ms)']]))
outliers = []

print('Perfomrance outlier')
for i in summary.index:
    if summary.loc[i,'Z score'] > 2.5:
        outliers.append(summary.loc[i,'ID'])
        print(summary.loc[i,'ID'])

print('Low accuracy')
for i in summary.index:
    if summary.loc[i,'Accuracy (%)'] < 50:
        outliers.append(summary.loc[i,'ID'])
        print(summary.loc[i,'ID'])

#%% Remove outliers
for i in summary.index:
    if summary.loc[i,'ID'] in outliers:
        summary.loc[i,'Performance (%/ms)'] = None

summary = summary.dropna(subset=['Performance (%/ms)'])

#%% Participant stats
descriptives = df[['ID','Gender','Country','Age','Duration (s)']].drop_duplicates(subset='ID').reset_index()
genders = descriptives.groupby(['Gender']).count()
countries = descriptives.groupby(['Country']).count()
age = descriptives['Age'].astype(float).mean()
age_SD = descriptives['Age'].astype(float).std()
age_range = [ descriptives['Age'].astype(float).min(), descriptives['Age'].astype(float).max() ]
#descriptives = descriptives.drop([18]) # extra long duration
durationMean = descriptives['Duration (s)'].astype(float).mean()
durationSD = descriptives['Duration (s)'].astype(float).std()

#%% Descriptive stats
desc_stim = summary.groupby(['Stimulus']).describe()

#%% Plot Accuracy and Reaction Time
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.stripplot(ax=axes[0], x='Stimulus', y='Reaction Time (ms)', data=summary, dodge=True, size=3, jitter=.25) 
sns.stripplot(ax=axes[1],x='Stimulus', y='Accuracy (%)', data=summary, dodge=True, size=3, jitter=.25)

#%% Plot Accuracy and Reaction Time with swarms and violins
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.stripplot(ax=axes[0], x='Stimulus', y='Reaction Time (ms)', data=summary, size=3, jitter=.25) 
sns.violinplot(ax=axes[0], x='Stimulus', y='Reaction Time (ms)', data=summary, color='lightgrey') 
sns.stripplot(ax=axes[1], x='Stimulus', y='Accuracy (%)', data=summary, size=3, jitter=.25)
sns.violinplot(ax=axes[1], x='Stimulus', y='Accuracy (%)', data=summary, color='lightgrey')

#%% Stats
target = 'Performance (%/ms)'

# Levene's test for homogeneity of varience
conditionLevene = pg.homoscedasticity(summary, dv=target, group='Stimulus')
blockLevene = pg.homoscedasticity(summary, dv=target, group='Block')

# Sphericity
spher, _, chisq, dof, pval = pg.sphericity(summary, dv=target, subject='ID', within='Block')

# ANOVA
aov = pg.rm_anova(dv=target, within=['Stimulus','Block'], subject='ID', data = summary)

# Posthoc contrasts
post_hoc = pg.pairwise_ttests(dv=target, within=['Block','Stimulus'], subject='ID', data=summary, padjust='bonf')

#%% Plot Performance
sns.pointplot(x='Block', y='Performance (%/ms)', hue='Stimulus', data=summary, capsize=.1)

#%% Correlation
x = summary['Reaction Time (ms)'].tolist()
y = summary['Accuracy (%)'].tolist()
corr = pg.corr(x,y, method='spearman')

#%% Plot Accuracy / RT correlation
sns.scatterplot(data=summary, x='Accuracy (%)', y='Reaction Time (ms)')

#%% Export
df.to_csv('Bamford_Study1a_longData.csv')
summary.to_csv('Bamford_Study1a_summarised.csv')

#%% Part 2
import pandas as pd
import json
from io import StringIO
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Import raw Qualtrics data
rawData = pd.read_csv('Study1b_data.csv')

#
df = rawData[['jsPsych-data','Q10','Q9','Q7','Q20','Q22']].dropna(subset=['jsPsych-data','Q10']).drop([0,1]).reset_index()

# Import JSON data
c = 0
data = []
for i in range(len(df)):
    c = c+1
    x = df.loc[i,'jsPsych-data']
    try:
        y = json.loads(x)
        d = y['filedata']
        d = StringIO(d)
        d = pd.read_csv(d, sep=",")
        d[['ID']] = c
        d[['Gender']] = df.loc[i,'Q10']
        d[['Country']] = df.loc[i,'Q9']
        d[['Age']] = df.loc[i,'Q7']
        d[['Hard?']] = df.loc[i,'Q20']
        d[['Comments?']] = df.loc[i,'Q22']
        d[['Duration (s)']] = d.iloc[-1,3]/1000
        data.append(d)
        print(i)
    except:
        print('error in '+str(i))
for i in range(len(data)):
    if i == 0:
        df = data[i]
    else:
        df = pd.concat([df,data[i]])

# Filter
df = df.dropna(subset=['correct']).reset_index() #drop missing values from correct column
df = df[['ID','Gender','Country','Age','trial_index','speed_condition','direction_condition','target_speed','target_direction','rt','correct','Hard?','Comments?','Duration (s)']]
#df = df.dropna(subset=['rt'])

# Add block numbers
df[['block']] = 0
exclusions = [72] # 
for i in df.index:
    if df.loc[i,'ID'] in exclusions:
        continue
    else:
        if df.loc[i,'trial_index'] > 21 and df.loc[i,'trial_index'] < 70:
            df.loc[i,'block'] = 1
        if df.loc[i,'trial_index'] > 70 and df.loc[i,'trial_index'] < 119:
            df.loc[i,'block'] = 2
        if df.loc[i,'trial_index'] > 119 and df.loc[i,'trial_index'] < 168:
            df.loc[i,'block'] = 3
        if df.loc[i,'trial_index'] > 168 and df.loc[i,'trial_index'] < 217:
            df.loc[i,'block'] = 4
        if df.loc[i,'trial_index'] > 217 and df.loc[i,'trial_index'] < 266:
            df.loc[i,'block'] = 5
    print(i)
df = df[df.block != 0]

dfCor = df[df.correct == True]
dfCor = dfCor.rename(columns={'block':'Block','rt':'Reaction Time (ms)','speed_condition':'Speed','direction_condition':'Direction','target_speed':'Target Speed'})
df = df.rename(columns={'block':'Block','rt':'Reaction Time (ms)','speed_condition':'Speed','direction_condition':'Direction','target_speed':'Target Speed'})

#%% Create summary table 
summary = dfCor[['ID','Block','Speed','Direction','Target Speed','Reaction Time (ms)']].groupby(['ID','Block','Target Speed','Speed','Direction']).mean()
summary[['correct']] = df[['ID','Block','Speed','Direction','Target Speed','correct']].groupby(['ID','Block','Target Speed','Speed','Direction']).sum()
summary[['trialCount']] = df[['ID','Block','Speed','Direction','Target Speed','trial_index']].groupby(['ID','Block','Target Speed','Speed','Direction']).count()
summary = summary.reset_index()
summary[['Accuracy (%)']] = 0
summary[['Performance (%/ms)']] = 0
for i in range(len(summary)):
    summary.loc[i,'Accuracy (%)'] = (summary.loc[i,'correct']/summary.loc[i,'trialCount'])*100
    summary.loc[i,'Performance (%/ms)'] = (summary.loc[i,'Accuracy (%)']/summary.loc[i,'Reaction Time (ms)'])*100

#%% Create summary table (without blocks, without target speed) - for descriptive plot (removed from paper)
summary = dfCor[['ID','Speed','Direction','Reaction Time (ms)']].groupby(['ID','Speed','Direction']).mean()
summary[['correct']] = df[['ID','Speed','Direction','correct']].groupby(['ID','Speed','Direction']).sum()
summary[['trialCount']] = df[['ID','Speed','Direction','trial_index']].groupby(['ID','Speed','Direction']).count()
summary = summary.reset_index()
summary[['Accuracy (%)']] = 0
summary[['Performance (%/ms)']] = 0
for i in range(len(summary)):
    summary.loc[i,'Accuracy (%)'] = (summary.loc[i,'correct']/summary.loc[i,'trialCount'])*100
    summary.loc[i,'Performance (%/ms)'] = (summary.loc[i,'Accuracy (%)']/summary.loc[i,'Reaction Time (ms)'])*100

#%% Participant stats
descriptives = df[['ID','Gender','Country','Age','Duration (s)']].drop_duplicates(subset='ID').reset_index()
genders = descriptives.groupby(['Gender']).count()
countries = descriptives.groupby(['Country']).count()
age = descriptives['Age'].astype(float).mean()
#descriptives = descriptives.drop([58,74]) # extra long duration
durationMean = descriptives['Duration (s)'].astype(float).mean()
durationSD = descriptives['Duration (s)'].astype(float).std()

#%% Descriptive Stats
desc_speed = summary[['Reaction Time (ms)','Accuracy (%)','Performance (%/ms)','Speed']].groupby(['Speed']).describe()
desc_direction = summary[['Reaction Time (ms)','Accuracy (%)','Performance (%/ms)','Direction']].groupby(['Direction']).describe()
desc_target = summary[['Reaction Time (ms)','Accuracy (%)','Performance (%/ms)','Target Speed']].groupby(['Target Speed']).describe()

#%% Stats
target = 'Performance (%/ms)'
dataset = summary

# Plot
ax = sns.pointplot(x='Speed', y=target, hue='Direction', data=dataset, capsize=.1)
ax.set(ylim=(16.25,17.5))

# ANOVA
aov = pg.rm_anova(dv=target, within=['Speed','Direction'], subject='ID', data = dataset, effsize='np2')

# Posthoc contrasts
post_hoc = pg.pairwise_ttests(dv=target, within=['Direction','Speed'], subject='ID', data=dataset, padjust='bonf')

# Levene's test for homogeneity of varience
spdLevene = pg.homoscedasticity(summary, dv=target, group='Speed')
dirLevene = pg.homoscedasticity(summary, dv=target, group='Direction')

# Sphericity
spher, _, chisq, dof, pval = pg.sphericity(dataset, dv=target, subject='ID', within=['Speed','Direction'])

#%% effect of speed?
target = 'Performance (%/ms)'
dataset = summary

sns.pointplot(x='Speed', y=target, hue='Target Speed', data=dataset, capsize=.1)
aov = pg.rm_anova(dv=target, within=['Speed','Target Speed'], subject='ID', data = dataset)
post_hoc = pg.pairwise_ttests(dv=target, within=['Target Speed','Speed'], subject='ID', data=dataset, padjust='bonf', effsize='np2')
spdLevene = pg.homoscedasticity(summary, dv=target, group='Speed')
tarLevene = pg.homoscedasticity(summary, dv=target, group='Target Speed')


#%% Plot
sns.set_theme(style='whitegrid')
dotSize = 3.5

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
sns.stripplot(ax=axes[0,0], x='Direction', y='Reaction Time (ms)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[0,0], x='Direction', y='Reaction Time (ms)', data=summary, color='lightgrey')

sns.stripplot(ax=axes[1,0], x='Direction', y='Accuracy (%)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[1,0], x='Direction', y='Accuracy (%)', data=summary, color='lightgrey')

sns.stripplot(ax=axes[0,1], x='Speed', y='Reaction Time (ms)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[0,1], x='Speed', y='Reaction Time (ms)', data=summary, color='lightgrey')

sns.stripplot(ax=axes[1,1], x='Speed', y='Accuracy (%)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[1,1], x='Speed', y='Accuracy (%)', data=summary, color='lightgrey')

sns.stripplot(ax=axes[2,0], x='Direction', y='Performance (%/ms)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[2,0], x='Direction', y='Performance (%/ms)', data=summary, color='lightgrey')

sns.stripplot(ax=axes[2,1], x='Speed', y='Performance (%/ms)', data=summary, dodge=True, size=dotSize, jitter=.25)
sns.violinplot(ax=axes[2,1], x='Speed', y='Performance (%/ms)', data=summary, color='lightgrey')

#%% Plot Accuracy / RT correlation
sns.scatterplot(data=summary, x='Accuracy (%)', y='Reaction Time (ms)')

#%% Correlation
x = summary['Reaction Time (ms)'].tolist()
y = summary['Accuracy (%)'].tolist()
corr = pg.corr(x,y, method='spearman')

#%% Export
df.to_csv('Bamford_Study1b_longData.csv')
summary.to_csv('Bamford_Study1b_summarised.csv')

#%% Identify outliers
# remove blocks with a <50 accuracy
outliers = []
for i in summary.index:
    if summary.loc[i,'Accuracy (%)'] < 50:
        outliers.append(i)
        print(summary.loc[i,'ID'])
        
# remove outliers?
summary = summary.drop(outliers)

#%% Identify outliers
# remove performance scores greater than 2.5 SDs
# summary[['Z score']] = summary[(np.abs(stats.zscore(summary[['Performance (%/ms)']])) < 2.5).all(axis=1)]

summary[['Z score']] = np.abs(stats.zscore(summary[['Performance (%/ms)']]))
outliers = []
for i in summary.index:
    if summary.loc[i,'Z score'] > 2.5:
        outliers.append(summary.loc[i,'ID'])
        print(summary.loc[i,'ID'])

for i in summary.index:
    if summary.loc[i,'ID'] in outliers:
        summary.loc[i,'Performance (%/ms)'] = None

# remove outliers?
summary = summary.dropna(subset=['Performance (%/ms)'])