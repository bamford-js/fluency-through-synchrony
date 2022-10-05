# -*- coding: utf-8 -*-
"""
 -- Study 3 Analysis --

Pre-processing for pupil data
Matching with PsychoPy data
Now combines across participants!

Created on Sun Apr 18 09:21:23 2021
@author: Joshua Bamford
"""

import pandas as pd 
import os
import seaborn as sns
import glob
import pingouin as pg
import math
import pywt
import numpy as np
# import librosa
import statistics
from scipy import stats
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

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

def importPsychoPy():
    fileList = []
    os.chdir(f'{directory}/data')
    print('loading PsychoPy data...')
    for file in glob.glob('*.csv'):
        print(file)
        fileList.append(file)
        p = int(file[0:3])
        df = pd.read_csv(file)
        df['Trial'] = df['conditions.thisTrialN'] + 1
        df['Participant'] = p
        df['Tempo Ratio'] = df['within_condition']
        df['Drums'] = df['between_condition']
        df['Liking'] = (df['slider_liking_L.response'] - 1) * 100
        df['Connection'] = (df['slider_connection_L.response'] - 1) * 100
        df['IOS'] = (df['slider_IOS_L.response'] - 1) * 100
        df['Difficulty'] = (df['slider_difficulty.response'] - 1) * 100
        df = df[[
            'Participant',
            'Trial',
            'Drums',
            'Tempo Ratio',
            'pair',
            'file_name',
            'Liking',
            'Connection',
            'IOS',
            'Difficulty'
            ]]
        if p == 1:
            data = df
        else:
            data = pd.concat([data,df])
    data = data.dropna().reset_index().drop(['index'], axis=1)
    print('PsychoPy data loaded.')
    print()
    return data

# Import Pupil data
def blinkless(d, blinks):
    blinkNum = len(blinks)
    for i in range(blinkNum):
        blink_start = blinks.loc[i, 'start_frame_index']
        blink_end = blinks.loc[i, 'end_frame_index'] + 1 # must include last frame
        blink = list(range(blink_start, blink_end))
        d.loc[d['world_index'].isin(blink), setDiameter] = None # drop if blink detected
        d.loc[d[setDiameter] == 0, setDiameter] = None # drop if diameter = 0
    d = d.dropna() # remove all null values, including blinks
    return d

def labelCondition(d, annotations):
    d['trial'] = 0
    for i in range(len(annotations)):
        trial = annotations.loc[i, 'trial']
        if annotations.loc[i, 'label'] == 'condition_start':
            conditionStart = annotations.loc[i, 'timestamp']
            conditionEnd = annotations.loc[i+1, 'timestamp']
            conditionRange = d[d['pupil_timestamp'].between(conditionStart, conditionEnd)].index
            print(f'    starting trial {trial}')
            for j in conditionRange:
                d.loc[j, 'trial'] = trial
        else:
            print(f'    end of trial {trial}')
    return d

def labelBlinkCondition(d, anno):
    d['trial'] = 0
    for i in range(len(anno)):
        trial = anno.loc[i, 'trial']
        if anno.loc[i, 'label'] == 'condition_start':
            conditionStart = anno.loc[i, 'timestamp']
            conditionEnd = anno.loc[i+1, 'timestamp']
            conditionRange = d[d['start_timestamp'].between(conditionStart, conditionEnd)].index
           # print(f'starting trial {trial}')
            for j in conditionRange:
                d.loc[j, 'trial'] = trial
       # else:
           # print(f'end of trial {trial}')
    return d

def blinkNumber(d):
    totalBlinks = {}
    for i in range(trials):
        theseBlinks = d.loc[d['trial']==i]
        totalBlinks[i] = len(theseBlinks)
    return totalBlinks

def sizeByTrial(d):
    trialMean = {}
    trialLen = {}
    for i in range(trials):
        x = d.loc[d['trial'] == i]
        x = x[[setDiameter]].values
        trialLen[i] = len(x) # number of frames of good data
        trialMean[i] = float(sum(x)/len(x))
    return trialMean, trialLen

# Run LHIPA
def modmax(d):
    # compute signal modulus
    m = [0.0]*len(d)
    for i in list(range(len(d))):
        m[i] = math.fabs(d[i])
    # if value is larger than both neighbours, and strictly larger than either, then it is a local maximum
    t = [0.0]*len(d)
    for i in list(range(len(d))):
        ll = m[i-1] if i >= 1 else m[i]
        oo = m[i]
        rr = m[i+1] if i < len(d)-2 else m[i]
        if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
            # compute magnitude
            t[i] = math.sqrt(d[i]**2)
        else:
            t[i] = 0.0
    return t

def lhipa(d):
    # get signal duration (in seconds)
  #  tt = d[-1].timestamp() - d[0].timestamp()
    tt = d.loc[len(d)-1, ['pupil_timestamp']].values - d.loc[0, ['pupil_timestamp']].values
    # this has been edited to use pupil_timestamp from the Pupil Core

    d = d[setDiameter].to_list()
    
    # find max decomposition level
    w = pywt.Wavelet('sym16')
    maxlevel = \
        pywt.dwt_max_level(len(d), filter_len=w.dec_len)
        
    # set high and low frequency band indeces
    hif, lof = 1, int(maxlevel/2)
    
    # get detail coefficients of pupil diameter signal d
    cD_H = pywt.downcoef('d', d, 'sym16', 'per', level=hif)
    cD_L = pywt.downcoef('d', d, 'sym16', 'per', level=lof)
    
    # normalize by 1/√2j
    cD_H[:] = [x/math.sqrt(2**hif) for x in cD_H]
    cD_L[:] = [x/math.sqrt(2**lof) for x in cD_L]
    
    # obtain the LH:HF ratio
    cD_LH = cD_L
    for i in list(range(len(cD_L))):
        cD_LH[i] = cD_L[i] / cD_H[int(((2**lof)/(2**hif))*i)]
        
    # detect modulus maxima , see Duchowski et al. [15]
    cD_LHm = modmax(cD_LH)
    
    # threshold using universal threshold λuniv = σˆp(2logn)
    # where σˆ is the standard deviation of the noise
    lam_univ = \
        np.std(cD_LHm)*math.sqrt(2.0*np.log2(len(cD_LHm)))
    cD_LHt = pywt.threshold(cD_LHm, lam_univ, mode="less")
    
    # compute LHIPA
    ctr = 0
    for i in range(len(cD_LHt)):
        if math.fabs(cD_LHt[i]) > 0: ctr += 1
    LHIPA = float(ctr)/float(tt)
    return LHIPA

def LHIPAbyCondition(d, trials):
    IPAscore = {}
    for i in range(trials):
        IPAscore[i] = lhipa((d.loc[d['trial'] == i]).reset_index())
    return IPAscore

def importPupil(data):
    data[['Pupil Size','Trial Length','IPA','Blinks']] = 0
    participants = data.drop_duplicates(subset=['Participant'])['Participant'].tolist()
    for p in participants:
        print(f'importing pupil data for participant {p}')
        path = f'{directory}/pupil/{p}_Study3/000/exports/000/'
        print(path)
        try:
            os.chdir(path)
            pupil = pd.read_csv('pupil_positions.csv') 
            blinks = pd.read_csv('blinks.csv') 
            annotations = pd.read_csv('annotations.csv')
            try:
                print('labelling blinks')
                blinks = labelBlinkCondition(blinks, annotations) 
            except:
                print('error with blinks')
            try:
                print('calculating blink number per trial')
                totalBlinks = blinkNumber(blinks)
            except:
                print('error with blinks')
            try:
                print('removing blinks')
                pupil = blinkless(pupil, blinks)
            except:
                print('error with blinks')
            try:
                print('labelling trials')
                pupil = labelCondition(pupil, annotations)
            except:
                print('error labelling trials')
            try:
                print('calculating mean pupil size')
                trialMean, trialLen = sizeByTrial(pupil)
            except:
                print('error with pupil size')
            try:
                print('calculating LHIPA')
                activity = LHIPAbyCondition(pupil, trials)
            except:
                print('error in LHIPA')
            print('writing to data file')
            for j in range(trials):
                print(f'trial {j}')
                if j == 0:
                    continue
                else:
                    try:
                        data.loc[data[(data['Participant']==p) & (data['Trial']==j)].index, 'Pupil Size'] = trialMean[j]
                    except:
                        print('error writing data')
                    try:
                        data.loc[data[(data['Participant']==p) & (data['Trial']==j)].index, 'Trial Length'] = trialLen[j]
                    except:
                        print('error writing data')
                    try:
                        data.loc[data[(data['Participant']==p) & (data['Trial']==j)].index, 'IPA'] = activity[j]
                    except:
                        print('error writing data')
                    try:
                        data.loc[data[(data['Participant']==p) & (data['Trial']==j)].index, 'Blinks'] = totalBlinks[j]
                    except:
                        print('error writing data')
        except:
            print('not exported')
        print(f'finished participant {p}')
        print()
    return data

def segment(x, p):
    amp = 0
    x = [abs(ele) for ele in x]
    i = len(x)
    print(f'Audio is {i} frames long')
    start = []
    end = []
    t = trials-1 # number of trials
    while t > 0:
        while amp < 0.05: # was 500
            i = i-1
            if i <= 0:
                print('too far')
                break
            amp = statistics.mean(x[i-100:i+100])
            # print(f'Participant {p}, Inactive, Trial {t}, Frame {i}, Amplitude {amp}')
        end.append(i)
        print(f'Trial {t} ended at frame {i} with amplitude {amp}')
        while amp > 0.0001: # find when amplitude drops to 0
            i = i-1
            if i <= 0:
                print('too far')
                break
            amp = statistics.mean(x[i-100:i+100])
            # print(f'Participant {p}, Active, Trial {t}, Frame {i}, Amplitude {amp}')
        start.append(i)
        print(f'Trial {t} started at frame {i} with amplitude {amp}')
        t = t-1
    return start, end

def audioStimImport(data):
    data[['audioStart']] = 0
    data[['audioEnd']] = 0
    participants = data.drop_duplicates(subset=['Participant'])['Participant'].tolist()
    for p in participants:
        # note: p starts at 1, and must go to after the last participant
        # skip 1
        print('')
        print(f'Importing audio for participant {p}')
        file = f'STE-{p:03}.wav' 
        try:
            print('loading '+file)
            sound, rate = librosa.load(file, mono=False)
            stimulus = sound[0,:] # select left channel
            stimulus = list(stimulus)
            #rate, sound = wf.read(file)
            # stimulus = sound[:,0] # select Left channel
            # print(sound[0:10])
            # print(stimulus[0:10])
            print('Drawing figure...')
            plt.figure()
            plt.plot(stimulus)
            plt.title(f'Participant {p}')
            plt.show()
            print(f'Audio rate is {rate}')
            start, end = segment(stimulus, p)  
            # print(start)
            # print(end)
            counter = 0
            for t in range(6,0,-1): # iterate through trials
                # note: t starts at 6, ends at 1. Skip practice trial (0)
                # counter starts at 0, ends at 5. Skip practice trial (7)
                # audio timestamp lists (start, end) in reverse order, hence the counter
                try:
                    j = data[(data['Trial'] == t) & (data['Participant'] == p)].index[0]
                    print(f'Trial {t} in row {j}')
                    data.loc[j, 'audioStart'] = start[counter]                 
                    data.loc[j, 'audioEnd'] = end[counter]
                except:
                    print(f'No data for trial {t}')
                counter = counter + 1
            print(f'Updated audio data for participant {p}')
        except:
            print('No audio file at '+file)
    return data

def tapAnalysisLibrosa(tap, start, end, rate): 
    x = tap[start:end]
    length = len(x)
    if length > 1000:
        onset_frames = librosa.onset.onset_detect(x, sr=rate, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
        onset_times = librosa.frames_to_time(onset_frames)
        print(onset_times[0:10])
        # calculate IOIs rather than raw times
        beats = []
        for i in range(len(onset_times)):
            if i == 0:
                continue
            else:
                IOI = onset_times[i] - onset_times[i-1]
                beats.append(IOI)
        print(beats[0:10])
        try:
            stability = statistics.variance(beats)
            mean = statistics.mean(beats)
            print(stability)
        except:
            print('cannot calculate beat variance')
            stability = 0
            mean = 0
    else:
        stability = 0
        mean = 0
    return stability, length, mean

def audioTapImport(df):
    df[['Beat Variance']] = 0
    df[['Recording Length']] = 0
    df[['Calculated Length']] = 0
    df[['File Length']] = 0
    df[['Beat Mean']] = 0
    p = 0
    errors = []
    for i in range(len(df)):
        t = df.loc[i,'Trial']
        if p != df.loc[i,'Participant']:
            p = df.loc[i,'Participant']
            print(f'Importing audio for participant {p}')
            file = f'STE-{p:03}.wav' 
            print(f'loading {file}...')
            sound, rate = librosa.load(file, mono=False)
            tap = sound[1,:] # select right channel
            #rate, sound = wf.read(file)
            #tap = sound[:,1] # select Right channel
            print(sound[0:10])
            print(tap[0:10])
        else:
            print(f'Audio for participant {p} already imported')
        # divide in half to correct for different file lengths
        # if trim produced by waveform but analysing in librosa
        # really should have stuck with the same library...
        start = int(df.loc[i,'audioStart']) # /2
        end = int(df.loc[i,'audioEnd']) # /2
        calcLen = end - start
        fileLen = len(tap)
        print(f'Recording for trial {t} starts at {start} and ends at {end}')
        if start < 1:
            print('Recording too short')
            errors.append([file,t])
            continue
        else:
            print('Analysing beat stability...')
            stability, length, mean = tapAnalysisLibrosa(tap, start, end, rate)
            print(f'Recording of {length} frames with stability score of {stability}')
            df.loc[i,'Beat Variance'] = stability
            df.loc[i,'Recording Length'] = length
            df.loc[i,'Calculated Length'] = calcLen
            df.loc[i,'File Length'] = fileLen
            df.loc[i,'Beat Mean'] = mean
        print(f'Completed analysis for participant {p} trial {t}')
        print('')
    return df




def analysis(df):
    df = df.rename(columns={'Beat Mean':'IOI Mean (s)','Beat Variance':'IOI Variance','Pupil Size':'Pupil Size (mm)'})
    df['Tempo Ratio'] = df['Tempo Ratio'].replace({'120-120':'Synchrony','120-080':'Complementary','120-113':'Non-synchrony'})
    analyses = {'Difficulty':0,'Pupil Size (mm)':0,'LHIPA':0,'IOI Variance':0,'IOI Mean (s)':0,'Bonding':0}
    pairs = [('Synchrony','Complementary'),('Synchrony','Non-synchrony'),('Complementary','Non-synchrony')]
    order = ['Synchrony','Complementary','Non-synchrony']
    for i in analyses:
        ax = sns.violinplot(x='Tempo Ratio', y=i, data=df, cut=0, order=order)
        #annotator = Annotator(ax,pairs,data=df,x='Tempo Ratio',y=i, order=order)
        #annotator.configure(test='t-test_paired',comparisons_correction='bonferroni')
        #annotator.apply_and_annotate()
        plt.show()        
        analyses[i] = pg.rm_anova(dv=i, within='Tempo Ratio', subject='Participant', data=df, correction=True)
    return pd.concat(analyses)



def importQualtrics():
    quest = pd.read_csv('QualtricsData.csv')
    quest = quest.rename(columns={'Q33':'Participant','Q11':'Handedness','Q7':'Age','Q10':'Sex'})
   # quest = quest.drop([0,1,2,3,4]).reset_index()
    key = {
        'Disagree strongly': 1,
        'Disagree a little': 2,
        'Neither agree nor disagree': 3,
        'Agree a little': 4,
        'Agree strongly': 5
        }
    quest[['Openness']] = 0
    quest[['Conscientiousness']] = 0
    quest[['Extraversion']] = 0
    quest[['Agreeableness']] = 0
    quest[['Neuroticism']] = 0
    GMSIkey = {
        'Completely Disagree': 1,
        'Strongly Disagree': 2,
        'Disagree': 3,
        'Neither Agree nor Disagree': 4,
        'Agree': 5,
        'Strongly Agree': 6,
        'Completely Agree': 7
        }
    quest[['Perceptual Ability']] = 0
    quest[['Musical Training']] = 0
    for i in range(len(quest)):
        for j in range(1,11):
            item = f'Q21_{j}'
            x = quest.loc[i,item]
            try:
                quest.loc[i,item] = key[x]
            except:
                print(f'missing data at row {i}, {item}')
        for j in range(1,12):
            item = f'Q25_{j}'
            x = quest.loc[i,item]
            try:
                quest.loc[i,item] = GMSIkey[x]
            except:
                print(f'missing data at row {i}, {item}')
        try:
            quest.loc[i,'Openness'] = (6 - quest.loc[i,'Q21_5']) + quest.loc[i,'Q21_10']
            quest.loc[i,'Conscientiousness'] = (6 - quest.loc[i,'Q21_3']) + quest.loc[i,'Q21_8']
            quest.loc[i,'Extraversion'] = (6 - quest.loc[i,'Q21_1']) + quest.loc[i,'Q21_6']
            quest.loc[i,'Agreeableness'] = (6 - quest.loc[i,'Q21_7']) + quest.loc[i,'Q21_2']
            quest.loc[i,'Neuroticism'] = (6 - quest.loc[i,'Q21_4']) + quest.loc[i,'Q21_9']
            quest.loc[i,'Perceptual Ability'] = quest.loc[i,'Q25_1'] + quest.loc[i,'Q25_2'] + (8 - quest.loc[i,'Q25_3']) + quest.loc[i,'Q25_4'] + (8 - quest.loc[i,'Q25_5']) + quest.loc[i,'Q25_7'] + quest.loc[i,'Q25_8'] + (8 - quest.loc[i,'Q25_9']) + quest.loc[i,'Q25_10'] 
        except:
            print('missing data')
    quest = quest[['Participant',
                   'Age',
                   'Sex',
                   'Handedness',
                   'Openness',
                   'Conscientiousness',
                   'Extraversion',
                   'Agreeableness',
                   'Neuroticism',
                   'Perceptual Ability',
                   ]]
    return quest

def writeQualtrics(data):
    traits = [
        'Age',
        'Sex',
        'Handedness',
        'Openness',
        'Conscientiousness',
        'Extraversion',
        'Agreeableness',
        'Neuroticism',
        'Perceptual Ability'
        ]
    for t in traits:
        data[t] = np.nan
    for i in range(len(data)):
        p = data.loc[i,'Participant']
        for t in traits:
            try:
                data.loc[i,t] = quest.loc[quest['Participant'] == p,t].iloc[0]
            except:
                print(f'missing data for Participant {p} in {t}')
    return data
            
#%% Import data
data = importPsychoPy()
data = importPupil(data)

#%% Segment audio files
directory = "E:/Study3/Audio"
os.chdir(directory)
data = audioStimImport(data)

# there were some errors in audio import, see Error Checking below

#%% Combine bonding measures
data['Bonding'] = (data['Liking'] + data['IOS'] + data['Connection']) / 3

#%% Remove old columns
data = data.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','Unnamed: 0.1.1.1'],axis=1)

#%% Analyse tapping
directory = "E:/Study3/Audio"
os.chdir(directory)
data = audioTapImport(data)

#%% Import Qualtrics data
quest = importQualtrics()
data = writeQualtrics(data)

#%%
quest.to_csv('QualtricsData_processed.csv')

#%% Write to data
data.to_csv('data_2022-05-30.csv')

#%% Split conditions
conditions = pd.get_dummies(data['Tempo Ratio'])
drums = pd.get_dummies(data['Drums'])
data = pd.concat([data,conditions,drums],axis=1)

#%% Export to CSV
directory = "E:/Study3"
os.chdir(directory)
data.to_csv('data_2022-02-22.csv')

# That's all for the basic data processing

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

#%% Cronbach's Alpha
# for the bonding scale
print(pg.cronbach_alpha(data1[['Liking','Connection','IOS']]))



#%% Regression plot

ax = sns.lmplot(x='Pupil Size', y='Bonding', hue='Difficulty',data=data1, scatter_kws={"s": 5})

#%% 
df = data1
df['Tempo Ratio'] = df['Tempo Ratio'].replace({'120-120':'Synchrony','120-080':'Complementary','120-113':'Non-synchrony'})
df['Drums'] = df['Drums'].replace({'same':'Matched','diff':'Unmatched'})
ax = sns.violinplot(x='Tempo Ratio', y='Bonding', hue='Drums',data=df)
plt.legend(bbox_to_anchor=(1.02,1), loc='upper left', borderaxespad=0)


#%% Error checking
# printing start and end times to check for errors
# use against the plots generated at import

for p in range(1,83):
    end = int(data.loc[((data['Participant']==p) & (data['Trial']==6)),'audioEnd'])
    start = int(data.loc[((data['Participant']==p) & (data['Trial']==1)),'audioStart'])
    print(f'Participant {p} started at {start} and ended at {end}')
    
# audio recordings with messy bits:
    # 14, 24
    
# volume scaling issue:
    # 22, 23, 38, 39
    
# audio with blip at end
    # 3
    
# audio with missing starts:
    # 15, 20, 62, 63, 60
    
# audio to be excluded:
    # 1, 17, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 57, 73, 75, 76, 78
    
#%% Clean errors

# set different thresholds to deal with scaling issues
# this is really lazy, just copied and edited functions from above

def segment(x, p):
    amp = 0
    x = [abs(ele) for ele in x]
    i = len(x)
    print(f'Audio is {i} frames long')
    start = []
    end = []
    t = trials-1 # number of trials
    while t > 0:
        while amp < 0.01: # changed for scale
            i = i-1
            if i <= 0:
                print('too far')
                break
            amp = statistics.mean(x[i-100:i+100])
            # print(f'Participant {p}, Inactive, Trial {t}, Frame {i}, Amplitude {amp}')
        end.append(i)
        print(f'Trial {t} ended at frame {i} with amplitude {amp}')
        while amp > 0.00005: # changed for scale
            i = i-1
            if i <= 0:
                print('too far')
                break
            amp = statistics.mean(x[i-100:i+100])
            # print(f'Participant {p}, Active, Trial {t}, Frame {i}, Amplitude {amp}')
        start.append(i)
        print(f'Trial {t} started at frame {i} with amplitude {amp}')
        t = t-1
    return start, end

def audioStimImport(data):
    for p in [22, 23, 38, 39]: # just audtio with scaling issues
        print('')
        print(f'Importing audio for participant {p}')
        file = f'STE-{p:03}.wav' 
        try:
            print('loading '+file)
            sound, rate = librosa.load(file, mono=False)
            stimulus = sound[0,:] # select left channel
            stimulus = list(stimulus)
            print('Drawing figure...')
            plt.figure()
            plt.plot(stimulus)
            plt.title(f'Participant {p}')
            plt.show()
            print(f'Audio rate is {rate}')
            start, end = segment(stimulus, p)  
            # print(start)
            # print(end)
            counter = 0
            for t in range(6,0,-1): # iterate through trials
                # note: t starts at 6, ends at 1. Skip practice trial (0)
                # counter starts at 0, ends at 5. Skip practice trial (7)
                # audio timestamp lists (start, end) in reverse order, hence the counter
                try:
                    j = data[(data['Trial'] == t) & (data['Participant'] == p)].index[0]
                    print(f'Trial {t} in row {j}')
                    data.loc[j, 'audioStart'] = start[counter]                 
                    data.loc[j, 'audioEnd'] = end[counter]
                except:
                    print(f'No data for trial {t}')
                counter = counter + 1
            print(f'Updated audio data for participant {p}')
        except:
            print('No audio file at '+file)
    return data

directory = "E:/Study3/Audio"
os.chdir(directory)
data = audioStimImport(data)

#%% IOI calculation

def tempoCal(IOI):
    freq = 1 / IOI
    bpm = freq * 60
    return bpm

tempo1 = tempoCal(.52)
tempo2 = tempoCal(.39)

ratio = tempo2/tempo1

#%%

tempo1 = 120/(256/243) # tritone
ratio1 = 256/243
tempo2 = 120/(3/2) # P5
ratio2 = 3/2