# -*- coding: utf-8 -*-
"""
 -- Study 2 Analysis --

Pre-processing for pupil data
Matching with PsychoPy data
This version now combines across participants!

Created on Sun Apr 18 09:21:23 2021
@author: Joshua Bamford
"""

import pandas as pd 
import os
import seaborn as sns
import glob
import pingouin as pg
from datetime import datetime
import math
import pywt
import numpy as np
import statistics
# import librosa
import matplotlib.pyplot as plt
from scipy import stats
from statannotations.Annotator import Annotator

#%% Import PsychoPy data
directory = "E:/Study2/data"
os.chdir(directory)

def loadData():
    fileList = []
    for file in glob.glob('*.csv'):
        print(file)
        fileList.append(file)
        p = int(file[0:3])
        if p in skipThese:
            continue
        else:
            df = pd.read_csv(file)
            df[['conditions0.thisN']] = df[['conditions0.thisN']] + 1
            df[['conditions1.thisN']] = df[['conditions1.thisN']] + 4
            df[['conditions2.thisN']] = df[['conditions2.thisN']] + 7
            df[['conditions0.thisN']] = df[['conditions0.thisN']].fillna(0)
            df[['conditions1.thisN']] = df[['conditions1.thisN']].fillna(0)
            df[['conditions2.thisN']] = df[['conditions2.thisN']].fillna(0)
            df['trial'] = df['conditions0.thisN'] + df['conditions1.thisN'] + df['conditions2.thisN']
            df = df.rename(columns={'key_resp.corr':'Acc','key_resp.rt':'RT'})
            df = df[['condition','PositionX1','Acc','RT','trial']].dropna().reset_index()
            df[['participant']] = p
            # recoding location of letter. May or may not analyse
            for i in range(len(df)):
                if df.loc[i,'PositionX1'] > 0:
                    df.loc[i,'PositionX1'] = 'right'
                elif df.loc[i,'PositionX1'] < 0:
                    df.loc[i,'PositionX1'] = 'left'
            results = df[['participant','Acc','RT','condition','trial']].groupby(['participant','condition','trial']).mean()
            if p == 1:
                data = results
            else:
                data = pd.concat([data,results])
    print('loaded data from PsychoPy')
    data = data.reset_index()
    return data, fileList

# Import Pupil data
def blinkless(d, blinks):
    for i in range(len(blinks)):
        blink_start = blinks.loc[i, 'start_frame_index']
        blink_end = blinks.loc[i, 'end_frame_index'] + 1 # must include last frame
        blink = list(range(blink_start, blink_end))
        d.loc[d['world_index'].isin(blink), setDiameter] = None # drop if blink detected
        d.loc[d[setDiameter] == 0, setDiameter] = None # drop if diameter = 0
    d = d.dropna() # remove all null values, including blinks
    return d

def labelCondition(d):
    d['condition'] = 'base'
    d['trial'] = 0
    for i in range(len(annotations)):
        if annotations.loc[i, 'label'] == 'condition_start':
            condition = annotations.loc[i, 'condition']
            trial = annotations.loc[i, 'trial']
            conditionStart = annotations.loc[i, 'timestamp']
            conditionEnd = annotations.loc[i+1, 'timestamp']
            conditionRange = d[d['pupil_timestamp'].between(conditionStart, conditionEnd)].index
            print('sorting trial {} condition {}'.format(trial,condition))
            for j in conditionRange:
                d.loc[j, 'condition'] = condition
                d.loc[j, 'trial'] = trial
        else:
            condition = annotations.loc[i, 'condition']
    return d

def sizeByTrial(d):
    print('calculating pupil size...')
    trialMean = {}
    trialLen = {}
    for i in range(trials):
        x = d.loc[d['trial'] == i]
        x = x[[setDiameter]].values
        trialLen[i] = len(x)
        trialMean[i] = float(sum(x)/len(x))
    return trialMean, trialLen

# Run IPA
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

def ipa(d):
    # obtain 2-level DWT of pupil diameter signal d
    try:
        (cA2, cD2, cD1) = pywt.wavedec(d, 'sym16', 'per', level=2)
    except ValueError:
        return
    # get signal duration (in seconds)
    tt = d.loc[len(d)-1, ['pupil_timestamp']] - d.loc[0, ['pupil_timestamp']] 
    # this has been edited to use pupil_timestamp from the Pupil Core
    # normalize by 1/2j , j = 2 for 2-level DWT
    cA2[:] = [x / math.sqrt(4.0) for x in cA2]
    cD1[:] = [x / math.sqrt(2.0) for x in cD1]
    cD2[:] = [x / math.sqrt(4.0) for x in cD2]
    # detect modulus maxima
    cD2m = modmax(cD2)
    # threshold using standard deviation of the noise
    lam_univ = np.std(cD2m) * math.sqrt(2.0*np.log2(len(cD2m)))
    cD2t = pywt.threshold(cD2m, lam_univ, mode="hard")
    # compute IPA
    ctr = 0
    for i in range(len(cD2t)):
        if math.fabs(cD2t[i]) > 0: ctr += 1
    IPA = float(ctr)/tt
    return IPA

def IPAbyCondition(d, trials, version):
    print('calculating IPA...')
    IPAscore = {}
    for i in range(trials):
        IPAscore[i] = version((d.loc[d['trial'] == i]).reset_index())
    return IPAscore

skipThese = [] # no pupil data for 18
data, fileList = loadData()
participantLog = pd.read_csv('E:/Study2/ParticipantLog.csv')
trials = 9+1
setDiameter = 'diameter_3d'
counter = 0
yesterday = 0
data[['pupil size','trial length','LHIPA']] = 0
participantLog['path'] = 0

for i in range(len(participantLog)):
    p = participantLog.loc[i,'Participant No.']
    print('loading participant {}'.format(p))
    if p in skipThese:
        continue
    else:
        today = datetime.strptime(participantLog.loc[i,'Date'], '%d/%m/%Y')
        participantLog.loc[i,'Date'] = today
        if yesterday == today: # counter only works if all data is there
            counter = counter + 1
        else:
            counter = 0
        yesterday = today
        try:
            path = 'E:/Study2/recordings/{}/00{}/exports/000/'.format(today.strftime('%Y_%m_%d'),counter)
            print('import from '+path)
            participantLog.loc[i,'path'] = path
            os.chdir(participantLog.loc[i,'path'])
            annotations = pd.read_csv('annotations.csv') 
            pupil = pd.read_csv('pupil_positions.csv') 
            blinks = pd.read_csv('blinks.csv') 
            fixations = pd.read_csv('fixations.csv')
            print('analysing pupil data')
            pupil = blinkless(pupil, blinks)
            pupil = labelCondition(pupil)
            trialMean, trialLen = sizeByTrial(pupil)
            activity = IPAbyCondition(pupil, trials, lhipa)
            #activity1 = IPAbyCondition(pupil, trials, ipa)
            print('analysing fixation data')
            fixations = fixations.rename(columns = {'start_timestamp':'pupil_timestamp'})
            fixations = labelCondition(fixations)
            fixations = fixations.groupby(['trial']).agg('count')
            # add to data
            print('writing data...')
            for j in range(trials):
                if j == 0:
                    continue
                else:
                    data.loc[data[(data['participant']==p) & (data['trial']==j)].index, 'pupil size'] = trialMean[j]
                    data.loc[data[(data['participant']==p) & (data['trial']==j)].index, 'trial length'] = trialLen[j]
                    data.loc[data[(data['participant']==p) & (data['trial']==j)].index, 'LHIPA'] = activity[j]
                    #data.loc[data[(data['participant']==p) & (data['trial']==j)].index, 'IPA'] = activity1[j]
                    data.loc[data[(data['participant']==p) & (data['trial']==j)].index, 'fixations'] = fixations.loc[j,'id']
        except:
            print('no pupil data at '+path)

# Prepare performance data
data = data.dropna(subset=['Acc','RT']).reset_index()
data = data[data['trial'].isin([4,5,6,7,8,9])]
data['performance'] = data['Acc']/data['RT']

#%% Audio
def segment(x, p):
    amp = 0
    x = [abs(ele) for ele in x] # convert to absolute values
    i = len(x)
    print(f'Audio is {i} frames long')
    start = []
    end = []
    t = trials # number of trials
    while t > 3:
        while amp < 0.01: # was 500
            i = i-1
            if i <= 0:
                print('too far')
                t=0
            amp = statistics.mean(x[i-100:i+100])
            #print(f'Participant {p}, Inactive, Trial {t}, Frame {i}, Amplitude {amp}')
        end.append(i)
        print(f'Trial {t} ended at frame {i} with amplitude {amp}')
        while amp > 0.0001: # find when amplitude drops to 0
            i = i-1
            if i <= 0:
                print('too far')
                t=0
            amp = statistics.mean(x[i-100:i+100])
            #print(f'Participant {p}, Active, Trial {t}, Frame {i}, Amplitude {amp}')
        start.append(i)
        print(f'Trial {t} started at frame {i} with amplitude {amp}')
        t = t-1
    return start, end

def audioStimImport(data, a, b, date):
    participants = data.drop_duplicates(subset=['Participant'])['Participant'].tolist()
    for p in participants[a:b]: 
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
            print(stimulus[0:10])
            print(f'Audio rate is {rate}')
            print('Drawing figure...')
            plt.figure()
            plt.plot(stimulus)
            plt.title(f'Participant {p}')
            plt.show()
            start, end = segment(stimulus, p)  
            # print(start)
            # print(end)
            counter = 0
            trial4 = 3 + 6 % len(start) # should be 6 trials in 'start' so this should equal 3. 
            # if less than 6, this will terminate the range early, so it doesn't overshoot the list.
            for t in range(9,trial4,-1): # iterate through trials
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
        data.to_csv(f'data_{date}_audioImport.csv')
    return data

trials = 9
data = audioStimImport(data)

#%% Do not run!

# audio file missing in p61-64? extra p60 in qualtrics?
# can check order of conditions to identify? audio from computer not recorded for some participants?
# memory error, too much data?

data = data.rename(columns = {
    'participant':'Participant',
    'trial':'Trial',
    'condition':'Condition',
    'pupil size':'Pupil Size',
    'trial length':'Trial Length',
    'fixations':'Fixations',
    'performance':'Performance'
    })

data[['audioStart']] = 0
data[['audioEnd']] = 0

directory = "E:/Study2/audio"
os.chdir(directory)
data = audioStimImport(data)

#%% Export to CSV
directory = "E:/Study2"
os.chdir(directory)
data.to_csv('data_2022-02-20.csv')

#%% Error checking
for p in range(1,106):
    try:
        end = int(data.loc[((data['Participant']==p) & (data['Trial']==9)),'audioEnd'])
        start = int(data.loc[((data['Participant']==p) & (data['Trial']==4)),'audioStart'])
        print(f'Participant {p} started at {start} and ended at {end}')
    except:
        print(f'Participant {p} does not exist')

# p4 had begining cut off
# p5 is a mess, exclude
# p6 is a mess, exclude
# p7 boom at end
# p9 boom at end
# p13 boom at end
# p15 too soft, lower thresholds
# p16 mess
# p17 should be fine, rerun
# p29 exclude
# p30 exclude
# p34 boom at end
# p49 don't know what happened
# p78 mess
# p80 don't know
# p90 don't know, poss boom
# exclusions = [4,5,6,7,9,13,15,16,17,29,30,34,49]

#%% Analyse tapping

def tapAnalysisLibrosa(tap, start, end, rate): 
    x = tap[start:end]
    length = len(x)
    if length > 500:
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
            try:
                sound, rate = librosa.load(file, mono=False)
            except:
                print(f'{file} not found')
                continue
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
    return df, errors

#
directory = "E:/Study2/audio"
os.chdir(directory)
data, errors = audioTapImport(data)

# Export to CSV

directory = "E:/Study2"
os.chdir(directory)
data.to_csv('data_2022-02-22.csv')

#%% Remove exclusions

exclusions = [4,5,6,7,9,13,15,16,17,29,30,34,49]
for i in range(len(data)):
    p = data.loc[i,'Participant']
    if p in exclusions:
        data.loc[i,'Beat Variance'] = 0
    if data.loc[i,'Beat Variance'] == 0:
        data.loc[i,'Beat Variance'] = None    
    if data.loc[i,'Pupil Size'] == 0:
        data.loc[i,'Pupil Size'] = None   
    if data.loc[i,'LHIPA'] == 0:
        data.loc[i,'LHIPA'] = None   
    if data.loc[i,'Beat Mean'] == 0:
        data.loc[i,'Beat Mean'] = None  

#%% Qualtrics import
def importQualtrics():
    quest = pd.read_csv('QualtricsData.csv')
    quest = quest.rename(columns={'Q33':'Participant','Q11':'Handedness','Q7':'Age','Q10':'Sex'})
    #quest = quest.drop([0,1,2,3,4]).reset_index()
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
quest = importQualtrics()

#%%
quest.to_csv('QualtricsData_processed.csv')

#%% Load questionare data to main dataset
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
    
#%% Export data to CSV
directory = "C:/Users/bamfo/OneDrive/Documents/DPhil/Study 2/Analysis"
os.chdir(directory)
data.to_csv('data_2022-04-18.csv')













#%% Load data from CSV
directory = "C:/Users/bamfo/OneDrive/Documents/DPhil/Study 2/Analysis"
os.chdir(directory)
data = pd.read_csv('data_2022-04-18.csv')

data = data.rename(columns={'Beat Variance':'IOI Variance','Beat Mean':'IOI Mean (s)'})

#%% Filter
# apply this to remove outliers

# remove trials shorter than 1000 frames. Implies a lot of excluded frames
# remove statistical outliers on all variables
# particularly rediculous pupil sizes
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

for i in analyses:
    # if i != 'Performance':
    #     data2 = data.dropna(subset=['Fixations'])
    # else:
    #     data2 = data
    # data2 = data1[(np.abs(stats.zscore(data1[[i]])) < 2.5).all(axis=1)].reset_index()
    # ax = sns.catplot(x='Condition', y=i, kind='violin', data=data1)
    
    ax = sns.violinplot(x='Condition', y=i, data=data1, cut=0, order=order)
    #annotator = Annotator(ax,pairs,data=data1,x='Condition',y=i, order=order, loc='outside')
    #annotator.configure(test='Mann-Whitney',comparisons_correction='bonferroni')
    #annotator.apply_and_annotate()
    plt.show()  
    
    
    analyses[i] = pg.rm_anova(dv=i, within='Condition', subject='Participant', data=data, correction=True)
analyses = pd.concat(analyses)
    
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


