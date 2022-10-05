#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.1.4),
    on August 23, 2021, at 22:46
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

import psychopy
psychopy.useVersion('latest')


from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.1.4'
expName = 'study3'  # from the Builder filename that created this script
expInfo = {'participant': '', 'trackEyes': '1'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='K:\\Josh\\OneDrive\\Documents\\DPhil\\Study 3\\Experiment\\study3_2021-08-10_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=(-0.1765, -0.1765, -0.1765), colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "welcome"
welcomeClock = core.Clock()
text_welcome = visual.TextStim(win=win, name='text_welcome',
    text='Welcome to this study.\nPlease read the following instructions carefully, as you cannot move back through the slides.\n\nPress any key to continue.',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_5 = keyboard.Keyboard()
trackEyes = int(expInfo['trackEyes'])
if expInfo['participant'] == '':
    p_num = 'test'
    in_balance = 0
else:
    p_num = int(expInfo['participant'])
    in_balance = 1

# if expInfo['between_condition'] == 's':
#     between_condition = 'same'
# else:
#     between_condition = 'diff'

if p_num < 37 or 109 > p_num > 72:
    if (p_num % 2) == 0: # is even
        between_condition = 'same'
    else:
        between_condition = 'diff'
elif 36 < p_num < 73 or p_num > 108:
    if (p_num % 2) == 0: # is even
        between_condition = 'diff'
    else:
        between_condition = 'same'   

import pandas as pd
from itertools import permutations
from random import randint

def stimuliGen(within_conditions, pairs, between_conditions):
    stimuli_matrix = []
    for sync in within_conditions:
        for p in pairs:
                for inst in between_conditions:
                    filename = 'videoStim/{}_{}_{}.mp4'.format(sync, inst, p)
                    tmp = {'within_condition': sync, 'between_condition': inst, 'pair': p, 'file_name': filename}
                    stimuli_matrix.append(tmp)
    return pd.DataFrame(stimuli_matrix)

def conditionSelector(conditions_in, pairs_in, stimuli_in, between_condition, counterbalance = 0):
    pair_select = randint(0, 719)
    if counterbalance == False:
        con_select1 = randint(0, 5)
        con_select2 = randint(0, 5)
    else:
        sel = []
        while len(sel) < 100:
            for i in range(6):
                for j in range(6):
                    x = [i, j]
                    sel.append(x)
        con_select1, con_select2 = sel[p_num]
    these_conditions = conditions_in[con_select1]+conditions_in[con_select2]
    these_pairs = pairs_in[pair_select]
    stimuli_selection = pd.DataFrame(columns=list(stimuli_in.columns))
    for j in range(6):
        stimulus = stimuli_in[ (stimuli_in['within_condition']==these_conditions[j]) & (stimuli_in['pair']==these_pairs[j]) & (stimuli_in['between_condition']==between_condition)]
        stimuli_selection = pd.concat([stimuli_selection,stimulus])
    return stimuli_selection

# Create various matrices
    
pairs = ['J-A', 'A-J', 'J-M', 'M-J', 'A-M', 'M-A']
pair_perm_matrix = list(permutations(pairs))

within_conditions = ['120-120','120-080','120-113']
con_perm_matrix = list(permutations(within_conditions))
between_conditions = ['same','diff']

stimuli_matrix = stimuliGen(within_conditions, pairs, between_conditions)

# Select a set of stimuli in a random order

selection = conditionSelector(con_perm_matrix, pair_perm_matrix, stimuli_matrix, between_condition, counterbalance = in_balance)
print(selection)
selection.to_csv('conditions.csv')
# set ths to conditions file for true randomisaton


if trackEyes == 1:
    import zmq
    import msgpack as serializer
    from time import time
    # setup zmq context and remote helper
    ctx = zmq.Context()
    pupil_remote = zmq.Socket(ctx, zmq.REQ)
    pupil_remote.connect("tcp://localhost:50020")
    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect("tcp://localhost:{}".format(pub_port))
    # set time for pupil core
    time_fn = time # could replace with psychopy.core.getTime(applyZero=True)
    pupil_remote.send_string("T {}".format(time_fn()))
    print(pupil_remote.recv_string())
    def notify(notification):
        """Sends ''notification'' to Pupil Remote"""
        topic = "notify."+notification["subject"]
        payload = serializer.dumps(notification, use_bin_type=True)
        pupil_remote.send_string(topic, flags=zmq.SNDMORE)
        pupil_remote.send(payload)
        return pupil_remote.recv_string()
    def send_trigger(trigger):
        payload = serializer.dumps(trigger, use_bin_type=True)
        pub_socket.send_string(trigger["topic"], flags=zmq.SNDMORE)
        pub_socket.send(payload)
    #start annotations plugin
    notify({"subject": "start_plugin", "name": "Annotation_Capture", "args": {}})
    def new_trigger(label, duration):
        return {
            "topic": "annotation",
            "label": label,
            "timestamp": time_fn(),
            "duration": duration,
            }
    print('pupil configured')
     

# Initialize components for Routine "instr1"
instr1Clock = core.Clock()
text_instr1 = visual.TextStim(win=win, name='text_instr1',
    text='In this task you will see pairs of people drumming side-by-side. For each video you will be asked to tap the drum in front of you with your dominant hand, in time with the person on the LEFT of the screen.\n\nPress any key to continue.',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_4 = keyboard.Keyboard()

# Initialize components for Routine "instr2"
instr2Clock = core.Clock()
text_instr2 = visual.TextStim(win=win, name='text_instr2',
    text='After each video, you will be asked some questions.\n\nPress any key to continue.',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_3 = keyboard.Keyboard()

# Initialize components for Routine "instr3"
instr3Clock = core.Clock()
text_instr3 = visual.TextStim(win=win, name='text_instr3',
    text='Most of these questions will be answered on a slider like the one below. In each case, please click and hold on the slider, and drag the marker to the desired location. When you release the mouse button your answer will be recorded and you will move to the next slide. Try this now to continue.',
    font='Arial',
    pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
slider_practice = visual.Slider(win=win, name='slider_practice',
    size=(1.0, 0.1), pos=(0, -0.3), units=None,
    labels=None, ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=None,
    color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=-1, readOnly=False)

# Initialize components for Routine "instr4"
instr4Clock = core.Clock()
text_instr4 = visual.TextStim(win=win, name='text_instr4',
    text='In addition to these sliders, sometimes you will also be asked to indicate your response using circles on the screen that you can drag into position. \n\nPress any key to continue.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_2 = keyboard.Keyboard()

# Initialize components for Routine "instr5"
instr5Clock = core.Clock()
text_instr5 = visual.TextStim(win=win, name='text_instr5',
    text='For these questions, you will see a circle like this on the right. When you click on or near the "A" circle, a "B" circle will appear.\n\nPress any key to continue.',
    font='Open Sans',
    pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_9 = keyboard.Keyboard()
image = visual.ImageStim(
    win=win,
    name='image', 
    image='circleA.png', mask=None,
    ori=0.0, pos=(0.2, -0.2), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)

# Initialize components for Routine "instr6"
instr6Clock = core.Clock()
text_instr6 = visual.TextStim(win=win, name='text_instr6',
    text='Once the "B" circle appears, keep holding down the mouse button, and drag the circle into the appropriate position. Your response is recorded when you release the mouse button. Try this on the next slide.\n\nPress any key to continue.',
    font='Open Sans',
    pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image='circleA.png', mask=None,
    ori=0.0, pos=(0.2, -0.2), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image='circleB.png', mask=None,
    ori=0.0, pos=(-0.2, -0.2), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
key_resp_10 = keyboard.Keyboard()

# Initialize components for Routine "instr_IOS"
instr_IOSClock = core.Clock()
slider_IOS = visual.Slider(win=win, name='slider_IOS',
    size=(.4, 0.3), pos=(0, -0.2), units=None,
    labels=None, ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=0.0,
    color='LightGray', fillColor='Blue', borderColor=[0,0,0], colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
selfCircle = visual.ImageStim(
    win=win,
    name='image', 
    image='circleB.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

otherCircle = visual.ImageStim(
    win=win,
    name='image', 
    image='circleA.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

selfCircle.opacity = 1
slider_IOS.marker = selfCircle
slider_IOS.tickLines.colors = 'dimgrey'
slider_IOS.markerPos = 1
slider_IOS.marker.opacity = 1
image_IOS = visual.ImageStim(
    win=win,
    name='image_IOS', 
    image='circleA.png', mask=None,
    ori=0.0, pos=(0.2, -0.2), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
text_IOS = visual.TextStim(win=win, name='text_IOS',
    text='Click next to the "A" circle and drag the "B" circle into the desired position. When you release the mouse button, your answer will be recorded and you will move to the next slide.',
    font='Open Sans',
    pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
text_2 = visual.TextStim(win=win, name='text_2',
    text='Press here!',
    font='Open Sans',
    pos=(-0.15, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
    color='Blue', colorSpace='rgb', opacity=0.0, 
    languageStyle='LTR',
    depth=-4.0);

# Initialize components for Routine "instr7"
instr7Clock = core.Clock()
text_instr7 = visual.TextStim(win=win, name='text_instr7',
    text='There will be six videos, each of one minute in length. These videos are taken from a standard set of stimuli for use in research. \n\nPlease take a moment to get comfortable with the drum. On the next slide you will see an example of the stimuli, except with only one drummer.\n\nPress any key to continue.',
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_8 = keyboard.Keyboard()

# Initialize components for Routine "demo_tap"
demo_tapClock = core.Clock()
demo_movie = visual.MovieStim3(
    win=win, name='demo_movie',
    noAudio = False,
    filename='videoStim\\\\demo.mov',
    ori=0.0, pos=(0, 0), opacity=None,
    loop=False,
    depth=0.0,
    )

# Initialize components for Routine "begin"
beginClock = core.Clock()
text_begin = visual.TextStim(win=win, name='text_begin',
    text="Take a short break.\n\nPress 'space' to begin session blockCount\n\nIt may take a moment to load.",
    font='Arial',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_6 = keyboard.Keyboard()
blockCount = 0
stmuliDump = [
    'movie_tap',
    'slider_liking_L',
    'slider_connection_L',
    'slider_IOS_L',
    'slider_liking_R',
    'slider_connection_R',
    'slider_IOS_R',
    'slider_difficulty'
    ]

# Initialize components for Routine "tap"
tapClock = core.Clock()
text_fixationCross = visual.TextStim(win=win, name='text_fixationCross',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "liking_L"
liking_LClock = core.Clock()
slider_liking_L = visual.Slider(win=win, name='slider_liking_L',
    size=(1.0, 0.1), pos=(0, -0.3), units=None,
    labels=('Not at all','Very much'), ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=None,
    color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
text_liking_L = visual.TextStim(win=win, name='text_liking_L',
    text='Please rate how much you like the person on the LEFT.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "recorded"
recordedClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Recorded!',
    font='Open Sans',
    pos=(0, -0.2), height=0.075, wrapWidth=None, ori=0.0, 
    color=(0.3569, 1.0000, -0.6314), colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "connection_L"
connection_LClock = core.Clock()
slider_connection_L = visual.Slider(win=win, name='slider_connection_L',
    size=(1.0, 0.1), pos=(0, -0.3), units=None,
    labels=('Not at all','Very connected'), ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=None,
    color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
text_connection_L = visual.TextStim(win=win, name='text_connection_L',
    text='Please rate how connected you felt to the person on the LEFT.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "recorded"
recordedClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Recorded!',
    font='Open Sans',
    pos=(0, -0.2), height=0.075, wrapWidth=None, ori=0.0, 
    color=(0.3569, 1.0000, -0.6314), colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "IOS_L"
IOS_LClock = core.Clock()
slider_IOS_L = visual.Slider(win=win, name='slider_IOS_L',
    size=(.4, 0.3), pos=(0, -0.2), units=None,
    labels=None, ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=0.0,
    color='LightGray', fillColor='Blue', borderColor=[0,0,0], colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
selfCircle = visual.ImageStim(
    win=win,
    name='image', 
    image='selfCircle.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

otherCircle = visual.ImageStim(
    win=win,
    name='image', 
    image='otherCircle.png', mask=None,
    ori=0, pos=(0, 0), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-2.0)

selfCircle.opacity = 1
slider_IOS_L.marker = selfCircle
slider_IOS_L.tickLines.colors = 'dimgrey'
slider_IOS_L.markerPos = 1
slider_IOS_L.marker.opacity = 1
image_IOS_L = visual.ImageStim(
    win=win,
    name='image_IOS_L', 
    image='otherCircle.png', mask=None,
    ori=0.0, pos=(0.2, -0.2), size=(0.5, 0.5),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
text_IOS_L = visual.TextStim(win=win, name='text_IOS_L',
    text='Please click next to the "other" circle and drag the "self" circle into the position that best describes your relationship to the person on the LEFT.',
    font='Open Sans',
    pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);

# Initialize components for Routine "recorded"
recordedClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Recorded!',
    font='Open Sans',
    pos=(0, -0.2), height=0.075, wrapWidth=None, ori=0.0, 
    color=(0.3569, 1.0000, -0.6314), colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "difficulty"
difficultyClock = core.Clock()
slider_difficulty = visual.Slider(win=win, name='slider_difficulty',
    size=(1.0, 0.1), pos=(0, -0.3), units=None,
    labels=('Very easy','Very difficult'), ticks=(1, 2), granularity=0.0,
    style='rating', styleTweaks=(), opacity=None,
    color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
    font='Open Sans', labelHeight=0.05,
    flip=False, depth=0, readOnly=False)
text_difficulty = visual.TextStim(win=win, name='text_difficulty',
    text='Please rate the difficulty of the task.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);

# Initialize components for Routine "recorded"
recordedClock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text='Recorded!',
    font='Open Sans',
    pos=(0, -0.2), height=0.075, wrapWidth=None, ori=0.0, 
    color=(0.3569, 1.0000, -0.6314), colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "end"
endClock = core.Clock()
text_4 = visual.TextStim(win=win, name='text_4',
    text="Thank you for taking part. Please call the experimenter for further instruction.\n\nPress 'space' to end.",
    font='Arial',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
key_resp_7 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "welcome"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_5.keys = []
key_resp_5.rt = []
_key_resp_5_allKeys = []
if trackEyes == 1:
    # start recording pupil
    pupil_remote.send_string("R {}_Study3".format(p_num))
    pupil_remote.recv_string()
    print('pupil recording')
# keep track of which components have finished
welcomeComponents = [text_welcome, key_resp_5]
for thisComponent in welcomeComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
welcomeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "welcome"-------
while continueRoutine:
    # get current time
    t = welcomeClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=welcomeClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_welcome* updates
    if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_welcome.frameNStart = frameN  # exact frame index
        text_welcome.tStart = t  # local t and not account for scr refresh
        text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
        text_welcome.setAutoDraw(True)
    
    # *key_resp_5* updates
    waitOnFlip = False
    if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_5.frameNStart = frameN  # exact frame index
        key_resp_5.tStart = t  # local t and not account for scr refresh
        key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
        key_resp_5.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_5.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_5.getKeys(keyList=None, waitRelease=False)
        _key_resp_5_allKeys.extend(theseKeys)
        if len(_key_resp_5_allKeys):
            key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
            key_resp_5.rt = _key_resp_5_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "welcome"-------
for thisComponent in welcomeComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_welcome.started', text_welcome.tStartRefresh)
thisExp.addData('text_welcome.stopped', text_welcome.tStopRefresh)
# check responses
if key_resp_5.keys in ['', [], None]:  # No response was made
    key_resp_5.keys = None
thisExp.addData('key_resp_5.keys',key_resp_5.keys)
if key_resp_5.keys != None:  # we had a response
    thisExp.addData('key_resp_5.rt', key_resp_5.rt)
thisExp.addData('key_resp_5.started', key_resp_5.tStartRefresh)
thisExp.addData('key_resp_5.stopped', key_resp_5.tStopRefresh)
thisExp.nextEntry()
# the Routine "welcome" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr1"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_4.keys = []
key_resp_4.rt = []
_key_resp_4_allKeys = []
# keep track of which components have finished
instr1Components = [text_instr1, key_resp_4]
for thisComponent in instr1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr1"-------
while continueRoutine:
    # get current time
    t = instr1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr1* updates
    if text_instr1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr1.frameNStart = frameN  # exact frame index
        text_instr1.tStart = t  # local t and not account for scr refresh
        text_instr1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr1, 'tStartRefresh')  # time at next scr refresh
        text_instr1.setAutoDraw(True)
    
    # *key_resp_4* updates
    waitOnFlip = False
    if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_4.frameNStart = frameN  # exact frame index
        key_resp_4.tStart = t  # local t and not account for scr refresh
        key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
        key_resp_4.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_4.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_4.getKeys(keyList=None, waitRelease=False)
        _key_resp_4_allKeys.extend(theseKeys)
        if len(_key_resp_4_allKeys):
            key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
            key_resp_4.rt = _key_resp_4_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr1"-------
for thisComponent in instr1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr1.started', text_instr1.tStartRefresh)
thisExp.addData('text_instr1.stopped', text_instr1.tStopRefresh)
# check responses
if key_resp_4.keys in ['', [], None]:  # No response was made
    key_resp_4.keys = None
thisExp.addData('key_resp_4.keys',key_resp_4.keys)
if key_resp_4.keys != None:  # we had a response
    thisExp.addData('key_resp_4.rt', key_resp_4.rt)
thisExp.addData('key_resp_4.started', key_resp_4.tStartRefresh)
thisExp.addData('key_resp_4.stopped', key_resp_4.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr2"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
instr2Components = [text_instr2, key_resp_3]
for thisComponent in instr2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr2"-------
while continueRoutine:
    # get current time
    t = instr2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr2* updates
    if text_instr2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr2.frameNStart = frameN  # exact frame index
        text_instr2.tStart = t  # local t and not account for scr refresh
        text_instr2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr2, 'tStartRefresh')  # time at next scr refresh
        text_instr2.setAutoDraw(True)
    
    # *key_resp_3* updates
    waitOnFlip = False
    if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
    if key_resp_3.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_3.getKeys(keyList=None, waitRelease=False)
        _key_resp_3_allKeys.extend(theseKeys)
        if len(_key_resp_3_allKeys):
            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr2"-------
for thisComponent in instr2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr2.started', text_instr2.tStartRefresh)
thisExp.addData('text_instr2.stopped', text_instr2.tStopRefresh)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
    key_resp_3.keys = None
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.addData('key_resp_3.started', key_resp_3.tStartRefresh)
thisExp.addData('key_resp_3.stopped', key_resp_3.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr2" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr3"-------
continueRoutine = True
# update component parameters for each repeat
slider_practice.reset()
# keep track of which components have finished
instr3Components = [text_instr3, slider_practice]
for thisComponent in instr3Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr3Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr3"-------
while continueRoutine:
    # get current time
    t = instr3Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr3Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr3* updates
    if text_instr3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr3.frameNStart = frameN  # exact frame index
        text_instr3.tStart = t  # local t and not account for scr refresh
        text_instr3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr3, 'tStartRefresh')  # time at next scr refresh
        text_instr3.setAutoDraw(True)
    
    # *slider_practice* updates
    if slider_practice.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        slider_practice.frameNStart = frameN  # exact frame index
        slider_practice.tStart = t  # local t and not account for scr refresh
        slider_practice.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(slider_practice, 'tStartRefresh')  # time at next scr refresh
        slider_practice.setAutoDraw(True)
    
    # Check slider_practice for response to end routine
    if slider_practice.getRating() is not None and slider_practice.status == STARTED:
        continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr3Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr3"-------
for thisComponent in instr3Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr3.started', text_instr3.tStartRefresh)
thisExp.addData('text_instr3.stopped', text_instr3.tStopRefresh)
thisExp.addData('slider_practice.response', slider_practice.getRating())
thisExp.addData('slider_practice.rt', slider_practice.getRT())
thisExp.addData('slider_practice.started', slider_practice.tStartRefresh)
thisExp.addData('slider_practice.stopped', slider_practice.tStopRefresh)
# the Routine "instr3" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr4"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_2.keys = []
key_resp_2.rt = []
_key_resp_2_allKeys = []
# keep track of which components have finished
instr4Components = [text_instr4, key_resp_2]
for thisComponent in instr4Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr4Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr4"-------
while continueRoutine:
    # get current time
    t = instr4Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr4Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr4* updates
    if text_instr4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr4.frameNStart = frameN  # exact frame index
        text_instr4.tStart = t  # local t and not account for scr refresh
        text_instr4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr4, 'tStartRefresh')  # time at next scr refresh
        text_instr4.setAutoDraw(True)
    
    # *key_resp_2* updates
    waitOnFlip = False
    if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_2.frameNStart = frameN  # exact frame index
        key_resp_2.tStart = t  # local t and not account for scr refresh
        key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
        key_resp_2.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_2.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_2.getKeys(keyList=None, waitRelease=False)
        _key_resp_2_allKeys.extend(theseKeys)
        if len(_key_resp_2_allKeys):
            key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
            key_resp_2.rt = _key_resp_2_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr4Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr4"-------
for thisComponent in instr4Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr4.started', text_instr4.tStartRefresh)
thisExp.addData('text_instr4.stopped', text_instr4.tStopRefresh)
# check responses
if key_resp_2.keys in ['', [], None]:  # No response was made
    key_resp_2.keys = None
thisExp.addData('key_resp_2.keys',key_resp_2.keys)
if key_resp_2.keys != None:  # we had a response
    thisExp.addData('key_resp_2.rt', key_resp_2.rt)
thisExp.addData('key_resp_2.started', key_resp_2.tStartRefresh)
thisExp.addData('key_resp_2.stopped', key_resp_2.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr4" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr5"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_9.keys = []
key_resp_9.rt = []
_key_resp_9_allKeys = []
# keep track of which components have finished
instr5Components = [text_instr5, key_resp_9, image]
for thisComponent in instr5Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr5Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr5"-------
while continueRoutine:
    # get current time
    t = instr5Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr5Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr5* updates
    if text_instr5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr5.frameNStart = frameN  # exact frame index
        text_instr5.tStart = t  # local t and not account for scr refresh
        text_instr5.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr5, 'tStartRefresh')  # time at next scr refresh
        text_instr5.setAutoDraw(True)
    
    # *key_resp_9* updates
    waitOnFlip = False
    if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_9.frameNStart = frameN  # exact frame index
        key_resp_9.tStart = t  # local t and not account for scr refresh
        key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
        key_resp_9.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_9.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_9.getKeys(keyList=None, waitRelease=False)
        _key_resp_9_allKeys.extend(theseKeys)
        if len(_key_resp_9_allKeys):
            key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
            key_resp_9.rt = _key_resp_9_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # *image* updates
    if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image.frameNStart = frameN  # exact frame index
        image.tStart = t  # local t and not account for scr refresh
        image.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
        image.setAutoDraw(True)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr5Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr5"-------
for thisComponent in instr5Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr5.started', text_instr5.tStartRefresh)
thisExp.addData('text_instr5.stopped', text_instr5.tStopRefresh)
# check responses
if key_resp_9.keys in ['', [], None]:  # No response was made
    key_resp_9.keys = None
thisExp.addData('key_resp_9.keys',key_resp_9.keys)
if key_resp_9.keys != None:  # we had a response
    thisExp.addData('key_resp_9.rt', key_resp_9.rt)
thisExp.addData('key_resp_9.started', key_resp_9.tStartRefresh)
thisExp.addData('key_resp_9.stopped', key_resp_9.tStopRefresh)
thisExp.nextEntry()
thisExp.addData('image.started', image.tStartRefresh)
thisExp.addData('image.stopped', image.tStopRefresh)
# the Routine "instr5" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr6"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_10.keys = []
key_resp_10.rt = []
_key_resp_10_allKeys = []
# keep track of which components have finished
instr6Components = [text_instr6, image_2, image_3, key_resp_10]
for thisComponent in instr6Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr6Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr6"-------
while continueRoutine:
    # get current time
    t = instr6Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr6Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr6* updates
    if text_instr6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_instr6.frameNStart = frameN  # exact frame index
        text_instr6.tStart = t  # local t and not account for scr refresh
        text_instr6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr6, 'tStartRefresh')  # time at next scr refresh
        text_instr6.setAutoDraw(True)
    
    # *image_2* updates
    if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image_2.frameNStart = frameN  # exact frame index
        image_2.tStart = t  # local t and not account for scr refresh
        image_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
        image_2.setAutoDraw(True)
    
    # *image_3* updates
    if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image_3.frameNStart = frameN  # exact frame index
        image_3.tStart = t  # local t and not account for scr refresh
        image_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
        image_3.setAutoDraw(True)
    
    # *key_resp_10* updates
    waitOnFlip = False
    if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_10.frameNStart = frameN  # exact frame index
        key_resp_10.tStart = t  # local t and not account for scr refresh
        key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
        key_resp_10.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_10.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_10.getKeys(keyList=None, waitRelease=False)
        _key_resp_10_allKeys.extend(theseKeys)
        if len(_key_resp_10_allKeys):
            key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
            key_resp_10.rt = _key_resp_10_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr6Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr6"-------
for thisComponent in instr6Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr6.started', text_instr6.tStartRefresh)
thisExp.addData('text_instr6.stopped', text_instr6.tStopRefresh)
thisExp.addData('image_2.started', image_2.tStartRefresh)
thisExp.addData('image_2.stopped', image_2.tStopRefresh)
thisExp.addData('image_3.started', image_3.tStartRefresh)
thisExp.addData('image_3.stopped', image_3.tStopRefresh)
# check responses
if key_resp_10.keys in ['', [], None]:  # No response was made
    key_resp_10.keys = None
thisExp.addData('key_resp_10.keys',key_resp_10.keys)
if key_resp_10.keys != None:  # we had a response
    thisExp.addData('key_resp_10.rt', key_resp_10.rt)
thisExp.addData('key_resp_10.started', key_resp_10.tStartRefresh)
thisExp.addData('key_resp_10.stopped', key_resp_10.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr6" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr_IOS"-------
continueRoutine = True
# update component parameters for each repeat
slider_IOS.reset()
# keep track of which components have finished
instr_IOSComponents = [slider_IOS, image_IOS, text_IOS, text_2]
for thisComponent in instr_IOSComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr_IOSClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr_IOS"-------
while continueRoutine:
    # get current time
    t = instr_IOSClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr_IOSClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *slider_IOS* updates
    if slider_IOS.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        slider_IOS.frameNStart = frameN  # exact frame index
        slider_IOS.tStart = t  # local t and not account for scr refresh
        slider_IOS.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(slider_IOS, 'tStartRefresh')  # time at next scr refresh
        slider_IOS.setAutoDraw(True)
    
    # Check slider_IOS for response to end routine
    if slider_IOS.getRating() is not None and slider_IOS.status == STARTED:
        continueRoutine = False
    
    # *image_IOS* updates
    if image_IOS.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        image_IOS.frameNStart = frameN  # exact frame index
        image_IOS.tStart = t  # local t and not account for scr refresh
        image_IOS.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(image_IOS, 'tStartRefresh')  # time at next scr refresh
        image_IOS.setAutoDraw(True)
    
    # *text_IOS* updates
    if text_IOS.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_IOS.frameNStart = frameN  # exact frame index
        text_IOS.tStart = t  # local t and not account for scr refresh
        text_IOS.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_IOS, 'tStartRefresh')  # time at next scr refresh
        text_IOS.setAutoDraw(True)
    
    # *text_2* updates
    if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
        # keep track of start time/frame for later
        text_2.frameNStart = frameN  # exact frame index
        text_2.tStart = t  # local t and not account for scr refresh
        text_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
        text_2.setAutoDraw(True)
    if text_2.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > text_2.tStartRefresh + 1.5-frameTolerance:
            # keep track of stop time/frame for later
            text_2.tStop = t  # not accounting for scr refresh
            text_2.frameNStop = frameN  # exact frame index
            win.timeOnFlip(text_2, 'tStopRefresh')  # time at next scr refresh
            text_2.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr_IOSComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr_IOS"-------
for thisComponent in instr_IOSComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('slider_IOS.response', slider_IOS.getRating())
thisExp.addData('slider_IOS.rt', slider_IOS.getRT())
thisExp.addData('slider_IOS.started', slider_IOS.tStartRefresh)
thisExp.addData('slider_IOS.stopped', slider_IOS.tStopRefresh)
thisExp.addData('image_IOS.started', image_IOS.tStartRefresh)
thisExp.addData('image_IOS.stopped', image_IOS.tStopRefresh)
thisExp.addData('text_IOS.started', text_IOS.tStartRefresh)
thisExp.addData('text_IOS.stopped', text_IOS.tStopRefresh)
thisExp.addData('text_2.started', text_2.tStartRefresh)
thisExp.addData('text_2.stopped', text_2.tStopRefresh)
# the Routine "instr_IOS" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "instr7"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_8.keys = []
key_resp_8.rt = []
_key_resp_8_allKeys = []
# keep track of which components have finished
instr7Components = [text_instr7, key_resp_8]
for thisComponent in instr7Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instr7Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instr7"-------
while continueRoutine:
    # get current time
    t = instr7Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instr7Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_instr7* updates
    if text_instr7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
        # keep track of start time/frame for later
        text_instr7.frameNStart = frameN  # exact frame index
        text_instr7.tStart = t  # local t and not account for scr refresh
        text_instr7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_instr7, 'tStartRefresh')  # time at next scr refresh
        text_instr7.setAutoDraw(True)
    
    # *key_resp_8* updates
    waitOnFlip = False
    if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
        # keep track of start time/frame for later
        key_resp_8.frameNStart = frameN  # exact frame index
        key_resp_8.tStart = t  # local t and not account for scr refresh
        key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
        key_resp_8.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_8.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_8.getKeys(keyList=None, waitRelease=False)
        _key_resp_8_allKeys.extend(theseKeys)
        if len(_key_resp_8_allKeys):
            key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
            key_resp_8.rt = _key_resp_8_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instr7Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instr7"-------
for thisComponent in instr7Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_instr7.started', text_instr7.tStartRefresh)
thisExp.addData('text_instr7.stopped', text_instr7.tStopRefresh)
# check responses
if key_resp_8.keys in ['', [], None]:  # No response was made
    key_resp_8.keys = None
thisExp.addData('key_resp_8.keys',key_resp_8.keys)
if key_resp_8.keys != None:  # we had a response
    thisExp.addData('key_resp_8.rt', key_resp_8.rt)
thisExp.addData('key_resp_8.started', key_resp_8.tStartRefresh)
thisExp.addData('key_resp_8.stopped', key_resp_8.tStopRefresh)
thisExp.nextEntry()
# the Routine "instr7" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "demo_tap"-------
continueRoutine = True
routineTimer.add(10.000000)
# update component parameters for each repeat
# keep track of which components have finished
demo_tapComponents = [demo_movie]
for thisComponent in demo_tapComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
demo_tapClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "demo_tap"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = demo_tapClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=demo_tapClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *demo_movie* updates
    if demo_movie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        demo_movie.frameNStart = frameN  # exact frame index
        demo_movie.tStart = t  # local t and not account for scr refresh
        demo_movie.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(demo_movie, 'tStartRefresh')  # time at next scr refresh
        demo_movie.setAutoDraw(True)
    if demo_movie.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > demo_movie.tStartRefresh + 10.0-frameTolerance:
            # keep track of stop time/frame for later
            demo_movie.tStop = t  # not accounting for scr refresh
            demo_movie.frameNStop = frameN  # exact frame index
            win.timeOnFlip(demo_movie, 'tStopRefresh')  # time at next scr refresh
            demo_movie.setAutoDraw(False)
    if demo_movie.status == FINISHED:  # force-end the routine
        continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in demo_tapComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "demo_tap"-------
for thisComponent in demo_tapComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
demo_movie.stop()

# set up handler to look after randomisation of conditions etc
conditions = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('conditions.csv', selection='0:7'),
    seed=None, name='conditions')
thisExp.addLoop(conditions)  # add the loop to the experiment
thisCondition = conditions.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisCondition.rgb)
if thisCondition != None:
    for paramName in thisCondition:
        exec('{} = thisCondition[paramName]'.format(paramName))

for thisCondition in conditions:
    currentLoop = conditions
    # abbreviate parameter names if possible (e.g. rgb = thisCondition.rgb)
    if thisCondition != None:
        for paramName in thisCondition:
            exec('{} = thisCondition[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "begin"-------
    continueRoutine = True
    # update component parameters for each repeat
    key_resp_6.keys = []
    key_resp_6.rt = []
    _key_resp_6_allKeys = []
    blockCount = blockCount + 1
    if blockCount == 1:
        text_begin = visual.TextStim(win=win, name='text_begin',
            text="We are now ready to start. Remember to tap the drum in time with the person on the left.\n\nPress 'space' to begin the first session.\n\nIt may take a moment to load.",
            font='Arial',
            pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
            color='white', colorSpace='rgb', opacity=1, 
            languageStyle='LTR',
            depth=0.0);
    else:
        text_begin = visual.TextStim(win=win, name='text_11',
            text="Take a short break.\n\nPress 'space' to begin session {} out of 6.".format(blockCount),
            font='Arial',
            pos=(0, 0), height=0.05, wrapWidth=None, ori=0, 
            color='white', colorSpace='rgb', opacity=1, 
            languageStyle='LTR',
            depth=0.0);
    
    
    # how do I unload these sliders so they don't eat my memory?
    del slider_liking_L
    del slider_connection_L
    del slider_IOS_L
    del slider_difficulty
    
    """
    try:
        movie_tap
    except NameError:
        print('not there')
    else:
        movie_tap._unload()
    
    try:
        slider_liking_L
    except NameError:
        print('not there')
    else:
        slider_liking_L._unload()
        
    try:
        slider_connection_L
    except NameError:
        print('not there')
    else:
        slider_connection_L._unload()
        
        
    try:
        slider_IOS_L
    except NameError:
        print('not there')
    else:
        slider_IOS_L._unload()
        
    try:
        slider_liking_R
    except NameError:
        print('not there')
    else:
        slider_liking_R._unload()
        
    try:
        slider_connection_R
    except NameError:
        print('not there')
    else:
        slider_connection_R._unload()
        
    try:
        slider_IOS_R
    except NameError:
        print('not there')
    else:
        slider_IOS_R._unload()
        
    try:
        slider_difficulty
    except NameError:
        print('not there')
    else:
        slider_difficulty._unload()
    """
        
    # something wrong with this. how to stop?
    
    #movie_tap._unload()
    #win.clearBuffer()
    # keep track of which components have finished
    beginComponents = [text_begin, key_resp_6]
    for thisComponent in beginComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    beginClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "begin"-------
    while continueRoutine:
        # get current time
        t = beginClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=beginClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_begin* updates
        if text_begin.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            text_begin.frameNStart = frameN  # exact frame index
            text_begin.tStart = t  # local t and not account for scr refresh
            text_begin.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_begin, 'tStartRefresh')  # time at next scr refresh
            text_begin.setAutoDraw(True)
        
        # *key_resp_6* updates
        waitOnFlip = False
        if key_resp_6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_6.frameNStart = frameN  # exact frame index
            key_resp_6.tStart = t  # local t and not account for scr refresh
            key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
            key_resp_6.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_6.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_6.getKeys(keyList=['space'], waitRelease=False)
            _key_resp_6_allKeys.extend(theseKeys)
            if len(_key_resp_6_allKeys):
                key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in beginComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "begin"-------
    for thisComponent in beginComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('text_begin.started', text_begin.tStartRefresh)
    conditions.addData('text_begin.stopped', text_begin.tStopRefresh)
    # check responses
    if key_resp_6.keys in ['', [], None]:  # No response was made
        key_resp_6.keys = None
    conditions.addData('key_resp_6.keys',key_resp_6.keys)
    if key_resp_6.keys != None:  # we had a response
        conditions.addData('key_resp_6.rt', key_resp_6.rt)
    conditions.addData('key_resp_6.started', key_resp_6.tStartRefresh)
    conditions.addData('key_resp_6.stopped', key_resp_6.tStopRefresh)
    print('begin block '+str(blockCount))
    
    #win.mouseVisible = False
    psychopy.event.Mouse(visible=False)
    # the Routine "begin" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "tap"-------
    continueRoutine = True
    routineTimer.add(61.000000)
    # update component parameters for each repeat
    movie_tap = visual.MovieStim3(
        win=win, name='movie_tap',units='pix', 
        noAudio = False,
        filename=file_name,
        ori=0, pos=(0, 0), opacity=1,
        loop=False,
        size=[1920, 1080],
        depth=0.0,
        )
    if trackEyes == 1:
        # annotate Pupil Core
        label = "condition_start"
        duration = 0.
        condition_trigger = new_trigger(label, duration)
        condition_trigger["trial"] = blockCount
        condition_trigger["within_condition"] = within_condition
        condition_trigger["between_condition"] = between_condition
        condition_trigger["pair"] = pair
        send_trigger(condition_trigger)
    # keep track of which components have finished
    tapComponents = [movie_tap, text_fixationCross]
    for thisComponent in tapComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    tapClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "tap"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = tapClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=tapClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *movie_tap* updates
        if movie_tap.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            movie_tap.frameNStart = frameN  # exact frame index
            movie_tap.tStart = t  # local t and not account for scr refresh
            movie_tap.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(movie_tap, 'tStartRefresh')  # time at next scr refresh
            movie_tap.setAutoDraw(True)
        if movie_tap.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > movie_tap.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                movie_tap.tStop = t  # not accounting for scr refresh
                movie_tap.frameNStop = frameN  # exact frame index
                win.timeOnFlip(movie_tap, 'tStopRefresh')  # time at next scr refresh
                movie_tap.setAutoDraw(False)
        if movie_tap.status == FINISHED:  # force-end the routine
            continueRoutine = False
        
        # *text_fixationCross* updates
        if text_fixationCross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_fixationCross.frameNStart = frameN  # exact frame index
            text_fixationCross.tStart = t  # local t and not account for scr refresh
            text_fixationCross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_fixationCross, 'tStartRefresh')  # time at next scr refresh
            text_fixationCross.setAutoDraw(True)
        if text_fixationCross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_fixationCross.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_fixationCross.tStop = t  # not accounting for scr refresh
                text_fixationCross.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_fixationCross, 'tStopRefresh')  # time at next scr refresh
                text_fixationCross.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in tapComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "tap"-------
    for thisComponent in tapComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    movie_tap.stop()
    if trackEyes == 1:
        # annotate Pupil Core
        label = "condition_end"
        duration = 0.
        condition_trigger = new_trigger(label, duration)
        condition_trigger["trial"] = blockCount
        condition_trigger["within_condition"] = within_condition
        condition_trigger["between_condition"] = between_condition
        condition_trigger["pair"] = pair
        send_trigger(condition_trigger)
        
    #win.mouseVisible = True
    psychopy.event.Mouse(visible=True)
    
    slider_difficulty = visual.Slider(win=win, name='slider_difficulty',
        size=(1.0, 0.1), pos=(0, -0.3), units=None,
        labels=('Very easy','Very difficult'), ticks=(1, 2), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, depth=0, readOnly=False)
        
    slider_IOS_L = visual.Slider(win=win, name='slider_IOS_L',
        size=(.4, 0.3), pos=(0, -0.2), units=None,
        labels=None, ticks=(1, 2), granularity=0.0,
        style='rating', styleTweaks=(), opacity=0.0,
        color='LightGray', fillColor='Blue', borderColor=[0,0,0], colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, depth=0, readOnly=False)
    slider_IOS_L.marker = selfCircle
    slider_IOS_L.tickLines.colors = 'dimgrey'
    slider_IOS_L.markerPos = 1
    slider_IOS_L.marker.opacity = 1
    
    slider_connection_L = visual.Slider(win=win, name='slider_connection_L',
        size=(1.0, 0.1), pos=(0, -0.3), units=None,
        labels=('Not at all','Very connected'), ticks=(1, 2), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, depth=0, readOnly=False)
        
    slider_liking_L = visual.Slider(win=win, name='slider_liking_L',
        size=(1.0, 0.1), pos=(0, -0.3), units=None,
        labels=('Not at all','Very much'), ticks=(1, 2), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        color='LightGray', fillColor='Blue', borderColor='Black', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, depth=0, readOnly=False)
    conditions.addData('text_fixationCross.started', text_fixationCross.tStartRefresh)
    conditions.addData('text_fixationCross.stopped', text_fixationCross.tStopRefresh)
    
    # ------Prepare to start Routine "liking_L"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_liking_L.reset()
    # keep track of which components have finished
    liking_LComponents = [slider_liking_L, text_liking_L]
    for thisComponent in liking_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    liking_LClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "liking_L"-------
    while continueRoutine:
        # get current time
        t = liking_LClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=liking_LClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_liking_L* updates
        if slider_liking_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_liking_L.frameNStart = frameN  # exact frame index
            slider_liking_L.tStart = t  # local t and not account for scr refresh
            slider_liking_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_liking_L, 'tStartRefresh')  # time at next scr refresh
            slider_liking_L.setAutoDraw(True)
        
        # Check slider_liking_L for response to end routine
        if slider_liking_L.getRating() is not None and slider_liking_L.status == STARTED:
            continueRoutine = False
        
        # *text_liking_L* updates
        if text_liking_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_liking_L.frameNStart = frameN  # exact frame index
            text_liking_L.tStart = t  # local t and not account for scr refresh
            text_liking_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_liking_L, 'tStartRefresh')  # time at next scr refresh
            text_liking_L.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in liking_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "liking_L"-------
    for thisComponent in liking_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('slider_liking_L.response', slider_liking_L.getRating())
    conditions.addData('slider_liking_L.rt', slider_liking_L.getRT())
    conditions.addData('slider_liking_L.started', slider_liking_L.tStartRefresh)
    conditions.addData('slider_liking_L.stopped', slider_liking_L.tStopRefresh)
    conditions.addData('text_liking_L.started', text_liking_L.tStartRefresh)
    conditions.addData('text_liking_L.stopped', text_liking_L.tStopRefresh)
    # the Routine "liking_L" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "recorded"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    recordedComponents = [text]
    for thisComponent in recordedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    recordedClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "recorded"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = recordedClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=recordedClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in recordedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "recorded"-------
    for thisComponent in recordedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('text.started', text.tStartRefresh)
    conditions.addData('text.stopped', text.tStopRefresh)
    
    # ------Prepare to start Routine "connection_L"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_connection_L.reset()
    # keep track of which components have finished
    connection_LComponents = [slider_connection_L, text_connection_L]
    for thisComponent in connection_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    connection_LClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "connection_L"-------
    while continueRoutine:
        # get current time
        t = connection_LClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=connection_LClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_connection_L* updates
        if slider_connection_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_connection_L.frameNStart = frameN  # exact frame index
            slider_connection_L.tStart = t  # local t and not account for scr refresh
            slider_connection_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_connection_L, 'tStartRefresh')  # time at next scr refresh
            slider_connection_L.setAutoDraw(True)
        
        # Check slider_connection_L for response to end routine
        if slider_connection_L.getRating() is not None and slider_connection_L.status == STARTED:
            continueRoutine = False
        
        # *text_connection_L* updates
        if text_connection_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_connection_L.frameNStart = frameN  # exact frame index
            text_connection_L.tStart = t  # local t and not account for scr refresh
            text_connection_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_connection_L, 'tStartRefresh')  # time at next scr refresh
            text_connection_L.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in connection_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "connection_L"-------
    for thisComponent in connection_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('slider_connection_L.response', slider_connection_L.getRating())
    conditions.addData('slider_connection_L.rt', slider_connection_L.getRT())
    conditions.addData('slider_connection_L.started', slider_connection_L.tStartRefresh)
    conditions.addData('slider_connection_L.stopped', slider_connection_L.tStopRefresh)
    conditions.addData('text_connection_L.started', text_connection_L.tStartRefresh)
    conditions.addData('text_connection_L.stopped', text_connection_L.tStopRefresh)
    # the Routine "connection_L" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "recorded"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    recordedComponents = [text]
    for thisComponent in recordedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    recordedClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "recorded"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = recordedClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=recordedClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in recordedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "recorded"-------
    for thisComponent in recordedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('text.started', text.tStartRefresh)
    conditions.addData('text.stopped', text.tStopRefresh)
    
    # ------Prepare to start Routine "IOS_L"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_IOS_L.reset()
    # keep track of which components have finished
    IOS_LComponents = [slider_IOS_L, image_IOS_L, text_IOS_L]
    for thisComponent in IOS_LComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    IOS_LClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "IOS_L"-------
    while continueRoutine:
        # get current time
        t = IOS_LClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=IOS_LClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_IOS_L* updates
        if slider_IOS_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_IOS_L.frameNStart = frameN  # exact frame index
            slider_IOS_L.tStart = t  # local t and not account for scr refresh
            slider_IOS_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_IOS_L, 'tStartRefresh')  # time at next scr refresh
            slider_IOS_L.setAutoDraw(True)
        
        # Check slider_IOS_L for response to end routine
        if slider_IOS_L.getRating() is not None and slider_IOS_L.status == STARTED:
            continueRoutine = False
        
        # *image_IOS_L* updates
        if image_IOS_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_IOS_L.frameNStart = frameN  # exact frame index
            image_IOS_L.tStart = t  # local t and not account for scr refresh
            image_IOS_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_IOS_L, 'tStartRefresh')  # time at next scr refresh
            image_IOS_L.setAutoDraw(True)
        
        # *text_IOS_L* updates
        if text_IOS_L.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_IOS_L.frameNStart = frameN  # exact frame index
            text_IOS_L.tStart = t  # local t and not account for scr refresh
            text_IOS_L.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_IOS_L, 'tStartRefresh')  # time at next scr refresh
            text_IOS_L.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in IOS_LComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "IOS_L"-------
    for thisComponent in IOS_LComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('slider_IOS_L.response', slider_IOS_L.getRating())
    conditions.addData('slider_IOS_L.rt', slider_IOS_L.getRT())
    conditions.addData('slider_IOS_L.started', slider_IOS_L.tStartRefresh)
    conditions.addData('slider_IOS_L.stopped', slider_IOS_L.tStopRefresh)
    conditions.addData('image_IOS_L.started', image_IOS_L.tStartRefresh)
    conditions.addData('image_IOS_L.stopped', image_IOS_L.tStopRefresh)
    conditions.addData('text_IOS_L.started', text_IOS_L.tStartRefresh)
    conditions.addData('text_IOS_L.stopped', text_IOS_L.tStopRefresh)
    # the Routine "IOS_L" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "recorded"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    recordedComponents = [text]
    for thisComponent in recordedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    recordedClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "recorded"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = recordedClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=recordedClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in recordedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "recorded"-------
    for thisComponent in recordedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('text.started', text.tStartRefresh)
    conditions.addData('text.stopped', text.tStopRefresh)
    
    # ------Prepare to start Routine "difficulty"-------
    continueRoutine = True
    # update component parameters for each repeat
    slider_difficulty.reset()
    # keep track of which components have finished
    difficultyComponents = [slider_difficulty, text_difficulty]
    for thisComponent in difficultyComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    difficultyClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "difficulty"-------
    while continueRoutine:
        # get current time
        t = difficultyClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=difficultyClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_difficulty* updates
        if slider_difficulty.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_difficulty.frameNStart = frameN  # exact frame index
            slider_difficulty.tStart = t  # local t and not account for scr refresh
            slider_difficulty.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_difficulty, 'tStartRefresh')  # time at next scr refresh
            slider_difficulty.setAutoDraw(True)
        
        # Check slider_difficulty for response to end routine
        if slider_difficulty.getRating() is not None and slider_difficulty.status == STARTED:
            continueRoutine = False
        
        # *text_difficulty* updates
        if text_difficulty.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_difficulty.frameNStart = frameN  # exact frame index
            text_difficulty.tStart = t  # local t and not account for scr refresh
            text_difficulty.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_difficulty, 'tStartRefresh')  # time at next scr refresh
            text_difficulty.setAutoDraw(True)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in difficultyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "difficulty"-------
    for thisComponent in difficultyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('slider_difficulty.response', slider_difficulty.getRating())
    conditions.addData('slider_difficulty.rt', slider_difficulty.getRT())
    conditions.addData('slider_difficulty.started', slider_difficulty.tStartRefresh)
    conditions.addData('slider_difficulty.stopped', slider_difficulty.tStopRefresh)
    conditions.addData('text_difficulty.started', text_difficulty.tStartRefresh)
    conditions.addData('text_difficulty.stopped', text_difficulty.tStopRefresh)
    # the Routine "difficulty" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "recorded"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    # keep track of which components have finished
    recordedComponents = [text]
    for thisComponent in recordedComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    recordedClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "recorded"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = recordedClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=recordedClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in recordedComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "recorded"-------
    for thisComponent in recordedComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    conditions.addData('text.started', text.tStartRefresh)
    conditions.addData('text.stopped', text.tStopRefresh)
    thisExp.nextEntry()
    
# completed 1 repeats of 'conditions'


# ------Prepare to start Routine "end"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_7.keys = []
key_resp_7.rt = []
_key_resp_7_allKeys = []
# keep track of which components have finished
endComponents = [text_4, key_resp_7]
for thisComponent in endComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
endClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "end"-------
while continueRoutine:
    # get current time
    t = endClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=endClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_4* updates
    if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_4.frameNStart = frameN  # exact frame index
        text_4.tStart = t  # local t and not account for scr refresh
        text_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
        text_4.setAutoDraw(True)
    
    # *key_resp_7* updates
    waitOnFlip = False
    if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_7.frameNStart = frameN  # exact frame index
        key_resp_7.tStart = t  # local t and not account for scr refresh
        key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
        key_resp_7.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if key_resp_7.status == STARTED and not waitOnFlip:
        theseKeys = key_resp_7.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_7_allKeys.extend(theseKeys)
        if len(_key_resp_7_allKeys):
            key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
            key_resp_7.rt = _key_resp_7_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end"-------
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_4.started', text_4.tStartRefresh)
thisExp.addData('text_4.stopped', text_4.tStopRefresh)
# check responses
if key_resp_7.keys in ['', [], None]:  # No response was made
    key_resp_7.keys = None
thisExp.addData('key_resp_7.keys',key_resp_7.keys)
if key_resp_7.keys != None:  # we had a response
    thisExp.addData('key_resp_7.rt', key_resp_7.rt)
thisExp.addData('key_resp_7.started', key_resp_7.tStartRefresh)
thisExp.addData('key_resp_7.stopped', key_resp_7.tStopRefresh)
thisExp.nextEntry()
# the Routine "end" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
if trackEyes == 1:
    # stop recording
    pupil_remote.send_string("r")
    pupil_remote.recv_string()
    print('pupil stopped')

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
