#Practice task before scanner
#Caroline Charpentier - October 2017

from __future__ import absolute_import, division
import sys  # to get file system encoding
#sys.path.append("C:\Program Files (x86)\PsychoPy2\Lib\site-packages\PsychoPy-1.84.2-py2.7.egg")
#sys.path.append("C:\Program Files (x86)\PsychoPy2\Lib\site-packages")
from psychopy import locale_setup, gui, visual, core, data, event, logging
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import time # helps to give the os a tiny break during video presentation
import csv
import xlsxwriter
from xlrd import open_workbook


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
#_thisDir = "C:\Users\Caroline\Box Sync\Post-doc Projects\Observational learning\Task_fmri_replication"
os.chdir(_thisDir)
stim_dir = _thisDir + os.sep + 'stimuli'
vid_dir = _thisDir + os.sep + 'videos'

# Add functions
from functions import *


#___________________________________________________________
#_________START CODE - SET UP + INITIALIZATION _____________
#___________________________________________________________

# Store info about the experiment session
expName = 'ObsLearn'  # from the Builder filename that created this script
expInfo = {u'ConteID': u'', u'participant': u'', u'age': u'', u'gender (m/f)': u''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_Pract%s_%s' % (expInfo['ConteID'],expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='', extraInfo=expInfo, runtimeInfo=None,
    originPath=None, savePickle=True, saveWideText=True, dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Setup the Window
dispWidth = 1024
dispHeight = 768
win = visual.Window(size=(dispWidth, dispHeight), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb', blendMode='avg', useFBO=False, units='pix')
#win = visual.Window(size=(dispWidth, dispHeight), fullscr=True, allowGUI=False, 
#    monitor='testMonitor', color=[0.65,0.65,0.65], colorSpace='rgb', useFBO=False, units='pix')

# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialize components for Routine "trial"
trialClock = core.Clock()
obs_play = visual.TextStim(win=win, name='obs_play', text='', font='Arial', 
                           pos=(0, 50), height=50, wrapWidth=None, ori=0, color='white', 
                           colorSpace='rgb', opacity=1, depth=0.0)
fixation1 = visual.TextStim(win=win, name='fixation_1', text='+', font='Arial', 
                           pos=(0, 0), height=50, wrapWidth=None, ori=0, color='white', 
                           colorSpace='rgb', opacity=1, depth=0.0)
topscreen = visual.Rect(win=win, name='topscreen1', width=700, height=460, ori=0, 
                        pos=(0, 130), lineWidth=1, lineColor=[0.8,0.8,0.8], lineColorSpace='rgb', 
                        fillColor=None, fillColorSpace='rgb', opacity=1, depth=-1.0, interpolate=True)
botscreen = visual.Rect(win=win, name='botscreen', width=303, height=230, ori=0, 
                        pos=(0, -230), lineWidth=1, lineColor=[0.8,0.8,0.8], lineColorSpace='rgb', 
                        fillColor=None, fillColorSpace='rgb', opacity=1, depth=-3.0, interpolate=True)
sm1 = visual.ImageStim(win=win, name='sm1', image='sin', mask=None, ori=0, pos=(-250, 100), 
                       size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                       flipVert=False, texRes=128, interpolate=True, depth=0.0)
sm2 = visual.ImageStim(win=win, name='sm2', image='sin', mask=None, ori=0, pos=(0, 100), 
                       size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                       flipVert=False, texRes=128, interpolate=True, depth=0.0)
sm3 = visual.ImageStim(win=win, name='sm3', image='sin', mask=None, ori=0, pos=(250, 100), 
                       size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                       flipVert=False, texRes=128, interpolate=True, depth=0.0)
choose = visual.TextStim(win=win, name='choose', text='CHOOSE', font='Arial', pos=(0,-300),
                         height=50, wrapWidth=None, ori=0, color='white', 
                         colorSpace='rgb', opacity=1, depth=0.0)
smch = visual.ImageStim(win=win, name='sm3', image='sin', mask=None, ori=0, pos=(0, 100), 
                       size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                       flipVert=False, texRes=128, interpolate=True, depth=0.0)
videopath = os.path.join(os.getcwd(), 'videos', 'action_down_m1.mp4') #put an example action to start
#if not os.path.exists(videopath):
#    raise RuntimeError("Video File could not be found:" + videopath)
video = visual.MovieStim2(win=win, name='video', noAudio=True, filename=videopath,
                          pos=(0, -230), size=[293,220], flipVert=False, flipHoriz=False, 
                          volume = 0.0, loop=False, depth=-4.0)

# Initialize components for Routine "feedback"
feedbackClock = core.Clock()
tokenShown = ''
tsFile = ''
outc = ''
sm1_ch = visual.ImageStim(win=win, name='sm1_ch', image='sin', mask=None, ori=0, pos=(-250, 0), 
                          size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                          flipVert=False, texRes=128, interpolate=True, depth=-3.0)
sm2_ch = visual.ImageStim(win=win, name='sm2_ch', image='sin', mask=None, ori=0, pos=(0, 0), 
                          size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                          flipVert=False, texRes=128, interpolate=True, depth=-3.0)
sm3_ch = visual.ImageStim(win=win, name='sm3_ch', image='sin', mask=None, ori=0, pos=(250, 0), 
                          size=[161,334], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                          flipVert=False, texRes=128, interpolate=True, depth=-3.0)
missed = visual.TextStim(win=win, name='missed', text='Missed response!\n\nWait until next trial...', font=u'Arial',
                         pos=[0,0], height=50, wrapWidth=None, ori=0, color=u'white', colorSpace='rgb',
                         opacity=1, depth=-1.0)
fixation2 = visual.TextStim(win=win, name='fixation_2', text='+', font='Arial', 
                           pos=(0, 0), height=50, wrapWidth=None, ori=0, color='white', 
                           colorSpace='rgb', opacity=1, depth=0.0)
topscreen2 = visual.Rect(win=win, name='topscreen2', width=700, height=460, ori=0, 
                        pos=(0, 130), lineWidth=1, lineColor=[0.8,0.8,0.8], lineColorSpace='rgb', 
                        fillColor=None, fillColorSpace='rgb', opacity=1, depth=-1.0, interpolate=True)
botscreen2 = visual.Rect(win=win, name='botscreen', width=303, height=230, ori=0, 
                        pos=(0, -230), lineWidth=1, lineColor=[0.8,0.8,0.8], lineColorSpace='rgb', 
                        fillColor=None, fillColorSpace='rgb', opacity=1, depth=-3.0, interpolate=True)
token = visual.ImageStim(win=win, name='token', image='sin', mask=None, ori=0, 
                         pos=(0, 130), size=[120,120], color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, 
                         flipVert=False, texRes=128, interpolate=True, depth=-2.0)
fixation_iti = visual.TextStim(win=win, name='fixation_iti', text='+', font='Arial', 
                           pos=(0, 0), height=50, wrapWidth=None, ori=0, color='white', 
                           colorSpace='rgb', opacity=1, depth=0.0)

# Initialize components for Routine "break/questions"
breakClock = core.Clock()
take_break = visual.TextStim(win=win, name='take_break', text='',
    font='Arial', pos=(0, 0), height=40, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, depth=-1.0)

# Initialize components for Routine "thanks"
thanksClock = core.Clock()
thanksMsg = visual.TextStim(win=win, name='thanksMsg', text='default text', font='Arial',
    pos=[0, 0], height=40, wrapWidth=None, ori=0, color=u'white', colorSpace='rgb', opacity=1, depth=0.0)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 
vidClock = core.Clock()

#___________________________________________________________
#_________RUN PRACTICE______________________________________
#___________________________________________________________
#create experiment handler for practice session
pracName = 'Practice'
filenamePrac = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['ConteID'],expInfo['participant'], pracName, expInfo['date'])
thisPrac = data.ExperimentHandler(name=pracName, version='',
        extraInfo=expInfo, runtimeInfo=None, originPath=None,
        savePickle=True, saveWideText=True, dataFileName=filenamePrac)

#run practice instructions
thisPrac = InstructionsPractice(thisPrac,endExpNow)

#run practice
Practice(expInfo,thisPrac,stim_dir,filenamePrac)

#___________________________________________________________
#_________SHOW INSTRUCTIONS_________________________________
#___________________________________________________________

thisExp = Instructions(thisExp,endExpNow)

# the Routine "instrMain" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
globalClock.reset()

#___________________________________________________________
#_________CREATE TRIAL FILE FOR MAIN TASK PRACTICE__________
#___________________________________________________________
nb_tr = 60
subj = str(expInfo['participant'])
tr_list_all = CreateTrialFile_practice(nb_tr,subj)

#___________________________________________________________
#_________PREPARE TRIAL LOOP_________________________________
#___________________________________________________________

list_breaks = [30,60]
               
#create variable to calculate earnings and proportion correct
earnings = 0
earnings_bl = 0
nb_corr = 0
tr_nb = 1
tr_obs_nb = 1
block_nb = 1

win.setColor([-1,-1,-1], 'rgb')
win.flip()

#create vectors of jittered durations for fixation crosses and ITI
jit_fix1 = np.repeat([1.0,2.0,3.0,4.0],10).tolist() #fixation 1 happens 40 times (observe trials only)
jit_fix1 = np.random.permutation(jit_fix1).tolist()

jit_fix2 = np.repeat([1.0,2.0,3.0,4.0],15).tolist() #fixation 2 happens 60 times
jit_fix2 = np.random.permutation(jit_fix2).tolist()

jit_iti = np.repeat([1.0,2.0,3.0,4.0,5.0],12).tolist() #iti happens 60 times
jit_iti = np.random.permutation(jit_iti).tolist()

fname = 'trial_lists/trial_list_practice_Sub%s.xlsx' % (subj)

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1, method='sequential', extraInfo=expInfo, 
    originPath=-1, trialList=data.importConditions(fname), seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial.keys():
        exec(paramName + '= thisTrial.' + paramName)

#show block condition (stable or volatile) before start of the block
#condRoutine = True
#condmsg = 'This will be a STABLE block\n\n(the same token stays valuable for a while before it switches to a different color)\n\nPress right arrow key to start.'
#cond = visual.TextStim(win, text = condmsg, color = 'white', units = 'pix', pos=(0, 0), height=40)
#event.clearEvents(eventType='keyboard')
#while condRoutine == True:
#    cond.draw()
#    win.flip()   
#    startKey = event.getKeys(keyList='right')
#    if "escape" in startKey:
#        endExpNow = True
#    if len(startKey) > 0:  # at least one key from the list was pressed
#        condRoutine = False
#    if endExpNow or event.getKeys(keyList=["escape"]):
#        core.quit()
#routineTimer.reset()
#globalClock.reset()

#show a fixation cross for 1s before start of first block
fixx = visual.TextStim(win, text = '+', color = 'white', units = 'pix',pos=(0, 0), height=50)
fixx.draw()
win.flip()
core.wait(1)

#___________________________________________________________
#___________MAIN TRIAL LOOP_________________________________
#___________________________________________________________


for thisTrial in trials:
    currentLoop = trials
    miss = 0
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec(paramName + '= thisTrial.' + paramName)
        
    # ------Prepare to start Routine "trial"-------
    t = 0
    trialClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    #assign slot machine image
    #determine entropy level
    if buCond == 1: #low BU
        bu = 'lbu'
    elif buCond == 2: #high BU
        bu = 'hbu'
    #determine horizontal order
    if horizOrd == 1: #G-R-B
        lc = 'G'
        mc = 'R'
        rc = 'B'
    elif horizOrd == 2: #R-B-G
        lc = 'R'
        mc = 'B'
        rc = 'G'
    elif horizOrd == 3: #B-G-R
        lc = 'B'
        mc = 'G'
        rc = 'R'
    #select image file name
    if unavAct == 1:
        sm1n = '%s%i_%s_ua.png' % (lc,vertOrd, bu)
        sm2n = '%s%i_%s_a.png' % (mc,vertOrd, bu)
        sm3n = '%s%i_%s_a.png' % (rc,vertOrd, bu)
        k_list = ['down','right']
    elif unavAct == 2:
        sm1n = '%s%i_%s_a.png' % (lc,vertOrd, bu)
        sm2n = '%s%i_%s_ua.png' % (mc,vertOrd, bu)
        sm3n = '%s%i_%s_a.png' % (rc,vertOrd, bu)
        k_list = ['left','right']
    elif unavAct == 3:
        sm1n = '%s%i_%s_a.png' % (lc,vertOrd, bu)
        sm2n = '%s%i_%s_a.png' % (mc,vertOrd, bu)
        sm3n = '%s%i_%s_ua.png' % (rc,vertOrd, bu)
        k_list = ['left','down']
    sm1.setImage(stim_dir + os.sep + sm1n)
    sm2.setImage(stim_dir + os.sep + sm2n)
    sm3.setImage(stim_dir + os.sep + sm3n)
    
    #pick image for choice feedback    
    if corrAct == 1:
        corrKey = 'left'
        smchn = '%s%i_%s_p.png' % (lc,vertOrd, bu)
        smch.setPos((-250,120))
    elif corrAct == 2:
        corrKey = 'down'
        smchn = '%s%i_%s_p.png' % (mc,vertOrd, bu)
        smch.setPos((0,120))
    elif corrAct == 3:
        corrKey = 'right'
        smchn = '%s%i_%s_p.png' % (rc,vertOrd, bu)
        smch.setPos((250,120))
    smch.setImage(stim_dir + os.sep + smchn)

    
    #pick video file
    if trType == 1: #observe
        nvid = randint(5)+1
        # vid_name = 'action_%s_%s%i.mp4' % (corrKey, expInfo['gender (m/f)'], nvid) #depends on gender
        vid_name = 'action_%s_m%i.mp4' % (corrKey, nvid)
        video.setMovie(vid_dir + os.sep + vid_name)
    
    #extract trial components
    if trType == 1:
        obs_play.setText('Observe')
        fixation1.setPos((0,120))
        sm1.setPos((-250,120))
        sm2.setPos((0,120))
        sm3.setPos((250,120))
        fix1j = jit_fix1[tr_obs_nb-1]
        trialComponents = [obs_play, topscreen, botscreen, sm1, sm2, sm3, fixation1, video, smch]
    elif trType == 2:
        obs_play.setText('Play')
        fixation1.setPos((0,0))
        sm1.setPos((-250,0))
        sm2.setPos((0,0))
        sm3.setPos((250,0))
        key_choice = event.BuilderKeyResponse()
        trialComponents = [obs_play, sm1, sm2, sm3, choose, key_choice]
        
    for thisComponent in trialComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "trial"-------
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        tg = globalClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *obs_play* updates
        if t >= 0.0 and obs_play.status == NOT_STARTED:
            # keep track of start time/frame for later
            obs_play.tStart = t
            obs_play.tgStart = tg
            obs_play.frameNStart = frameN  # exact frame index
            obs_play.setAutoDraw(True) 
                
        frameRemains = 0.0 + 1.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
        if obs_play.status == STARTED and t >= frameRemains:
            obs_play.setAutoDraw(False)
               
        if trType == 1: #observe
            # *topscreen* updates
            if t >= 1.0 and topscreen.status == NOT_STARTED:
                # keep track of start time/frame for later
                topscreen.tStart = t
                topscreen.tgStart = tg
                topscreen.frameNStart = frameN  # exact frame index
                topscreen.setAutoDraw(True)
                botscreen.setAutoDraw(True)
                
            # *slot machine* updates
            if t >= 1.0 and sm1.status == NOT_STARTED:
                # keep track of start time/frame for later
                sm1.tStart = t
                sm1.tgStart = tg
                sm1.frameNStart = frameN  # exact frame index
                sm1.setAutoDraw(True)
                sm2.setAutoDraw(True)
                sm3.setAutoDraw(True)
            
            frameRemains = 1.0 + 2.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
            if sm1.status == STARTED and t >= frameRemains:
                sm1.setAutoDraw(False)
                sm2.setAutoDraw(False)
                sm3.setAutoDraw(False)
                sm1.status = NOT_STARTED
            
            # *fixation1* updates
            if t >= 3.0 and fixation1.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation1.tStart = t
                fixation1.tgStart = tg
                fixation1.frameNStart = frameN  # exact frame index
                fixation1.setAutoDraw(True)
                    
            frameRemains = 3.0 + fix1j - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation1.status == STARTED and t >= frameRemains:
                fixation1.setAutoDraw(False)
            
            # *video* updates
            if t >= 3.0+fix1j:
                if sm1.status == NOT_STARTED:
                    sm1.setAutoDraw(True)
                    sm2.setAutoDraw(True)
                    sm3.setAutoDraw(True)
                shouldflip = video.play() # Start the movie stim by preparing it to play
                video.tStart = t
                video.tgStart = tg
                video.frameNStart = frameN  # exact frame index
                
                tv = 0
                vidClock.reset()  # clock
                while video.status != visual.FINISHED:
                    tv = vidClock.getTime()
                    if shouldflip:     
                        win.flip()
                    else:
                        time.sleep(0.001) #give the os a break if a flip is not needed
                    shouldflip = video.draw()   
                    
                    if tv > 1.1 and smch.status == NOT_STARTED:
                        if corrAct == 1:
                            if sm1.status == STARTED and t >= frameRemains:
                                sm1.setAutoDraw(False)
                        elif corrAct == 2:
                            if sm2.status == STARTED and t >= frameRemains:
                                sm2.setAutoDraw(False)
                        elif corrAct == 3:
                            if sm3.status == STARTED and t >= frameRemains:
                                sm3.setAutoDraw(False)
                        smch.setAutoDraw(True) #show partner's choice feedback
                        
                if video.status == FINISHED:  # force-end the routine
                    continueRoutine = False
        
        elif trType == 2: #play
            # *slot machine* updates
            if t >= 1.0 and sm1.status == NOT_STARTED:
                # keep track of start time/frame for later
                sm1.tStart = t
                sm1.tgStart = tg
                sm1.frameNStart = frameN  # exact frame index
                sm1.setAutoDraw(True)
                sm2.setAutoDraw(True)
                sm3.setAutoDraw(True)
               
            # *key_choice* updates
            if t >= 3.0 and key_choice.status == NOT_STARTED:
                # keep track of start time/frame for later
                key_choice.tStart = t
                key_choice.frameNStart = frameN  # exact frame index
                key_choice.status = STARTED
                # keyboard checking is just starting
                win.callOnFlip(key_choice.clock.reset)  # t=0 on next screen flip
                event.clearEvents(eventType='keyboard')
                choose.tgStart = tg
                        
            if key_choice.status == STARTED:
                choose.setAutoDraw(True)
                theseKeys = event.getKeys(keyList=k_list)    
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    key_choice.onset = globalClock.getTime()
                    key_choice.keys = theseKeys[-1]  # just the last key pressed
                    key_choice.rt = key_choice.clock.getTime()
                    if key_choice.keys == 'left':
                        choice = 1
                    elif key_choice.keys == 'down':
                        choice = 2
                    elif key_choice.keys == 'right':
                        choice = 3
                    # was this 'correct'?
                    if (key_choice.keys == str(corrKey)) or (key_choice.keys == corrKey):
                        key_choice.corr = 1
                    else:
                        key_choice.corr = 0
                    # a response ends the routine
                    continueRoutine = False
                frameRemains = 3.0 + 3.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
                if t >= frameRemains:
                    miss = 1
                    continueRoutine = False # end the routine after 3 seconds if no response
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    # store data for trials (TrialHandler)
    trials.addData('obsplay_onset',obs_play.tgStart)
    trials.addData('sm_onset',sm1.tgStart) 
    trials.addData('miss',miss)
    if trType == 2: #play 
        trials.addData('choose_onset',choose.tgStart)
        if miss == 0:  # we had a response
            trials.addData('choice',key_choice.keys)
            trials.addData('isCorr', key_choice.corr)
            trials.addData('choiceRT', key_choice.rt)
            trials.addData('resp_onset',key_choice.onset)      
    elif trType == 1: #observe
        trials.addData('fixation1_onset',fixation1.tgStart)
        trials.addData('video_nb',nvid)
        trials.addData('video_onset',video.tgStart)
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "feedback"-------
    fix2j = jit_fix2[tr_nb-1]
    fixij = jit_iti[tr_nb-1]
    t = 0
    feedbackClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    if trType == 2: #play
        routineTimer.add(0.5000 + fix2j + 1.0000 + fixij)
    elif trType == 1: #observe
        routineTimer.add(fix2j + 1.0000 + fixij)
                
    #determine which token to show
    if trType == 1:
        ch = corrAct #partner is always correct
    elif trType == 2 and miss == 0:
        ch = choice
    else: #missed response
        ch = ''
    n = random()
    if (ch == 1 and lc == 'G') or (ch == 2 and mc == 'G') or (ch == 3 and rc == 'G'): #green sm chosen
        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5):
            tokenShown = 1
            tsFile = 'green.png'
        elif (buCond == 1 and n > 0.75 and n <= 0.95) or (buCond == 2 and n > 0.5 and n <= 0.8):
            tokenShown = 2
            tsFile = 'red.png'
        elif (buCond == 1 and n > 0.95) or (buCond == 2 and n > 0.8):
            tokenShown = 3
            tsFile = 'blue.png'
        if buCond == 1:
            P_green = 0.75
            P_red = 0.2
            P_blue = 0.05
        elif buCond == 2:
            P_green = 0.5
            P_red = 0.3
            P_blue = 0.2
    elif (ch == 1 and lc == 'R') or (ch == 2 and mc == 'R') or (ch == 3 and rc == 'R'): #red sm chosen
        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5):
            tokenShown = 2
            tsFile = 'red.png'
        elif (buCond == 1 and n > 0.75 and n <= 0.95) or (buCond == 2 and n > 0.5 and n <= 0.8):
            tokenShown = 3
            tsFile = 'blue.png'
        elif (buCond == 1 and n > 0.95) or (buCond == 2 and n > 0.8):
            tokenShown = 1
            tsFile = 'green.png'
        if buCond == 1:
            P_red = 0.75
            P_blue = 0.2
            P_green = 0.05
        elif buCond == 2:
            P_red = 0.5
            P_blue = 0.3
            P_green = 0.2
    elif (ch == 1 and lc == 'B') or (ch == 2 and mc == 'B') or (ch == 3 and rc == 'B'): #blue sm chosen
        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5):
            tokenShown = 3
            tsFile = 'blue.png'
        elif (buCond == 1 and n > 0.75 and n <= 0.95) or (buCond == 2 and n > 0.5 and n <= 0.8):
            tokenShown = 1
            tsFile = 'green.png'
        elif (buCond == 1 and n > 0.95) or (buCond == 2 and n > 0.8):
            tokenShown = 2
            tsFile = 'red.png'
        if buCond == 1:
            P_blue = 0.75
            P_green = 0.2
            P_red = 0.05
        elif buCond == 2:
            P_blue = 0.5
            P_green = 0.3
            P_red = 0.2
    
    if ch != '':
        #determine outcome value and whether token shown is the goal
        if tokenShown == goalToken:
            isgoal = 1
            if trType == 2: #play
                ov = 10
            else:
                ov = 0       
        else:
            isgoal = 0
            ov = 0   
        if goalToken == 1:
            P_goal = P_green
        elif goalToken == 2:
            P_goal = P_red
        elif goalToken == 3:
            P_goal = P_blue
        
        #determine choice feedback position and stimuli
        if trType == 2:
            if key_choice.keys == 'left':
                sm1chn = '%s%i_%s_p.png' % (lc,vertOrd, bu)
                sm2chn = sm2n
                sm3chn = sm3n
            elif key_choice.keys == 'down':
                sm1chn = sm1n
                sm2chn = '%s%i_%s_p.png' % (mc,vertOrd, bu)
                sm3chn = sm3n
            elif key_choice.keys == 'right':
                sm1chn = sm1n
                sm2chn = sm2n
                sm3chn = '%s%i_%s_p.png' % (rc,vertOrd, bu)
            sm1_ch.setImage(stim_dir + os.sep + sm1chn)
            sm2_ch.setImage(stim_dir + os.sep + sm2chn)
            sm3_ch.setImage(stim_dir + os.sep + sm3chn)
            fixation2.setPos((0,0))
            token.setPos((0,0))   
        elif trType == 1:
            fixation2.setPos((0,120))      
            token.setPos((0,120))    
        token.setImage(stim_dir + os.sep + tsFile)
    
    else: #missed response
        ov = 0
        
    # keep track of which components have finished
    if trType == 1:
        feedbackComponents = [fixation2, topscreen2, botscreen2, token, fixation_iti]
    elif trType == 2 and miss == 0:
        feedbackComponents = [sm1_ch, sm2_ch, sm3_ch, fixation2, token, fixation_iti]
    else: #missed
        feedbackComponents = [missed, fixation_iti]
    for thisComponent in feedbackComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    
    # -------Start Routine "feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = feedbackClock.getTime()
        tg = globalClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        if trType == 2 and miss == 0: # play
            # *sm1_ch* updates
            if t >= 0.0 and sm1_ch.status == NOT_STARTED:
                # keep track of start time/frame for later
                sm1_ch.tStart = t
                sm1_ch.tgStart = tg
                sm1_ch.frameNStart = frameN  # exact frame index
                sm1_ch.setAutoDraw(True)
                sm2_ch.setAutoDraw(True)
                sm3_ch.setAutoDraw(True)
                        
            frameRemains = 0.0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
            if sm1_ch.status == STARTED and t >= frameRemains:
                sm1_ch.setAutoDraw(False)
                sm2_ch.setAutoDraw(False)
                sm3_ch.setAutoDraw(False)
                
            # *fixation2* updates
            if t >= 0.5 and fixation2.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation2.tStart = t
                fixation2.tgStart = tg
                fixation2.frameNStart = frameN  # exact frame index
                fixation2.setAutoDraw(True)
                    
            frameRemains = 0.5 + fix2j - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation2.status == STARTED and t >= frameRemains:
                fixation2.setAutoDraw(False)
        
            # *token* updates
            if t >= 0.5 + fix2j and token.status == NOT_STARTED:
                # keep track of start time/frame for later
                token.tStart = t
                token.tgStart = tg
                token.frameNStart = frameN  # exact frame index
                token.setAutoDraw(True)
                    
            frameRemains = 0.5 + fix2j + 1.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
            if token.status == STARTED and t >= frameRemains:
                token.setAutoDraw(False)
            
            # *fixation_iti* updates
            if t >= 1.5 + fix2j and fixation_iti.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation_iti.tStart = t
                fixation_iti.tgStart = tg
                fixation_iti.frameNStart = frameN  # exact frame index
                fixation_iti.setAutoDraw(True)
                    
            frameRemains = 1.5 + fix2j + fixij - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation_iti.status == STARTED and t >= frameRemains:
                fixation_iti.setAutoDraw(False)
        
        elif trType == 1: #observe
            # *topscreen2* and *botscreen2* updates - stay on for the whole trial
            if t >= 0.0 and topscreen2.status == NOT_STARTED:
                # keep track of start time/frame for later
                topscreen2.tStart = t
                topscreen2.tgStart = tg
                topscreen2.frameNStart = frameN  # exact frame index
                topscreen2.setAutoDraw(True)
                botscreen2.setAutoDraw(True)
            frameRemains = 0.0 + fix2j + 1.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
            if topscreen2.status == STARTED and t >= frameRemains:
                topscreen2.setAutoDraw(False)
                botscreen2.setAutoDraw(False)
                
            # *fixation2* updates
            if t >= 0.0 and fixation2.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation2.tStart = t
                fixation2.tgStart = tg
                fixation2.frameNStart = frameN  # exact frame index
                fixation2.setAutoDraw(True)
                    
            frameRemains = 0.0 + fix2j - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation2.status == STARTED and t >= frameRemains:
                fixation2.setAutoDraw(False)
        
            # *token* updates
            if t >= fix2j and token.status == NOT_STARTED:
                # keep track of start time/frame for later
                token.tStart = t
                token.tgStart = tg
                token.frameNStart = frameN  # exact frame index
                token.setAutoDraw(True)
                    
            frameRemains = fix2j + 1.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
            if token.status == STARTED and t >= frameRemains:
                token.setAutoDraw(False)
            
            # *fixation_iti* updates
            if t >= fix2j + 1.0 and fixation_iti.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation_iti.tStart = t
                fixation_iti.tgStart = tg
                fixation_iti.frameNStart = frameN  # exact frame index
                fixation_iti.setAutoDraw(True)
                    
            frameRemains = fix2j + 1.0 + fixij - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation_iti.status == STARTED and t >= frameRemains:
                fixation_iti.setAutoDraw(False)
            
        else: #missed response
            # *missed* updates
            if t >= 0.0 and missed.status == NOT_STARTED:
                # keep track of start time/frame for later
                missed.tStart = t
                missed.tgStart = tg
                missed.frameNStart = frameN  # exact frame index
                missed.setAutoDraw(True)
                    
            frameRemains = 1.5 + fix2j - win.monitorFramePeriod * 0.75  # most of one frame period left
            if missed.status == STARTED and t >= frameRemains:
                missed.setAutoDraw(False)
            
            # *fixation_iti* updates
            if t >= 1.5 + fix2j and fixation_iti.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation_iti.tStart = t
                fixation_iti.tgStart = tg
                fixation_iti.frameNStart = frameN  # exact frame index
                fixation_iti.setAutoDraw(True)
                    
            frameRemains = 1.5 + fix2j + fixij - win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation_iti.status == STARTED and t >= frameRemains:
                fixation_iti.setAutoDraw(False)
                   
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "feedback"-------
    for thisComponent in feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    #ADD TRIAL DATA TO RESULT FILE
    trials.addData('outcome', ov)
    trials.addData('fixationiti_onset', fixation_iti.tgStart)
    if miss == 0:
        trials.addData('fixation2_onset', fixation2.tgStart)
        trials.addData('token_onset', token.tgStart)
        trials.addData('pGreen', P_green)
        trials.addData('pRed', P_red)
        trials.addData('pBlue', P_blue)
        trials.addData('pGoal', P_goal)
        trials.addData('randn', n)
        trials.addData('tokenShown', tokenShown)
        trials.addData('isGoal', isgoal)
    else:
        trials.addData('missed_onset', missed.tgStart)
        
    earnings_bl = earnings_bl + ov
    earnings = earnings + ov
    if trType == 2 and miss == 0:
        nb_corr = nb_corr + key_choice.corr
        trials.addData('ch_fb_onset',sm1_ch.tgStart)
    
    if trType == 1:
        tr_obs_nb = tr_obs_nb + 1 #only update observe trial number on observe trials
    
    #clear variables to make sure they are re-computed on every trial
    del ch
    del ov
    if miss == 0:
        del tokenShown
        del tsFile
        del P_green
        del P_red
        del P_blue
        del P_goal
        del isgoal
    
    
    #___________________________________________________________
    #_____________BREAK/END OF PRACTICE RUN_____________________
    #___________________________________________________________
    if tr_nb in list_breaks:
        
        if tr_nb == 30:
            break_text = 'End of first block.\n\nIn this block you have earned $%.2f!\n\nTake a short break if you need to. Press any key to start the next block.' %(earnings_bl/100.0)
        else:
            break_text = 'End of the practice task.\n\nIn this block you have earned $%.2f!\n\nTotal earnings = $%.2f' % (earnings_bl/100.0,earnings/100.0)
        
        take_break.setText(break_text)
        # ------Prepare to start Routine "break"-------
        t = 0
        breakClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        # update component parameters for each repeat
        ok1 = event.BuilderKeyResponse()
        # keep track of which components have finished
        breakComponents = [take_break, ok1]
        for thisComponent in breakComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
    
        # -------Start Routine "break"-------
        while continueRoutine:
            # get current time
            t = breakClock.getTime()
            tg = globalClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            
            # *take_break* updates
            if t >= 0.0 and take_break.status == NOT_STARTED:
                # keep track of start time/frame for later
                take_break.tStart = t
                take_break.tgStart = tg
                take_break.frameNStart = frameN  # exact frame index
                take_break.setAutoDraw(True)
            
            # *ok1* updates
            if t >= 0.0 and ok1.status == NOT_STARTED:
                # keep track of start time/frame for later
                ok1.tStart = t
                ok1.tgStart = tg
                ok1.frameNStart = frameN  # exact frame index
                ok1.status = STARTED
                # keyboard checking is just starting
                win.callOnFlip(ok1.clock.reset)  # t=0 on next screen flip
                event.clearEvents(eventType='keyboard')
            if ok1.status == STARTED:
                theseKeys = event.getKeys(keyList=['left', 'down', 'right'])
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    ok1.keys = theseKeys[-1]  # just the last key pressed
                    ok1.rt = ok1.clock.getTime()
                    # a response ends the routine
                    continueRoutine = False
                    
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in breakComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                win.close()
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
    
        # -------Ending Routine "break"-------
        for thisComponent in breakComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        if tr_nb == 30: #move to second block
            block_nb = block_nb + 1
            earnings_bl = 0
            
            #show block condition (stable or volatile) before start of the block
#            condRoutine = True
#            condmsg = 'This will be a VOLATILE block\n\n(the valuable token will change many times during the block)\n\nPress right arrow key to start.'
#            cond = visual.TextStim(win, text = condmsg, color = 'white', units = 'pix', pos=(0, 0), height=40)
#            event.clearEvents(eventType='keyboard')
#            while condRoutine == True:
#                cond.draw()
#                win.flip()   
#                startKey = event.getKeys(keyList='right')
#                if "escape" in startKey:
#                    endExpNow = True
#                if len(startKey) > 0:  # at least one key from the list was pressed
#                    condRoutine = False
#                if endExpNow or event.getKeys(keyList=["escape"]):
#                    core.quit()                  
            # the Routine "break" was not non-slip safe, so reset the non-slip timer
#            routineTimer.reset()
#            globalClock.reset()
            
            #show a fixation cross for 500ms before start of each block
            fixx = visual.TextStim(win, text = '+', color = 'white', units = 'pix',pos=(0, 0), height=50)
            fixx.draw()
            win.flip()
            core.wait(1)
           
    tr_nb = tr_nb + 1
    thisExp.nextEntry()
    
prop_corr = nb_corr*100/(nb_tr/3)
print 'Percent correct = %.2f' %(prop_corr)
print 'Earnings main task practice run = $%.2f' %(earnings/100) #print earnings in output window    

#___________________________________________________________
#_____________SAVE DATA IN VARIOUS FILES____________________
#___________________________________________________________

# get names of stimulus parameters
if trials.trialList in ([], [None], None):
    params = []
else:
    params = trials.trialList[0].keys()

# save data for this loop
trials.saveAsExcel(filename + '_final.xlsx', sheetName='trials',
    stimOut=params, dataOut=['all_raw'])
thisExp.saveAsWideText(filename+'_backup.csv')
thisExp.saveAsPickle(filename+'_data')
       
wb = open_workbook(filename + '_final.xlsx')
s = wb.sheet_by_index(0)
d = {}
for col in range(s.ncols):
    key_name = s.cell(0,col).value
    key_vals = s.col_values(col,start_rowx=1,end_rowx=nb_tr+1)
    d[key_name] = key_vals
wb_fin = xlsxwriter.Workbook(filename+'_final_ord.xlsx', {'strings_to_numbers': True})
ws_fin = wb_fin.add_worksheet()
ws_fin.write(0,0,"index")
ws_fin.write_column(1,0,tuple([int(num) for num in d['order']]))
ws_fin.write(0,1,"trialNb")
ws_fin.write_column(1,1,tuple([int(num) for num in d['trialNb']]))
ws_fin.write(0,2,"runPos")
ws_fin.write_column(1,2,tuple([int(num) for num in d['runPos']]))
ws_fin.write(0,3,"runNb")
ws_fin.write_column(1,3,tuple([int(num) for num in d['runNb']]))
ws_fin.write(0,4,"trType")
ws_fin.write_column(1,4,tuple([int(num) for num in d['trType']]))
ws_fin.write(0,5,"goalToken")
ws_fin.write_column(1,5,tuple([int(num) for num in d['goalToken']]))
ws_fin.write(0,6,"tvCond")
ws_fin.write_column(1,6,tuple([int(num) for num in d['tvCond']]))
ws_fin.write(0,7,"buCond")
ws_fin.write_column(1,7,tuple([int(num) for num in d['buCond']]))
ws_fin.write(0,8,"unavAct")
ws_fin.write_column(1,8,tuple([int(num) for num in d['unavAct']]))
ws_fin.write(0,9,"corrAct") 
ws_fin.write_column(1,9,tuple([int(num) for num in d['corrAct']]))
ws_fin.write(0,10,"bestAct") 
ws_fin.write_column(1,10,tuple([int(num) for num in d['bestAct']]))
ws_fin.write(0,11,"vertOrd") 
ws_fin.write_column(1,11,tuple([int(num) for num in d['vertOrd']]))
ws_fin.write(0,12,"horizOrd") 
ws_fin.write_column(1,12,tuple([int(num) for num in d['horizOrd']]))

ws_fin.write(0,13,"obs_play_onset") 
ws_fin.write_column(1,13,tuple([float(num) for num in d['obsplay_onset_raw']]))
ws_fin.write(0,14,"sm_onset") 
ws_fin.write_column(1,14,tuple([float(num) for num in d['sm_onset_raw']]))

ws_fin.write(0,15,"fixation1_onset")
fix1_ons = []
for num in d['fixation1_onset_raw']:
    if num == u'':
        fix1_ons.append('')
    else:
        fix1_ons.append(float(num))
ws_fin.write_column(1,15,tuple(fix1_ons))

ws_fin.write(0,16,"choose_onset")
ch_ons = []
for num in d['choose_onset_raw']:
    if num == u'':
        ch_ons.append('')
    else:
        ch_ons.append(float(num))
ws_fin.write_column(1,16,tuple(ch_ons))

ws_fin.write(0,17,"choice")
choicek = []
for num in d['choice_raw']:
    if num == u"'left'":
        choicek.append(1)
    elif num == u"'down'":
        choicek.append(2)
    elif num == u"'right'":
        choicek.append(3)
    elif num == u"'--'":
        choicek.append('')
ws_fin.write_column(1,17,tuple(choicek))

ws_fin.write(0,18,"choiceRT") 
chRT = []
for num in d['choiceRT_raw']:
    if num == u'':
        chRT.append('')
    else:
        chRT.append(float(num))
ws_fin.write_column(1,18,tuple(chRT))

ws_fin.write(0,19,"isCorr")
iscorr = []
for num in d['isCorr_raw']:
    if num == u'':
        iscorr.append('')
    else:
        iscorr.append(int(num))
ws_fin.write_column(1,19,tuple(iscorr))

ws_fin.write(0,20,"resp_onset")
resp_ons = []
for num in d['resp_onset_raw']:
    if num == u'':
        resp_ons.append('')
    else:
        resp_ons.append(float(num))
ws_fin.write_column(1,20,tuple(resp_ons))

ws_fin.write(0,21,"ch_fb_onset")
chfb_ons = []
for num in d['ch_fb_onset_raw']:
    if num == u'':
        chfb_ons.append('')
    else:
        chfb_ons.append(float(num))
ws_fin.write_column(1,21,tuple(chfb_ons))

ws_fin.write(0,22,"video_onset")
video_ons = []
for num in d['video_onset_raw']:
    if num == u'':
        video_ons.append('')
    else:
        video_ons.append(float(num))
ws_fin.write_column(1,22,tuple(video_ons))

ws_fin.write(0,23,"video_nb")
video_n = []
for num in d['video_nb_raw']:
    if num == u'':
        video_n.append('')
    else:
        video_n.append(int(num))
ws_fin.write_column(1,23,tuple(video_n))

ws_fin.write(0,24,"fixation2_onset")
ws_fin.write(0,25,"pGreen")
ws_fin.write(0,26,"pRed")
ws_fin.write(0,27,"pBlue")
ws_fin.write(0,28,"pGoal")
ws_fin.write(0,29,"randn")
ws_fin.write(0,30,"isGoal")
ws_fin.write(0,31,"tokenShown")
ws_fin.write(0,32,"token_onset")
fix2 = []
pgr = []
pre = []
pbl = []
pg = []
rn = []
ig = []
ts = []
to = []
for num in range(60):
    if d['miss_raw'][num] == 0:
        fix2.append(float(d['fixation2_onset_raw'][num]))
        pgr.append(round(d['pGreen_raw'][num],2))
        pre.append(round(d['pRed_raw'][num],2))
        pbl.append(round(d['pBlue_raw'][num],2))
        pg.append(round(d['pGoal_raw'][num],2))
        rn.append(float(d['randn_raw'][num]))
        ig.append(int(d['isGoal_raw'][num]))
        ts.append(int(d['tokenShown_raw'][num]))
        to.append(float(d['token_onset_raw'][num]))
    else:
        fix2.append('')
        pgr.append('')
        pre.append('')
        pbl.append('')
        pg.append('')
        rn.append('')
        ig.append('')
        ts.append('')
        to.append('')
ws_fin.write_column(1,24,tuple(fix2))
ws_fin.write_column(1,25,tuple(pgr))
ws_fin.write_column(1,26,tuple(pre))
ws_fin.write_column(1,27,tuple(pbl))
ws_fin.write_column(1,28,tuple(pg))
ws_fin.write_column(1,29,tuple(rn))
ws_fin.write_column(1,30,tuple(ig))
ws_fin.write_column(1,31,tuple(ts))
ws_fin.write_column(1,32,tuple(to))

ws_fin.write(0,33,"outcome") 
ws_fin.write_column(1,33,tuple([int(num) for num in d['outcome_raw']]))
ws_fin.write(0,34,"fixation_iti_onset") 
ws_fin.write_column(1,34,tuple([float(num) for num in d['fixationiti_onset_raw']]))
ws_fin.write(0,35,"miss")
ws_fin.write_column(1,35,tuple([int(num) for num in d['miss_raw']]))
if 'missed_onset_raw' in d:
    ws_fin.write(0,36,"missed_onset")
    miss_ons = []
    for num in d['missed_onset_raw']:
        if num == u'':
            miss_ons.append('')
        else:
            miss_ons.append(float(num))
    ws_fin.write_column(1,36,tuple(miss_ons))

wb_fin.close()

#___________________________________________________________
#_________________CLOSE EVERYTHING DOWN_____________________
#___________________________________________________________

logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
