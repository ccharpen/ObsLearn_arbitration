#Main task adapted for fMRI
#Changes from behavioral version: [to add]
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
import time #helps to give the os a tiny break during video presentation
import csv
import xlsxwriter
from xlrd import open_workbook

#Determine whether eye tracker should be used (swith to 0 if not)
var_eyetrack = 0

if var_eyetrack == 1:
    import pylink #to communicate with eye tracker
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy #to set up graphics interface for eye tracker calibration


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
#_thisDir = 'C:\Users\Caroline\Dropbox\Post-doc Projects\Observational learning\Task_fmri'
os.chdir(_thisDir)
stim_dir = _thisDir + os.sep + 'stimuli'
vid_dir = _thisDir + os.sep + 'videos'

# Add functions
from functions import *

#___________________________________________________________
#_________START CODE - SET UP + INITIALIZATION _____________
#___________________________________________________________

# Store info about the run number
expName = 'ObsLearnfmri'  # from the Builder filename that created this script
expInfo = {u'participant': u'', u'run': u''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename     = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
filename_run = _thisDir + os.sep + u'data/%s_%s_run%s' % (expInfo['participant'], expName, expInfo['run'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='', extraInfo=expInfo, runtimeInfo=None,
    originPath=None, savePickle=True, saveWideText=True, dataFileName=filename_run)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Setup the Window
dispWidth = 1280
dispHeight = 1024
win = visual.Window(size=(dispWidth, dispHeight), fullscr=True, screen=1, allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb', blendMode='avg', useFBO=False, units='pix')
#win = visual.Window(size=(dispWidth, dispHeight), fullscr=True, allowGUI=False, 
#    monitor='testMonitor', color=[0.65,0.65,0.65], colorSpace='rgb', useFBO=False, units='pix')

# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
#    if var_eyetrack == 1:
#        frameDur = 1.0 / 75.0  # frame rate is 75Hz on Eyelink display computer, double check for scanner computer
#    else:
#        frameDur = 1.0 / 60.0  # could not measure, so guess

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
#_________CREATE TRIAL FILE_________________________________
#___________________________________________________________

fnamepy = 'data/%s_data_alltr.npy' % (expInfo['participant'])
subj = str(expInfo['participant'])
if int(expInfo['run']) == 1:
    nb_tr = 240 
    version = int(expInfo['participant'])%3+1 #randomise which version of trial list to use
    #create trial list file by counterbalancing run order
    tr_list_all = CreateTrialFile_fmri(nb_tr,subj)    
    #add empty labels for storage later
    tr_list_all2 = {'obsplay_onset':[], 'sm_onset':[], 'fixation1_onset':[], 'choose_onset':[], 'choice':[], 'choiceRT':[], 'isCorr': [],
                    'resp_onset':[], 'ch_fb_onset':[], 'video_onset':[], 'video_nb':[], 'fixation2_onset':[], 
                    'pGreen':[], 'pRed':[], 'pBlue':[], 'pGoal':[], 'randn':[], 'isGoal':[], 'tokenShown':[],  
                    'token_onset':[], 'outcome':[],  'fixationiti_onset':[], 'miss':[], 'missed_onset':[]}
    tr_list_all.update(tr_list_all2)
    np.save(fnamepy,tr_list_all)
else:
    tr_list_all = np.load(fnamepy).item()

#create variable to calculate earnings and proportion correct
earnings_bl = 0
nb_corr = 0
tr_nb = 1
tr_obs_nb = 1

#create vectors of jittered durations for fixation crosses and ITI
jit_fix1 = np.repeat([1.0,2.0,3.0,4.0],5).tolist() #fixation 1 happens 20 times (observe trials only)
jit_fix1 = np.random.permutation(jit_fix1).tolist()

jit_fix2 = np.repeat([1.0,2.0,3.0,4.0],7).tolist() + [1.0, 4.0] #fixation 2 happens 30 times
jit_fix2 = np.random.permutation(jit_fix2).tolist()

jit_iti = np.repeat([1.0,2.0,3.0,4.0,5.0],6).tolist() #iti happens 30 times
jit_iti = np.random.permutation(jit_iti).tolist()

#___________________________________________________________
#_____________SET UP EYE TRACKER____________________________
#___________________________________________________________

if var_eyetrack == 1:
    # set up a link to the tracker
    tk = pylink.EyeLink('100.1.1.1') #  or None (for dummy mode)
    
    #get tracker version
    vs = tk.getTrackerVersion()
    
    #open graphics window
    genv = EyeLinkCoreGraphicsPsychoPy(tk, win)
    pylink.openGraphicsEx(genv)
    win.mouseVisible = False
    
    #open edf file
    edfname = 'P%s_%s.edf' % (str(expInfo['participant']), str(expInfo['run']))
    tk.openDataFile(edfname)
    pylink.flushGetkeyQueue()
    tk.setOfflineMode();
    
    # this line helps to make sure the tracker gets the correct display dimensions
    tk.sendCommand("screen_pixel_coords =  0 0 %d %d" %(dispWidth - 1, dispHeight - 1))
    tk.sendMessage("DISPLAY_COORDS  0 0 %d %d" %(dispWidth - 1, dispHeight - 1))

    # set EDF file contents 
    tk.sendCommand("file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON")
    tk.sendCommand("file_sample_data  = LEFT,RIGHT,GAZE,AREA,GAZERES,STATUS")
    # set link data (used for gaze cursor) 
    tk.sendCommand("link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON")
    tk.sendCommand("link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS")
    
    #set calibration type (here 9 targets)
    calst = 'HV{}'.format(5)
    tk.setCalibrationType(calst)
    pylink.setCalibrationSounds("off", "off", "off") #deactivate sounds
    pylink.setDriftCorrectSounds("off", "off", "off")
    
    instt = visual.TextStim(win, text = 'Please fixate cross below during eye tracker set up', color = 'white', units = 'pix', pos = (0,300))
    fixx = visual.TextStim(win, text = '+', color = 'white', units = 'pix',pos=(0, 0), height=50)
    instt.draw()
    fixx.draw()
    win.flip()
    keyp = event.waitKeys()

    # set up the camera and do calibration/validation
    tk.doTrackerSetup()
    print 'Calibration and validation OK'
    
    #do a drift check
    #try:
    #    err = tk.doDriftCorrect(dispWidth/2, dispHeight/2, 1,1)
    #    print 'Drift correction OK'
    #except:
    #    tk.doTrackerSetup()
    # take the tracker offline
    tk.setOfflineMode()
    pylink.pumpDelay(100)
    
    # reset timer after eye tracking set up
    routineTimer.reset()
    globalClock.reset()

#___________________________________________________________
#_________WAIT FOR TRIGGER__________________________________
#___________________________________________________________

event.Mouse(visible=False)
win.setColor([-1,-1,-1], 'rgb')
win.flip()

fname = 'trial_lists/trial_list_fMRI_Sub%s_run%s.xlsx' % (subj,expInfo['run'])

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=1, method='sequential', extraInfo=expInfo, 
    originPath=-1, trialList=data.importConditions(fname), seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial.keys():
        exec(paramName + '= thisTrial.' + paramName)

#extract whether this is a stable or volatile block
vol_cond = trials.trialList[1].tvCond
WaitForScanner(win,endExpNow,vol_cond)

# the Routine was not non-slip safe, so reset the non-slip timer
routineTimer.reset()
globalClock.reset()

#___________________________________________________________
#___________MAIN TRIAL LOOP_________________________________
#___________________________________________________________

#start eye tracker recording
if var_eyetrack == 1:
    error = tk.startRecording(1,1,1,1)
    pylink.pumpDelay(100) # wait for 100 ms to make sure data of interest is recorded
    tk.sendMessage('SYNCTIME %d' % pylink.currentTime())
    
#show a fixation cross for 1s before start of first block
fixx = visual.TextStim(win, text = '+', color = 'white', units = 'pix',pos=(0, 0), height=50)
fixx.draw()
win.flip()
core.wait(1)

for thisTrial in trials:
    currentLoop = trials
    miss = 0
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec(paramName + '= thisTrial.' + paramName)
    
    #record trial ID for eye tracker
    if var_eyetrack == 1:
        tk.sendMessage('TRIALID %s' % str(tr_nb))
    
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
        k_list = ['2','3']
    elif unavAct == 2:
        sm1n = '%s%i_%s_a.png' % (lc,vertOrd, bu)
        sm2n = '%s%i_%s_ua.png' % (mc,vertOrd, bu)
        sm3n = '%s%i_%s_a.png' % (rc,vertOrd, bu)
        k_list = ['1','3']
    elif unavAct == 3:
        sm1n = '%s%i_%s_a.png' % (lc,vertOrd, bu)
        sm2n = '%s%i_%s_a.png' % (mc,vertOrd, bu)
        sm3n = '%s%i_%s_ua.png' % (rc,vertOrd, bu)
        k_list = ['1','2']
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
            
            if var_eyetrack == 1:
                if trType == 1: #observe
                    tk.sendMessage('TTL=1') #time stamped message (TTL numeric code to use Julien functions)
                elif trType == 2: #play
                    tk.sendMessage('TTL=2')
                
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
                
                if var_eyetrack == 1: #slot machine timestamp - observe trials
                    if corrAct == 1:
                        tk.sendMessage('TTL=3')
                    elif corrAct == 2:
                        tk.sendMessage('TTL=4')
                    elif corrAct == 3:
                        tk.sendMessage('TTL=5')
                    if bestAct == 1:
                        tk.sendMessage('TTL=9')
                    elif bestAct == 2:
                        tk.sendMessage('TTL=10')
                    elif bestAct == 3:
                        tk.sendMessage('TTL=11')
                    if bestAct == corrAct: #best action available
                        tk.sendMessage('TTL=15')
                    else:
                        tk.sendMessage('TTL=16')
            
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=19') #time stamped message (TTL numeric code to use Julien functions)
                    
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
                
                if var_eyetrack == 1: #video onset timestamp
                    if corrAct == 1:
                        tk.sendMessage('TTL=20')
                    elif corrAct == 2:
                        tk.sendMessage('TTL=21')
                    elif corrAct == 3:
                        tk.sendMessage('TTL=22')
                
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
                
                if var_eyetrack == 1: #slot machine timestamp - play trials
                    if corrAct == 1:
                        tk.sendMessage('TTL=6')
                    elif corrAct == 2:
                        tk.sendMessage('TTL=7')
                    elif corrAct == 3:
                        tk.sendMessage('TTL=8')
                    if bestAct == 1:
                        tk.sendMessage('TTL=12')
                    elif bestAct == 2:
                        tk.sendMessage('TTL=13')
                    elif bestAct == 3:
                        tk.sendMessage('TTL=14')
                    if bestAct == corrAct: #best action available
                        tk.sendMessage('TTL=17')
                    else:
                        tk.sendMessage('TTL=18')
               
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
                
                if var_eyetrack == 1: #choose message timestamp
                    tk.sendMessage('TTL=23')
                        
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
                    # was this 'correct'?
                    if (key_choice.keys == str(corrAct)) or (key_choice.keys == corrAct):
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
    tr_list_all['obsplay_onset'].append(obs_play.tgStart)
    trials.addData('sm_onset',sm1.tgStart)
    tr_list_all['sm_onset'].append(sm1.tgStart)    
    trials.addData('miss',miss)
    tr_list_all['miss'].append(miss)
    if trType == 2: #play 
        trials.addData('choose_onset',choose.tgStart)
        tr_list_all['choose_onset'].append(choose.tgStart)
        if miss == 0:  # we had a response
            trials.addData('choice',key_choice.keys)
            tr_list_all['choice'].append(key_choice.keys)
            trials.addData('isCorr', key_choice.corr)
            tr_list_all['isCorr'].append(key_choice.corr)
            trials.addData('choiceRT', key_choice.rt)
            tr_list_all['choiceRT'].append(key_choice.rt)
            trials.addData('resp_onset',key_choice.onset)
            tr_list_all['resp_onset'].append(key_choice.onset)
        else:
            tr_list_all['choice'].append('')
            tr_list_all['isCorr'].append('')
            tr_list_all['choiceRT'].append('')
            tr_list_all['resp_onset'].append('')
        tr_list_all['fixation1_onset'].append('')
        tr_list_all['video_onset'].append('')   
        tr_list_all['video_nb'].append('')         
    elif trType == 1: #observe
        trials.addData('fixation1_onset',fixation1.tgStart)
        tr_list_all['fixation1_onset'].append(fixation1.tgStart)
        trials.addData('video_nb',nvid)
        tr_list_all['video_nb'].append(nvid)
        trials.addData('video_onset',video.tgStart)
        tr_list_all['video_onset'].append(video.tgStart)
        tr_list_all['choose_onset'].append('')
        tr_list_all['choice'].append('')
        tr_list_all['isCorr'].append('')
        tr_list_all['choiceRT'].append('')
        tr_list_all['resp_onset'].append('')
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
        ch = int(key_choice.keys)
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
            if key_choice.keys == '1':
                sm1chn = '%s%i_%s_p.png' % (lc,vertOrd, bu)
                sm2chn = sm2n
                sm3chn = sm3n
            elif key_choice.keys == '2':
                sm1chn = sm1n
                sm2chn = '%s%i_%s_p.png' % (mc,vertOrd, bu)
                sm3chn = sm3n
            elif key_choice.keys == '3':
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
        n = ''
        tokenShown = ''
        tsFile = ''
        P_green = ''
        P_red = ''
        P_blue = ''
        isgoal = ''
        ov = 0
        P_goal = ''
    
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
                
                if var_eyetrack == 1: #choice feedback timestamp - play trials
                    if int(key_choice.keys) == 1:
                        tk.sendMessage('TTL=24')
                    elif int(key_choice.keys) == 2:
                        tk.sendMessage('TTL=25')
                    elif int(key_choice.keys) == 3:
                        tk.sendMessage('TTL=26')
                        
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=28') #time stamped message (TTL numeric code to use Julien functions)
                    
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
                
                if var_eyetrack == 1: #token onset timestamps
                    if tokenShown == 1: #green
                        tk.sendMessage('TTL=32')
                    elif tokenShown == 2: #red
                        tk.sendMessage('TTL=33')
                    elif tokenShown == 3: #blue
                        tk.sendMessage('TTL=34')
                    if isgoal == 1:
                        tk.sendMessage('TTL=37')
                        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5): #expected (high proba) token was shown
                            tk.sendMessage('TTL=41')
                            tk.sendMessage('TTL=47')
                        else: 
                            tk.sendMessage('TTL=42')
                            tk.sendMessage('TTL=48')
                    elif isgoal == 0:
                        tk.sendMessage('TTL=38')
                        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5): #expected (60% proba) token was shown
                            tk.sendMessage('TTL=41')
                            tk.sendMessage('TTL=49')
                        else: 
                            tk.sendMessage('TTL=42')
                            tk.sendMessage('TTL=50')
                    
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=51') #time stamped message (TTL numeric code to use Julien functions)
                    
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=27') #time stamped message (TTL numeric code to use Julien functions)
                    
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
                
                if var_eyetrack == 1: #token onset timestamps
                    if tokenShown == 1: #green
                        tk.sendMessage('TTL=29')
                    elif tokenShown == 2: #red
                        tk.sendMessage('TTL=30')
                    elif tokenShown == 3: #blue
                        tk.sendMessage('TTL=31')
                    if isgoal == 1:
                        tk.sendMessage('TTL=35')
                        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5): #expected token was shown
                            tk.sendMessage('TTL=39')
                            tk.sendMessage('TTL=43')
                        else: 
                            tk.sendMessage('TTL=40')
                            tk.sendMessage('TTL=44')
                    elif isgoal == 0:
                        tk.sendMessage('TTL=36')
                        if (buCond == 1 and n <= 0.75) or (buCond == 2 and n <= 0.5): #expected token was shown
                            tk.sendMessage('TTL=39')
                            tk.sendMessage('TTL=45')
                        else: 
                            tk.sendMessage('TTL=40')
                            tk.sendMessage('TTL=46')
                    
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=51') #time stamped message (TTL numeric code to use Julien functions)
                    
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
                
                if var_eyetrack == 1: #missed choice timestamp
                    tk.sendMessage('TTL=52')
                    
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
                
                if var_eyetrack == 1:
                    tk.sendMessage('TTL=51') #time stamped message (TTL numeric code to use Julien functions)
                    
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
    trials.addData('pGreen', P_green)
    tr_list_all['pGreen'].append(P_green)
    trials.addData('pRed', P_red)
    tr_list_all['pRed'].append(P_red)
    trials.addData('pBlue', P_blue)
    tr_list_all['pBlue'].append(P_blue)
    trials.addData('pGoal', P_goal)
    tr_list_all['pGoal'].append(P_goal)
    trials.addData('randn', n)
    tr_list_all['randn'].append(n)
    trials.addData('tokenShown', tokenShown)
    tr_list_all['tokenShown'].append(tokenShown)
    trials.addData('isGoal', isgoal)
    tr_list_all['isGoal'].append(isgoal)
    trials.addData('outcome', ov)
    tr_list_all['outcome'].append(ov)
    trials.addData('fixationiti_onset', fixation_iti.tgStart)
    tr_list_all['fixationiti_onset'].append(fixation_iti.tgStart)
    if miss == 0:
        trials.addData('fixation2_onset', fixation2.tgStart)
        tr_list_all['fixation2_onset'].append(fixation2.tgStart)
        trials.addData('token_onset', token.tgStart)
        tr_list_all['token_onset'].append(token.tgStart)
        tr_list_all['missed_onset'].append('')
    else:
        tr_list_all['fixation2_onset'].append('')
        tr_list_all['token_onset'].append('')
        trials.addData('missed_onset', missed.tgStart)
        tr_list_all['missed_onset'].append(missed.tgStart)
        
    earnings_bl = earnings_bl + ov
    if trType == 2 and miss == 0:
        nb_corr = nb_corr + key_choice.corr
        trials.addData('ch_fb_onset',sm1_ch.tgStart)
        tr_list_all['ch_fb_onset'].append(sm1_ch.tgStart)
    else: 
        tr_list_all['ch_fb_onset'].append('')
    
    if trType == 1:
        tr_obs_nb = tr_obs_nb + 1 #only update observe trial number on observe trials
    
    if var_eyetrack == 1:
        tk.sendMessage('TRIAL OK')
    
    tr_nb = tr_nb + 1
    thisExp.nextEntry()
    np.save(fnamepy,tr_list_all)
    
    #clear variables to make sure they are re-computed on every trial
    del ch
    del tokenShown
    del tsFile
    del P_green
    del P_red
    del P_blue
    del P_goal
    del isgoal
    del ov
    
    
#_________________________________________________________________________
#______END OF BLOCK - break for bl 1 to 7, end of task for bl 8___________
#_________________________________________________________________________
    
#stop recording eye tracking at the end of the block
if var_eyetrack == 1:
    tk.stopRecording()
    tk.setOfflineMode()

#SAVE DATA FOR EACH BLOCK
thisExp.saveAsWideText(filename_run+'_backup.csv')
thisExp.saveAsPickle(filename_run+'_data')
    
if int(expInfo['run']) <= 7:
    break_text = 'End of block %i/8.\n\nIn this block you have earned $%.2f!\n\nPlease keep still until the scanner stops (up to 30s)!' %(int(expInfo['run']),earnings_bl/100.0)
    take_break.setText(break_text)
    # ------Prepare to start Routine "break"-------
    t = 0
    breakClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(10.0000)
    # keep track of which components have finished
    breakComponents = [take_break]
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
        frameRemains = 0.0 + 10.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
        if take_break.status == STARTED and t >= frameRemains:
            take_break.setAutoDraw(False)
                    
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
        
            
elif int(expInfo['run']) == 8:
    
    # get names of stimulus parameters
#    if trials.trialList in ([], [None], None):
#        params = []
#    else:
#        params = trials.trialList[0].keys()
#    # save data for this loop
#    trials.saveAsExcel(filename + '_final.xlsx', sheetName='trials',
#        stimOut=params, dataOut=['all_raw'])
    corr_list = [cor for cor in tr_list_all['isCorr'] if type(cor)==int]
    prop_corr = sum(corr_list)*100/len(corr_list)
    earnings = sum(tr_list_all['outcome'])
    print 'Total earnings = $%.2f' %(earnings/100) #print earnings in output window
    print 'Percent correct = %.2f' %(prop_corr)
    
    # ------Prepare to start Routine "thanks"-------
    t = 0
    thanksClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(10.000000)
    # show cumulated earning
    thanksText = 'End of the task.\n\nPlease keep still until the scanner stops (up to 30s)!\n\nTotal earnings = $%.2f' % (earnings/100.0)
    thanksMsg.setText(thanksText)
    # keep track of which components have finished
    thanksComponents = [thanksMsg]
    for thisComponent in thanksComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "thanks"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = thanksClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        
        # *thanksMsg* updates
        if t >= 0.0 and thanksMsg.status == NOT_STARTED:
            # keep track of start time/frame for later
            thanksMsg.tStart = t
            thanksMsg.frameNStart = frameN  # exact frame index
            thanksMsg.setAutoDraw(True)
        frameRemains = 0.0 + 10.0 - win.monitorFramePeriod * 0.75  # most of one frame period left
        if thanksMsg.status == STARTED and t >= frameRemains:
            thanksMsg.setAutoDraw(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanksComponents:
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
    
    # -------Ending Routine "thanks"-------
    for thisComponent in thanksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)

    #save data in final file (can probably be done from tr_list_all dictionary)    
    comb_vals = [range(240), tr_list_all['trialNb'], tr_list_all['runPos'], tr_list_all['runNb'], tr_list_all['trType'], tr_list_all['goalToken'],
                 tr_list_all['tvCond'], tr_list_all['buCond'], tr_list_all['unavAct'], tr_list_all['corrAct'], tr_list_all['bestAct'],
                 tr_list_all['vertOrd'], tr_list_all['horizOrd'], tr_list_all['obsplay_onset'], tr_list_all['sm_onset'], tr_list_all['fixation1_onset'],
                 tr_list_all['choose_onset'], tr_list_all['choice'], tr_list_all['choiceRT'], tr_list_all['isCorr'], tr_list_all['resp_onset'], 
                 tr_list_all['ch_fb_onset'], tr_list_all['video_onset'], tr_list_all['video_nb'], tr_list_all['fixation2_onset'], 
                 tr_list_all['pGreen'], tr_list_all['pRed'], tr_list_all['pBlue'], tr_list_all['pGoal'], tr_list_all['randn'], tr_list_all['isGoal'],   
                 tr_list_all['tokenShown'], tr_list_all['token_onset'], tr_list_all['outcome'], tr_list_all['fixationiti_onset'],
                 tr_list_all['miss'], tr_list_all['missed_onset']]     
        
    final_trdata = np.asarray(comb_vals).transpose() 
    wb_fin = xlsxwriter.Workbook(filename+'_final_ord.xlsx', {'strings_to_numbers': True})
    ws_fin = wb_fin.add_worksheet()
    #write column titles
    ws_fin.write(0,0,"index")
    ws_fin.write(0,1,"trialNb")
    ws_fin.write(0,2,"runPos")
    ws_fin.write(0,3,"runNb")
    ws_fin.write(0,4,"trType")
    ws_fin.write(0,5,"goalToken")    
    ws_fin.write(0,6,"tvCond")
    ws_fin.write(0,7,"buCond")
    ws_fin.write(0,8,"unavAct")
    ws_fin.write(0,9,"corrAct") 
    ws_fin.write(0,10,"bestAct") 
    ws_fin.write(0,11,"vertOrd")   
    ws_fin.write(0,12,"horizOrd") 
    ws_fin.write(0,13,"obsplay_onset") 
    ws_fin.write(0,14,"sm_onset") 
    ws_fin.write(0,15,"fixation1_onset")    
    ws_fin.write(0,16,"choose_onset") 
    ws_fin.write(0,17,"choice")
    ws_fin.write(0,18,"choiceRT")
    ws_fin.write(0,19,"isCorr")
    ws_fin.write(0,20,"resp_onset")
    ws_fin.write(0,21,"ch_fb_onset") 
    ws_fin.write(0,22,"video_onset") 
    ws_fin.write(0,23,"video_nb") 
    ws_fin.write(0,24,"fixation2_onset")
    ws_fin.write(0,25,"pGreen")
    ws_fin.write(0,26,"pRed") 
    ws_fin.write(0,27,"pBlue") 
    ws_fin.write(0,28,"pGoal") 
    ws_fin.write(0,29,"randn") 
    ws_fin.write(0,30,"isGoal") 
    ws_fin.write(0,31,"tokenShown") 
    ws_fin.write(0,32,"token_onset") 
    ws_fin.write(0,33,"outcome") 
    ws_fin.write(0,34,"fixation_iti_onset")
    ws_fin.write(0,35,"miss") 
    ws_fin.write(0,36,"missed_onset") 
        
    size = np.shape(final_trdata)
    rows = int(size[0])
    cols = int(size[1])
    for i in range(rows):
        for j in range(cols):
            ws_fin.write(i+1,j,final_trdata[i][j])
        
    wb_fin.close()


#___________________________________________________________
#_________________CLOSE EVERYTHING DOWN_____________________
#___________________________________________________________

if var_eyetrack == 1:
    tk.closeDataFile()
    tk.receiveDataFile(edfname, 'eye_data/' + edfname)
    tk.close()
    pylink.closeGraphics()

logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
