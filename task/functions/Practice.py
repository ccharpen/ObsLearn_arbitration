# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:45:21 2017

@author: Caroline Charpentier
"""

from __future__ import absolute_import, division
import sys  # to get file system encoding
from psychopy import locale_setup, gui, visual, core, data, event, logging, sound
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import csv
import xlsxwriter
from xlrd import open_workbook

def Practice(expInfo,thisPrac,stim_dir,filenamePrac):

    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file
    
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
        
    # Setup the Window
    win = visual.Window(size=(1024, 768), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
        monitor='testMonitor', color='black', colorSpace='rgb', blendMode='avg', useFBO=False, units='pix')
            
    # Initialize components for Routine "trial"
    trialClock = core.Clock()
    fixation = visual.TextStim(win=win, name='fixation', text='+', font='Arial', pos=(0, 0), height=50, 
                               wrapWidth=None, ori=0, color='white', colorSpace='rgb', opacity=1, depth=0.0)
    goal_txt = visual.TextStim(win=win, name='goal_txt', text='Valuable token:', font='Arial', pos=(0, 180), 
                               height=40, wrapWidth=None, ori=0, color='white', colorSpace='rgb', opacity=1, depth=-1.0)
    goal_cir = visual.ImageStim(win=win, name='goal_cir', image='sin', mask=None, ori=0, pos=(0, 0), size=[120,120],
                                color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False,
                                texRes=128, interpolate=True, depth=-2.0)
    sm1 = visual.ImageStim(win=win, name='sm1', image='sin', mask=None, ori=0, pos=(-250, 0), size=[161,334],
                           color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False,
                           texRes=128, interpolate=True, depth=-5.0)
    sm2 = visual.ImageStim(win=win, name='sm2', image='sin', mask=None, ori=0, pos=(0, 0), size=[161,334],
                           color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False,
                           texRes=128, interpolate=True, depth=-5.0)
    sm3 = visual.ImageStim(win=win, name='sm3', image='sin', mask=None, ori=0, pos=(250, 0), size=[161,334],
                           color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False,
                           texRes=128, interpolate=True, depth=-5.0)
    choose = visual.TextStim(win=win, name='choose', text='CHOOSE', font='Arial', pos=(0,-300),
                         height=50, wrapWidth=None, ori=0, color='white', 
                         colorSpace='rgb', opacity=1, depth=0.0)
    
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
    token = visual.ImageStim(win=win, name='token', image='sin', mask=None, ori=0, pos=(0, 0), size=[120,120],
                             color=[1,1,1], colorSpace='rgb', opacity=1, flipHoriz=False, flipVert=False,
                             texRes=128, interpolate=True, depth=-1.0)
    reward = visual.TextStim(win=win, name='reward', text='default text', font=u'Arial', pos=(0, 0), 
                             height=60, wrapWidth=None, ori=0, color=u'white', colorSpace='rgb', 
                             opacity=1, depth=-2.0)
    
    # Initialize components for Routine "thanks"
    thanksClock = core.Clock()
    thanksMsg = visual.TextStim(win=win, name='thanksMsg', text='default text', font='Arial', pos=[0, 0], 
                                height=40, wrapWidth=None, ori=0, color=u'white', colorSpace='rgb', 
                                opacity=1, depth=0.0)
    
    # Create some handy timers
    globalClock = core.Clock()  # to track the time since experiment started
    routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 
    
    #choose trial file depending on participant number
    #p_nb = int(expInfo['participant'])
    #f_nb = (p_nb % 6) + 1
    #fname = 'trial_lists/practice_%s.xlsx' % (str(f_nb)) #change practice file based on subject number
    fname = 'trial_lists/practice_example.xlsx' #same practice for everyone
    #create variable to calculate earnings and proportion correct
    earnings = 0
    nb_corr = 0
    
    nb_tr = 27 #can be changed if necessary
    tr = 1 #keep track of trial number
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1, method='sequential', extraInfo=expInfo, 
        originPath=-1, trialList=data.importConditions(fname), seed=None, name='trials')
    thisPrac.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial.keys():
            exec(paramName + '= thisTrial.' + paramName)
    
    for thisTrial in trials:
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
        goal_cir.setImage(stim_dir + os.sep + tokenFile)
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
        if unavailAct == 1:
            sm1n = '%s%i_%s_ua.png' % (lc, vertOrd, bu)
            sm2n = '%s%i_%s_a.png' % (mc, vertOrd, bu)
            sm3n = '%s%i_%s_a.png' % (rc, vertOrd, bu)
        elif unavailAct == 2:
            sm1n = '%s%i_%s_a.png' % (lc, vertOrd, bu)
            sm2n = '%s%i_%s_ua.png' % (mc, vertOrd, bu)
            sm3n = '%s%i_%s_a.png' % (rc, vertOrd, bu)
        elif unavailAct == 3:
            sm1n = '%s%i_%s_a.png' % (lc, vertOrd, bu)
            sm2n = '%s%i_%s_a.png' % (mc, vertOrd, bu)
            sm3n = '%s%i_%s_ua.png' % (rc, vertOrd, bu)
        sm1.setImage(stim_dir + os.sep + sm1n)
        sm2.setImage(stim_dir + os.sep + sm2n)
        sm3.setImage(stim_dir + os.sep + sm3n)
        key_choice = event.BuilderKeyResponse()
        # keep track of which components have finished
        trialComponents = [fixation, goal_txt, goal_cir, sm1, sm2, sm3, choose, key_choice]
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "trial"-------
        while continueRoutine:
            # get current time
            t = trialClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            if t >= 0.0 and fixation.status == NOT_STARTED:
                # keep track of start time/frame for later
                fixation.tStart = t
                fixation.frameNStart = frameN  # exact frame index
                fixation.setAutoDraw(True)
            frameRemains = 0.0 + 1.0- win.monitorFramePeriod * 0.75  # most of one frame period left
            if fixation.status == STARTED and t >= frameRemains:
                fixation.setAutoDraw(False)
            
            # *goal_txt* updates
            if t >= 1 and goal_txt.status == NOT_STARTED:
                # keep track of start time/frame for later
                goal_txt.tStart = t
                goal_txt.frameNStart = frameN  # exact frame index
                goal_txt.setAutoDraw(True)
            frameRemains = 1 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left
            if goal_txt.status == STARTED and t >= frameRemains:
                goal_txt.setAutoDraw(False)
            
            # *goal_cir* updates
            if t >= 1 and goal_cir.status == NOT_STARTED:
                # keep track of start time/frame for later
                goal_cir.tStart = t
                goal_cir.frameNStart = frameN  # exact frame index
                goal_cir.setAutoDraw(True)
            frameRemains = 1 + 2 - win.monitorFramePeriod * 0.75  # most of one frame period left
            if goal_cir.status == STARTED and t >= frameRemains:
                goal_cir.setAutoDraw(False)
                    
            # *sm_ch* updates
            if t >= 3 and sm1.status == NOT_STARTED:
                # keep track of start time/frame for later
                sm1.tStart = t
                sm1.frameNStart = frameN  # exact frame index
                sm1.setAutoDraw(True)
                sm2.setAutoDraw(True)
                sm3.setAutoDraw(True)
            
            # *key_choice* updates                
            if t >= 5 and key_choice.status == NOT_STARTED:
                # keep track of start time/frame for later
                key_choice.tStart = t
                key_choice.frameNStart = frameN  # exact frame index
                key_choice.status = STARTED
                # keyboard checking is just starting
                win.callOnFlip(key_choice.clock.reset)  # t=0 on next screen flip
                event.clearEvents(eventType='keyboard')
                choose.tgStart = t
                
            if key_choice.status == STARTED:
                choose.setAutoDraw(True)
                theseKeys = event.getKeys(keyList=[availKey1,availKey2])
                
                # check for quit:
                if "escape" in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:  # at least one key was pressed
                    key_choice.keys = theseKeys[-1]  # just the last key pressed
                    key_choice.rt = key_choice.clock.getTime()
                    if key_choice.keys == 'left':
                        choice = 1
                    elif key_choice.keys == 'down':
                        choice = 2
                    elif key_choice.keys == 'right':
                        choice = 3
                    # was this 'correct'?
                    if choice == corrAct:
                        key_choice.corr = 1
                    else:
                        key_choice.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
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
        # check responses
        if key_choice.keys in ['', [], None]:  # No response was made
            key_choice.keys=None
            key_choice.corr = 0

        # store data for trials (TrialHandler)
        trials.addData('choice',choice)
        trials.addData('isCorr', key_choice.corr)
        if key_choice.keys != None:  # we had a response
            trials.addData('choiceRT', key_choice.rt)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "feedback"-------
        t = 0
        feedbackClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        routineTimer.add(4.500000)
        # update component parameters for each repeat
        if key_choice.keys == 'left':
            sm1chn = '%s%i_%s_p.png' % (lc, vertOrd, bu)
            sm2chn = sm2n
            sm3chn = sm3n
        elif key_choice.keys == 'down':
            sm1chn = sm1n
            sm2chn = '%s%i_%s_p.png' % (mc, vertOrd, bu)
            sm3chn = sm3n
        elif key_choice.keys == 'right':
            sm1chn = sm1n
            sm2chn = sm2n
            sm3chn = '%s%i_%s_p.png' % (rc, vertOrd, bu)
        sm1_ch.setImage(stim_dir + os.sep + sm1chn)
        sm2_ch.setImage(stim_dir + os.sep + sm2chn)
        sm3_ch.setImage(stim_dir + os.sep + sm3chn)
        
        n = random()
        ch = key_choice.keys
        if (ch  == 'left' and lc == 'G') or (ch == 'down' and mc == 'G') or (ch == 'right' and rc == 'G'): #green sm chosen
            if (bu == 'lbu' and n <= 0.75) or (bu == 'hbu' and n <= 0.5):
                tokenShown = 'green'
                tsFile = 'green.png'
            elif (bu == 'lbu' and n > 0.75 and n <= 0.95) or (bu == 'hbu' and n > 0.5 and n <= 0.8):
                tokenShown = 'red'
                tsFile = 'red.png'
            elif (bu == 'lbu' and n > 0.95) or (bu == 'hbu' and n > 0.8):
                tokenShown = 'blue'
                tsFile = 'blue.png'
            if bu == 'lbu':
                P_green = 0.75
                P_red = 0.2
                P_blue = 0.05
            elif bu == 'hbu':
                P_green = 0.5
                P_red = 0.3
                P_blue = 0.2
        elif (ch  == 'left' and lc == 'R') or (ch == 'down' and mc == 'R') or (ch == 'right' and rc == 'R'): #red sm chosen
            if (bu == 'lbu' and n <= 0.75) or (bu == 'hbu' and n <= 0.5):
                tokenShown = 'red'
                tsFile = 'red.png'
            elif (bu == 'lbu' and n > 0.75 and n <= 0.95) or (bu == 'hbu' and n > 0.5 and n <= 0.8):
                tokenShown = 'blue'
                tsFile = 'blue.png'
            elif (bu == 'lbu' and n > 0.95) or (bu == 'hbu' and n > 0.8):
                tokenShown = 'green'
                tsFile = 'green.png'
            if bu == 'lbu':
                P_red = 0.75
                P_blue = 0.2
                P_green = 0.05
            elif bu == 'hbu':
                P_red = 0.5
                P_blue = 0.3
                P_green = 0.2
        elif (ch  == 'left' and lc == 'B') or (ch == 'down' and mc == 'B') or (ch == 'right' and rc == 'B'): #blue sm chosen:
            if (bu == 'lbu' and n <= 0.75) or (bu == 'hbu' and n <= 0.5):
                tokenShown = 'blue'
                tsFile = 'blue.png'
            elif (bu == 'lbu' and n > 0.75 and n <= 0.95) or (bu == 'hbu' and n > 0.5 and n <= 0.8):
                tokenShown = 'green'
                tsFile = 'green.png'
            elif (bu == 'lbu' and n > 0.95) or (bu == 'hbu' and n > 0.8):
                tokenShown = 'red'
                tsFile = 'red.png'
            if bu == 'lbu':
                P_blue = 0.75
                P_green = 0.2
                P_red = 0.05
            elif bu == 'hbu':
                P_blue = 0.5
                P_green = 0.3
                P_red = 0.2
            
        if tokenShown == goalToken:
            outc = '+10c'
            ov = 10
            isgoal = 1
        else:
            outc = '0c'
            ov = 0
            isgoal = 0
        if goalToken == 'green':
            P_goal = P_green
        elif goalToken == 'red':
            P_goal = P_red
        elif goalToken == 'blue':
            P_goal = P_blue

        token.setImage(stim_dir + os.sep + tsFile)
        reward.setText(outc)
        # keep track of which components have finished
        feedbackComponents = [sm1_ch, sm2_ch, sm3_ch, token, reward]
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "feedback"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = feedbackClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ch_fb* updates
            if t >= 0.0 and sm1_ch.status == NOT_STARTED:
                # keep track of start time/frame for later
                sm1_ch.tStart = t
                sm1_ch.frameNStart = frameN  # exact frame index
                sm1_ch.setAutoDraw(True)
                sm2_ch.setAutoDraw(True)
                sm3_ch.setAutoDraw(True)
            frameRemains = 0.0 + 0.5- win.monitorFramePeriod * 0.75  # most of one frame period left
            if sm1_ch.status == STARTED and t >= frameRemains:
                sm1_ch.setAutoDraw(False)
                sm2_ch.setAutoDraw(False)
                sm3_ch.setAutoDraw(False)

            # *token* updates
            if t >= 0.5 and token.status == NOT_STARTED:
                # keep track of start time/frame for later
                token.tStart = t
                token.frameNStart = frameN  # exact frame index
                token.setAutoDraw(True)
            frameRemains = 0.5 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left
            if token.status == STARTED and t >= frameRemains:
                token.setAutoDraw(False)
            
            # *reward* updates
            if t >= 2.5 and reward.status == NOT_STARTED:
                # keep track of start time/frame for later
                reward.tStart = t
                reward.frameNStart = frameN  # exact frame index
                reward.setAutoDraw(True)
            frameRemains = 2.5 + 2- win.monitorFramePeriod * 0.75  # most of one frame period left
            if reward.status == STARTED and t >= frameRemains:
                reward.setAutoDraw(False)
            
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
        
        trials.addData('pGreen', P_green)
        trials.addData('pRed', P_red)
        trials.addData('pBlue', P_blue)
        trials.addData('pGoal', P_goal)
        trials.addData('randn', n)
        trials.addData('tokenShown', tokenShown)
        trials.addData('tsFile', tsFile)
        trials.addData('isGoal', isgoal)
        trials.addData('outcome', ov)
        earnings = earnings + ov
        nb_corr = nb_corr + key_choice.corr
        
        thisPrac.nextEntry()
        tr = tr + 1
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filenamePrac + '_final.xlsx', sheetName='trials',
        stimOut=params, dataOut=['all_raw'])
    
    prop_corr = nb_corr*100/nb_tr
    print 'Total earnings practice = $%.2f' %(earnings/100) #print earnings in output window
    print 'Percent correct = %.2f' %(prop_corr)
    
    # ------Prepare to start Routine "thanks"-------
    t = 0
    thanksClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    routineTimer.add(5.000000)
    # show cumulated earning
    thanksText = 'End of the practice session\n\nTotal earnings = $%.2f \n\nPercent correct = %.2f' % (earnings/100.0, prop_corr)
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
        frameRemains = 0.0 + 5 - win.monitorFramePeriod * 0.75  # most of one frame period left
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
    
    #-------------SAVE DATA IN VARIOUS FILES-------------------
    # these shouldn't be strictly necessary (should auto-save)
    thisPrac.saveAsWideText(filenamePrac+'_backup.csv')
    thisPrac.saveAsPickle(filenamePrac+'_data')
    
    wb = open_workbook(filenamePrac + '_final.xlsx')
    s = wb.sheet_by_index(0)
    d = {}
    for col in range(s.ncols):
        key_name = s.cell(0,col).value
        key_vals = s.col_values(col,start_rowx=1,end_rowx=nb_tr+1)
        d[key_name] = key_vals
    wb_fin = xlsxwriter.Workbook(filenamePrac+'_final_ord.xlsx', {'strings_to_numbers': True})
    ws_fin = wb_fin.add_worksheet()
    ws_fin.write(0,0,"index")
    ws_fin.write_column(1,0,tuple([int(num) for num in d['order']]))
    ws_fin.write(0,1,"trialNb")
    ws_fin.write_column(1,1,tuple([int(num) for num in d['trialNb']]))
    ws_fin.write(0,2,"goalToken")
    ws_fin.write_column(1,2,tuple([str(num) for num in d['goalToken']]))
    ws_fin.write(0,3,"unavAct")
    ws_fin.write_column(1,3,tuple([int(num) for num in d['unavailAct']]))
    ws_fin.write(0,4,"corrAct") 
    ws_fin.write_column(1,4,tuple([int(num) for num in d['corrAct']]))
    ws_fin.write(0,5,"bestAct") 
    ws_fin.write_column(1,5,tuple([int(num) for num in d['bestAct']]))
    ws_fin.write(0,6,"choice")
    ws_fin.write_column(1,6,tuple([int(num) for num in d['choice_raw']]))
    ws_fin.write(0,7,"isCorr") 
    ws_fin.write_column(1,7,tuple([int(num) for num in d['isCorr_raw']]))
    ws_fin.write(0,8,"choiceRT") 
    ws_fin.write_column(1,8,tuple([float(num) for num in d['choiceRT_raw']]))
    ws_fin.write(0,9,"pGreen") 
    ws_fin.write_column(1,9,tuple([round(num,2) for num in d['pGreen_raw']]))
    ws_fin.write(0,10,"pRed") 
    ws_fin.write_column(1,10,tuple([round(num,2) for num in d['pRed_raw']]))
    ws_fin.write(0,11,"pBlue") 
    ws_fin.write_column(1,11,tuple([round(num,2) for num in d['pBlue_raw']]))
    ws_fin.write(0,12,"pGoal") 
    ws_fin.write_column(1,12,tuple([round(num,2) for num in d['pGoal_raw']]))
    ws_fin.write(0,13,"randn") 
    ws_fin.write_column(1,13,tuple([float(num) for num in d['randn_raw']]))
    ws_fin.write(0,14,"isGoal") 
    ws_fin.write_column(1,14,tuple([int(num) for num in d['isGoal_raw']]))
    ws_fin.write(0,15,"tokenShown") 
    ws_fin.write_column(1,15,tuple([str(num) for num in d['tokenShown_raw']]))
    ws_fin.write(0,16,"outcome") 
    ws_fin.write_column(1,16,tuple([int(num) for num in d['outcome_raw']]))
    wb_fin.close()
    
    logging.flush()
    # make sure everything is closed down
    thisPrac.abort()  # or data files will save again on exit