# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 10:32:29 2017

@author: Caroline
"""

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


def WaitForScanner(win,endExpNow,vol_cond):
    
    # Initialize components for Routine
    instrMainClock = core.Clock()
#    if vol_cond == 1:
#        waitMsgTxt = 'This will be a STABLE block\n\n(the same token stays valuable for a while before it switches to a different color)\n\nPlease wait for scanner...'
#    elif vol_cond == 2:    
#        waitMsgTxt = 'This will be a VOLATILE block\n\n(the valuable token will change many times during the block)\n\nPlease wait for scanner...'
    waitMsgTxt = 'Please wait for scanner...'
    waitMsg = visual.TextStim(win=win, name='waitMsg', text=waitMsgTxt, font='Arial',
        pos=[0, 0], height=40, wrapWidth=None, ori=0, color=u'white', colorSpace='rgb', opacity=1, depth=0.0)
    key_trigger = event.BuilderKeyResponse()
    
    # ------Prepare to start Routine -------
    t = 0
    instrMainClock.reset()  # clock
    continueRoutine = True
        
    # keep track of which components have finished
    instrMainComponents = [waitMsg, key_trigger]
    for thisComponent in instrMainComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine -------
    while continueRoutine:
        t = instrMainClock.getTime()
        
        # *waitMsg* updates
        if t >= 0.0 and waitMsg.status == NOT_STARTED:
            # keep track of start time/frame for later
            waitMsg.tStart = t
            waitMsg.setAutoDraw(True)
        
        # *ok1* updates
        if t >= 0.0 and key_trigger.status == NOT_STARTED:
            # keep track of start time/frame for later
            key_trigger.tStart = t
            key_trigger.status = STARTED
            # keyboard checking is just starting
            win.callOnFlip(key_trigger.clock.reset)  # t=0 on next screen flip
            event.clearEvents(eventType='keyboard')
        if key_trigger.status == STARTED:
            theseKeys = event.getKeys(keyList=['5'])
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                key_trigger.keys = theseKeys[-1]  # just the last key pressed
                key_trigger.rt = key_trigger.clock.getTime()
                # a response ends the routine
                continueRoutine = False
            
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instrMainComponents:
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
            
    # -------Ending Routine -------
    for thisComponent in instrMainComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)