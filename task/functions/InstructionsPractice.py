# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:45:21 2017

@author: Caroline Charpentier
"""

from __future__ import absolute_import, division
import sys  # to get file system encoding
#sys.path.append("C:\Program Files (x86)\PsychoPy2\Lib\site-packages\PsychoPy-1.84.2-py2.7.egg")
#sys.path.append("C:\Program Files (x86)\PsychoPy2\Lib\site-packages")
from psychopy import locale_setup, gui, visual, core, data, event, logging, sound
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle


def InstructionsPractice(thisPrac,endExpNow):
    
    # Setup the Window
    dispWidth = 1024
    dispHeight = 768
    wini = visual.Window(size=(dispWidth, dispHeight), fullscr=True, screen=0, allowGUI=False, allowStencil=False,
        monitor='testMonitor', color=[0.5,0.5,0.5], colorSpace='rgb', blendMode='avg', useFBO=False, units='pix')

    # Initialize components for Routine "instrMain"
    instrMainClock = core.Clock()
    inst=''
    instrPic = visual.ImageStim(win=wini, name='instrPic', image='sin', mask=None, ori=0, 
                                pos=(0, 0), color=[1,1,1], colorSpace='rgb', opacity=1,
                                flipHoriz=False, flipVert=False, texRes=128, interpolate=False, depth=-1.0)
    
    # Show instructions on screen
    inst_n = 0
    inst_list = ['instr_prac1.png','instr_prac2.png','instr_prac3.png','instr_prac4.png','instr_prac5.png',
                 'instr_prac6.png','instr_prac7.png','instr_prac8.png','instr_prac9.png','instr_prac10.png',
                 'instr_prac11.png','instr_prac12.png','instr_prac13.png']
    inst_pres = 1
    while inst_pres == 1:
        # ------Prepare to start Routine "instrMain"-------
        t = 0
        instrMainClock.reset()  # clock
        continueRoutine = True
        # update component parameters for each repeat
        inst = 'instructions/' + inst_list[inst_n]
        instrPic.setImage(inst)
        ok1 = event.BuilderKeyResponse()
        # keep track of which components have finished
        instrMainComponents = [instrPic, ok1]
        for thisComponent in instrMainComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
    
        # -------Start Routine "instrMain"-------
        while continueRoutine:
            # get current time
            t = instrMainClock.getTime()
            
            # *instrPic* updates
            if t >= 0.0 and instrPic.status == NOT_STARTED:
                # keep track of start time/frame for later
                instrPic.tStart = t
                instrPic.setAutoDraw(True)
            
            # *ok1* updates
            if t >= 0.0 and ok1.status == NOT_STARTED:
                # keep track of start time/frame for later
                ok1.tStart = t
                ok1.status = STARTED
                # keyboard checking is just starting
                wini.callOnFlip(ok1.clock.reset)  # t=0 on next screen flip
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
            for thisComponent in instrMainComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                wini.close()
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                wini.flip()
    
        # -------Ending Routine "instrMain"-------
        for thisComponent in instrMainComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
    
        # check responses
        thisPrac.addData('ok1.keys',ok1.keys)
        thisPrac.addData('ok1.rt', ok1.rt)
        thisPrac.nextEntry()
            
        # determine which image to show next
        if inst_n not in [5, 6, 8, 9]:
            if ok1.keys=='right': # right click = move forward
                if inst_n == 12:
                    inst_pres = 0
                    break
                else:
                    inst_n = inst_n + 1
            elif ok1.keys=='left': #left click = move back
                if inst_n >= 1:
                    inst_n = inst_n - 1
        elif inst_n in [5, 9]: #test questions where correct response is down
            if ok1.keys == 'down':
                inst_n = inst_n + 1
        elif inst_n in [6, 8]: #test questions where correct response is right
            if ok1.keys == 'right':
                inst_n = inst_n + 1
    
    return(thisPrac)