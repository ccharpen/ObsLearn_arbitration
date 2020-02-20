from __future__ import absolute_import, division
import sys  # to get file system encoding
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import xlsxwriter
from xlrd import open_workbook

def CreateTrialFile_practice(nb_tr,subj):

    #randomize run order and create trial file for that subject
    tr_list_all = {'runPos':[], 'runNb':[], 'trialNb':[], 'trType':[], 'goalToken':[], 'tvCond':[], 'buCond':[], 
                   'unavAct':[], 'corrAct':[], 'bestAct':[], 'vertOrd':[], 'horizOrd':[]}
    
    #randomize vertical order of colors across blocks
    #1= green (top), red (middle), blue (bottom)
    #2= red (top), blue (middle), green (bottom)
    #3= blue (top), green (middle), red (bottom)
    vert_order = np.random.permutation(3).tolist()
    vord = [vert_order[i]+1 for i in range(2)]
    
    #randomize horizontal order of colors across blocks
    #1= green (left), red (middle), blue (right)
    #2= red (left), blue (middle), green (right)
    #3= blue (left), green (middle), red (right)
    horiz_order = np.random.permutation(3).tolist()
    hord = [horiz_order[i]+1 for i in range(2)]
            
    tr_fname = 'trial_lists/practice_before_scan.xlsx'
    wb = open_workbook(tr_fname)
    s = wb.sheet_by_index(0)
    for col in range(s.ncols-3): #the last 3 columns need to be changed depending on horiz_order
        key_name = str(s.cell(0,col).value)
        key_vals = [int(num) for num in s.col_values(col,start_rowx=1,end_rowx=s.nrows)]
        tr_list_all[key_name] = tr_list_all[key_name] + key_vals

    tr_list_all['vertOrd'] = [vord[0] for i in range(int(nb_tr/2))] + [vord[1] for i in range(int(nb_tr/2))]
    tr_list_all['horizOrd'] = [hord[0] for i in range(int(nb_tr/2))] + [hord[1] for i in range(int(nb_tr/2))]
    UA = [int(num) for num in s.col_values(7,start_rowx=1,end_rowx=s.nrows)] #vector of unavailable actions
    CA = [int(num) for num in s.col_values(8,start_rowx=1,end_rowx=s.nrows)] #vector of correct actions
    BA = [int(num) for num in s.col_values(9,start_rowx=1,end_rowx=s.nrows)] #vector of best actions
    
    #update unavailable, correct and best action depending on horizontal order position
    for tr in range(nb_tr):
        if tr_list_all['horizOrd'][tr] == 1:
            UAt = UA[tr]
            CAt = CA[tr]
            BAt = BA[tr]
        elif tr_list_all['horizOrd'][tr] == 2:
            if UA[tr] == 1:
                UAt = 3
            elif UA[tr] == 2:
                UAt = 1
            elif UA[tr] == 3:
                UAt = 2
            if CA[tr] == 1:
                CAt = 3
            elif CA[tr] == 2:
                CAt = 1
            elif CA[tr] == 3:
                CAt = 2
            if BA[tr] == 1:
                BAt = 3
            elif BA[tr] == 2:
                BAt = 1
            elif BA[tr] == 3:
                BAt = 2
        elif tr_list_all['horizOrd'][tr] == 3:
            if UA[tr] == 1:
                UAt = 2
            elif UA[tr] == 2:
                UAt = 3
            elif UA[tr] == 3:
                UAt = 1
            if CA[tr] == 1:
                CAt = 2
            elif CA[tr] == 2:
                CAt = 3
            elif CA[tr] == 3:
                CAt = 1
            if BA[tr] == 1:
                BAt = 2
            elif BA[tr] == 2:
                BAt = 3
            elif BA[tr] == 3:
                BAt = 1
        tr_list_all['unavAct'].append(UAt)
        tr_list_all['corrAct'].append(CAt)
        tr_list_all['bestAct'].append(BAt)
    
    #write complete trial list in excel file
    comb_vals = [tr_list_all['runPos'], tr_list_all['runNb'], tr_list_all['trialNb'], tr_list_all['trType'], tr_list_all['goalToken'], 
                 tr_list_all['tvCond'], tr_list_all['buCond'], tr_list_all['unavAct'], tr_list_all['corrAct'], tr_list_all['bestAct'],
                 tr_list_all['vertOrd'], tr_list_all['horizOrd']]             
    final_list = np.asarray(comb_vals).transpose() 
    
    fname = 'trial_lists/trial_list_practice_Sub%s.xlsx' % (subj)
    workbook = xlsxwriter.Workbook(fname, {'strings_to_numbers': True})
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,"runPos")
    worksheet.write(0,1,"runNb")
    worksheet.write(0,2,"trialNb")
    worksheet.write(0,3,"trType")
    worksheet.write(0,4,"goalToken")
    worksheet.write(0,5,"tvCond")   
    worksheet.write(0,6,"buCond")
    worksheet.write(0,7,"unavAct") 
    worksheet.write(0,8,"corrAct") 
    worksheet.write(0,9,"bestAct")  
    worksheet.write(0,10,"vertOrd")
    worksheet.write(0,11,"horizOrd")  
    
    size = np.shape(final_list)
    rows = int(size[0])
    cols = int(size[1])
    for i in range(rows):
        for j in range(cols):
            worksheet.write(i+1,j,final_list[i][j])
    
    workbook.close()
    
    return(tr_list_all)