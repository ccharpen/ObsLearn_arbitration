from __future__ import absolute_import, division
import sys  # to get file system encoding
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import xlsxwriter
from xlrd import open_workbook

def CreateTrialFile(fname):

    #randomize run order and create trial file for that subject
    tr_list_all = {'runPos':[], 'runNb':[], 'trialNb':[], 'trType':[], 'goalToken':[], 'tvCond':[], 'buCond':[], 
                   'unavAct':[], 'corrAct':[], 'bestAct':[], 'vertOrd':[]}
    run_order = np.random.permutation(8).tolist()
    vert_order = np.random.permutation(3).tolist() + np.random.permutation(3).tolist() + np.random.permutation(3).tolist()
    vord = [vert_order[i]+1 for i in range(8)]
    run_pos = 1
    for run in run_order:
        run_fname = 'trial_lists/run%i.xlsx' %(run + 1)    
        wb = open_workbook(run_fname)
        s = wb.sheet_by_index(0)
        for col in range(s.ncols):
            key_name = str(s.cell(0,col).value)
            key_vals = [int(num) for num in s.col_values(col,start_rowx=1,end_rowx=s.nrows)]
            tr_list_all[key_name] = tr_list_all[key_name] + key_vals
        tr_list_all['runPos'] = tr_list_all['runPos'] + [run_pos for i in range(s.nrows-1)]
        tr_list_all['vertOrd'] = tr_list_all['vertOrd'] + [vord[run_pos-1] for i in range(s.nrows-1)]
        run_pos = run_pos + 1
    
    comb_vals = [tr_list_all['runPos'], tr_list_all['runNb'], tr_list_all['trialNb'], tr_list_all['trType'], tr_list_all['goalToken'], 
                 tr_list_all['tvCond'], tr_list_all['buCond'], tr_list_all['unavAct'], tr_list_all['corrAct'], tr_list_all['bestAct'],
                 tr_list_all['vertOrd']]             
    final_list = np.asarray(comb_vals).transpose() 
    
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
    
    size = np.shape(final_list)
    rows = int(size[0])
    cols = int(size[1])
    for i in range(rows):
        for j in range(cols):
            worksheet.write(i+1,j,final_list[i][j])
    
    workbook.close()
    
    return(tr_list_all)