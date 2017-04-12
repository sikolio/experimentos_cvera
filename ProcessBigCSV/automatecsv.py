import csv

import pandas as pd
import numpy as np

import argparse

file = pd.read_csv('FLIR.csv', sep=';')

def processFrame(df, enddata, highLimit, lowLimit, rangosdetemp):
    frameData = []
    maxarea = 200000
    if df.mean().mean() < lowLimit:
        return
    elif df.mean().mean() > highLimit:
        return
    else:
        p100 = sum(df[df>=minimum].count())
        if p100 < maxarea:
            frameData.append(p100)
            for a, b in zip(rangosdetemp, rangosdetemp[1:]):
                frameData.append(sum(df[df>=a][df<b].count()) / p100)
            print(frameData)
            enddata.append(frameData)
            
def useFile(camino, salida, start, finish, highLimit, lowLimit, rangosdetemp):
    data = []
    with open(camino) as f:
        reader = csv.reader(f, delimiter=';')

        for x in range(finish):
            print('Frame {}'.format(x))
            frame = []
            for _ in range(480):
                frame.append(next(reader))
            if x < start:
                continue
            df = pd.DataFrame(frame)
            df = df.applymap(lambda x: int(float(x.replace(',','.'))))
            processFrame(df, data, highLimit, lowLimit, rangosdetemp)
            
            
    newDF = pd.DataFrame(data)
    newDF.to_csv(salida)
    
for row in file.iterrows():
    print(row)
    minimum = float(row[1]['min'].replace(',','.'))
    maximum = float(row[1]['max'].replace(',','.'))
    lowLimit = 20
    highLimit = 70
    start = row[1]['inicial']
    rangosdetemp = list(np.linspace(minimum, maximum, num=10))
    finish = row[1]['final']
    useFile(row[1]['path'], row[1]['nombre'], start, 
            finish, highLimit, lowLimit, rangosdetemp)
