import csv

import pandas as pd
import numpy as np


minimum = 30
maximum = 65
lowLimit = 18
highLimit = 40
start = 13

data = []

def processFrame(df, enddata):
    frameData = []
    if df.mean().mean() < lowLimit:
        return
    elif df.mean().mean() > highLimit:
        return
    else:
        rang = list(np.linspace(minimum, maximum, 5)) #definir como parametros
        p100 = sum(df[df>=minimum].count())
        frameData.append(p100)
        for a, b in zip(rang, rang[1:]):
            frameData.append(sum(df[df>=a][df<b+0.001].count()) / p100)
        print(frameData)
        enddata.append(frameData)


def returnValues(temperature):
    n = 0.0046 * temperature + 0.1449
    k = -3.8484 * temperature + 255.16
    tau = -1.0362 * temperature + 84.246

    return (n, k, tau)


with open('../../Desktop/2016-09-27 10cc60C #2.csv') as f:
    reader = csv.reader(f, delimiter=';')

    for x in range(4008):
        print('Frame {}'.format(x))
        frame = []
        for _ in range(480):
            frame.append(next(reader))
        if x < start:
            continue
        df = pd.DataFrame(frame)
        df = df.applymap(lambda x: int(float(x.replace(',','.'))))
        processFrame(df, data)

newDF = pd.DataFrame(data)
newDF.to_csv('result')
