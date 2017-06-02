# Experimentos con max av en vez de FLIR
# coding: utf-8

import pandas as pd
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

from sympy import solve
from sympy.abc import x

from scipy import optimize
import scipy.spatial.distance as distance

from math import atan2,degrees

#lectura de archivos
FILE_ENDINGS = ['.csv', ' - dinamica.csv', ' - perfil.csv', ' - max av.csv', ' - fspot.csv', ' - max av cup.csv', ' - frente.csv']

#UTILS

def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def piecewise_linear_double(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0, x >= x0],
                           [generate_function(k1, x0, y0),
                            generate_function(k2, x0, y0)])

def piecewise_linear_triple(x, x0, x1, x2, y0, k1, k2, k3):
    return np.piecewise(x, [x < x0, (x < x1) & (x >= x0), x >= x1],
                        [generate_function(k1, x0, y0),
                         generate_function(k2, x1, y0),
                         generate_function(k3, x2, y0)])

def generate_ecuation_from_two_rects(k1, x0_1, y0_1, k2, x0_2, y0_2):


    return '{} * x + {}'.format(k1 - k2, (k1 * x0_1 * y0_1) - (k2 * x0_2 * y0_2))

def generate_ecuation(k, x0, y0):
    return '{}*x + {}'.format(k, k * x0 * y0)

def generate_function(k, x0, y0):
    return lambda x: k*x + y0-k*x0

#Experimento es una lista que guarda todos los experimentos y accede a los DF que son los definidos en file_endings
class Experimento(object):
    def __init__(self, filename, folder="."):
        self.files = ['{}{}'.format(filename, file_ending) for file_ending in FILE_ENDINGS]
        self.folder = folder

        pandas_settings = {
            'delimiter': ';',
            'thousands': '.',
            'decimal': ',',
            'engine': 'python',
        }
        #se definen los 4 DF param, dinamica, perfil y flir.
        self.param = pd.read_csv(os.path.join(folder, self.files[0]), engine= 'python', delimiter=';')
        self.dinamica = pd.read_csv(os.path.join(folder, self.files[1]), **pandas_settings)
        self.perfil = pd.read_csv(os.path.join(folder, self.files[2]), **pandas_settings)
        try:
            self.temp = pd.read_csv(os.path.join(folder, self.files[3]), **pandas_settings)
            self.temp_outliers()
        except Exception as e:
            print(e)
            self.temp = None
        try:
            self.temp4cup = pd.read_csv(os.path.join(folder, self.files[4]), **pandas_settings)
            self.temp4cup_outliers()
        except Exception as e:
            print(e)
            self.temp4cup = None
            #"C:\Users\CCVV\OneDrive\TESIS\experimentos_cvera\Data\10cc50C - fspot.csv"
        try:
            self.tempcup = pd.read_csv(os.path.join(folder, self.files[5]), **pandas_settings)
            self.tempcup_outliers()
        except Exception as e:
            print(e)
            self.tempcup = None
        try:
            self.frente = pd.read_csv(os.path.join(folder, self.files[6]), **pandas_settings)
            self.frente_outliers()
        except Exception as e:
            print(e)
            self.frente = None
 #' - max av.csv', ' -  fspot.csv', ' - max av cup.csv'
        #paso los resultados de dinamica a param para poder usarlos facilmente
        self.param['result: largo final'] = self.dinamica['avance: distancia desde punto eyeccion'].iloc[-1]
        self.param['result: largo total'] = self.dinamica['avance: largo total flujo'].iloc[-1]
        self.param['result: ancho max final'] = self.dinamica['avance: ancho maximo'].iloc[-1]
        self.param['result: espesor max final'] = self.perfil['perfil: espesor'].max()
        self.param['result: distancia espesor final'] = self.perfil['perfil: distancia'].iloc[self.perfil['perfil: espesor'].argmax()]

    #se guarda el nuevo archivo con los datos sacados de distintos DF
    def to_csvs(self, safe=False):
        if safe:
            try:
                for file in [os.path.join(self.folder, filename) for filename in self.files]:
                    os.rename(file, '{}.old'.format(file))
            except Exception as e:
                print('The files already exists', e)
                return
        else:
            for file in self.files:
                os.remove(file)

        self.param.to_csv(os.path.join(folder, self.files[0]), sep=';')
        self.dinamica.to_csv(os.path.join(folder, self.files[1]), sep=';')
        self.perfil.to_csv(os.path.join(folder, self.files[2]), sep=';')
        if self.temp is not None:
            self.temp.to_csv(os.path.join(folder, self.files[3]), sep=';')
        if self.temp4cup is not None:
            self.temp4cup.to_csv(os.path.join(folder, self.files[4]), sep=';')
        if self.tempcup is not None:
            self.tempcup.to_csv(os.path.join(folder, self.files[5]), sep=';')
        if self.frente is not None:
            self.frente.to_csv(os.path.join(folder, self.files[5]), sep=';')            


    #a ver si ya se ha hecho este proceso con los archivos y generar una version .old
    def undone(self):
        for file in self.files:
            os.rename('{}.old'.format(file), file)

    #funcion que calcula los rangos de velocidades del flir.
    #def get_ranges(self):
    #    if self.flir is None:
    #        return []

    #    tmin, tmax = self.param['min'][0], self.param['max'][0]
    #    return [a for a in np.linspace(tmin, tmax, 10)]

    #funcion que determina en que rango esta la temperatura dada
    def rango_temperatura(self, temp):
        bins = self.get_ranges()
        if len(bins) < 1:
            return 0
        return np.digitize([temp], bins)[0]

    def temp_outliers(self, std_count=2):
        if self.temp is None:
            return None
        df = self.temp.copy()
        df['moving average'] = df['max'].rolling(20, center=True).median().fillna(method='bfill').fillna(method='ffill')
        df['moving std'] = df['max'].rolling(20, center=True).std().fillna(method='bfill').fillna(method='ffill')

        df = df[np.abs(df['max'] - df['moving average'])<=(2*df['moving std'])]
        self.temp = df
        return self.temp
    
    def frente_outliers(self, std_count=2):
        if self.frente is None:
            return None
        df = self.frente.copy()
        df['moving average'] = df['frente'].rolling(20, center=True).median().fillna(method='bfill').fillna(method='ffill')
        df['moving std'] = df['frente'].rolling(20, center=True).std().fillna(method='bfill').fillna(method='ffill')

        df = df[np.abs(df['frente'] - df['moving average'])<=(2*df['moving std'])]
        self.frente = df
        return self.frente
    
    def tempcup_outliers(self, std_count=2):
        if self.tempcup is None:
            return None
        df = self.tempcup.copy()
        df['moving average'] = df['max'].rolling(20, center=True).median().fillna(method='bfill').fillna(method='ffill')
        df['moving std'] = df['max'].rolling(20, center=True).std().fillna(method='bfill').fillna(method='ffill')

        df = df[np.abs(df['max'] - df['moving average'])<=(2*df['moving std'])]
        self.tempcup = df
        return self.tempcup
    
    def temp4cup_outliers(self, std_count=2):
        if self.temp4cup is None:
            return None
        df = self.temp4cup.copy()
        df['moving average'] = df['sp1'].rolling(20, center=True).median().fillna(method='bfill').fillna(method='ffill')
        df['moving std'] = df['sp1'].rolling(20, center=True).std().fillna(method='bfill').fillna(method='ffill')

        df = df[np.abs(df['sp1'] - df['moving average'])<=(2*df['moving std'])]
        self.temp4cup = df
        return self.temp4cup

    #funcion para generar rectas segun x datos de DF dinamica
    def fit_to_two_curves(self):
        def two_lines(x, a, b, c, d):
            one = a*x + b
            two = c*x + d
            return np.maximum(one, two)

        x = np.array(self.dinamica['avance: tiempo'])
        y = np.array(self.dinamica['avance: largo total'])
        
    def get_intersections(self, number=3, interactive=True):
        if number == 3:
             return self.fit_to_three_rects(interactive)
        elif number == 2:
             return self.fit_to_two_rects(interactive)
        else:
             return self.fit_to_one_rect(interactive)
        
    def fit_to_one_rect(self, interactive=True):
        xs = np.log(np.array(self.dinamica['avance: tiempo'][1:]))
        ys = np.log(np.array(self.dinamica['avance: largo total flujo'][1:]))
        
        fit = np.polyfit(xs, ys,1)
        if interactive:
            fit_fn = np.poly1d(fit)
            plt.plot(xs,ys, 'yo', xs, fit_fn(xs), '--k')
             
        return fit[0]
        
    def fit_to_two_rects(self, interactive=True):
        xs = np.log(np.array(self.dinamica['avance: tiempo'][1:]))
        ys = np.log(np.array(self.dinamica['avance: largo total flujo'][1:]))

        p , e = optimize.curve_fit(piecewise_linear_double, xs, ys)
        xd = np.linspace(xs[0], xs[-1], 100)

        rect1 = generate_function(p[2], p[0], p[1])
        rect2 = generate_function(p[3], p[0], p[1])

        intersection1 = intersection(
            line((xd[10], rect1(xd[10])), (xd[90], rect1(xd[90]))),
            line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90])))
        )

        pendiente1 = p[2]
        pendiente2 = p[3]

        index_inter1 = distance.cdist([intersection1], np.array(list(zip(xs, ys)))).argmin()

        closest_int1 = list(zip(xs, ys))[index_inter1]

        if interactive:
            plt.plot(xs, ys, "o")
            print(['{}'.format(px) for px in p])
            print(e)


            plt.plot(xd, generate_function(p[2], p[0], p[1])(xd), color='red')
            plt.plot(xd, generate_function(p[3], p[0], p[1])(xd), color='blue')

            plt.plot([intersection1[0]], [intersection1[1]], 'rx')

            plt.plot([closest_int1[0]], closest_int1[1], 'r+')

            plt.show()

        return pendiente1, pendiente2, closest_int1, index_inter1
        
    def fit_to_three_rects(self, interactive=True):
        xs = np.log(np.array(self.dinamica['avance: tiempo'][1:] / 1000))
        ys = np.log(np.array(self.dinamica['avance: largo total flujo'][1:]))

        p , e = optimize.curve_fit(piecewise_linear_triple, xs, ys)
        xd = np.linspace(xs[0], xs[-1], 100)

        rect1 = generate_function(p[4], p[0], p[3])
        rect2 = generate_function(p[5], p[1], p[3])
        rect3 = generate_function(p[6], p[2], p[3])

        intersection1 = intersection(
            line((xd[10], rect1(xd[10])), (xd[90], rect1(xd[90]))),
            line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90])))
        )
        intersection2 = intersection(
            line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90]))),
            line((xd[10], rect3(xd[10])), (xd[90], rect3(xd[90])))
        )

        pendiente1 = p[4]
        pendiente2 = p[5]
        pendiente3 = p[6]


        index_inter1 = distance.cdist([intersection1], np.array(list(zip(xs, ys)))).argmin()
        index_inter2 = distance.cdist([intersection2], np.array(list(zip(xs, ys)))).argmin()

        closest_int1 = list(zip(xs, ys))[index_inter1]
        closest_int2 = list(zip(xs, ys))[index_inter2]

        if interactive:
            plt.plot(xs, ys, "o")
            print(['{}'.format(px) for px in p])
            print(e)


            plt.plot(xd, generate_function(p[4], p[0], p[3])(xd), color='red')
            plt.plot(xd, generate_function(p[5], p[1], p[3])(xd), color='blue')
            plt.plot(xd, generate_function(p[6], p[2], p[3])(xd), color='green')

            plt.plot([intersection1[0]], [intersection1[1]], 'rx')
            plt.plot([intersection2[0]], [intersection2[1]], 'bx')

            plt.plot([closest_int1[0]], closest_int1[1], 'r+')
            plt.plot([closest_int2[0]], closest_int2[1], 'b+')

            plt.show()

        return pendiente1, pendiente2, pendiente3, closest_int1, closest_int2, index_inter1, index_inter2

    #fit to 3 curves, return 3 slopes and the 2 nearest points to the intersections
    def get_closest_points_to_intersections(self, interactive=True):
        xs = np.log(np.array(self.dinamica['avance: tiempo'][1:] / 1000))
        ys = np.log(np.array(self.dinamica['avance: largo total flujo'][1:]))


        try:
            p , e = optimize.curve_fit(piecewise_linear_triple, xs, ys)
            if e[0][0] == float('inf') or e[0][0] == float('-inf'):
                raise ValueError('Infinity and beyond')
            xd = np.linspace(xs[0], xs[-1], 100)

            rect1 = generate_function(p[4], p[0], p[3])
            rect2 = generate_function(p[5], p[1], p[3])
            rect3 = generate_function(p[6], p[2], p[3])

            intersection1 = intersection(
                line((xd[10], rect1(xd[10])), (xd[90], rect1(xd[90]))),
                line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90])))
            )
            intersection2 = intersection(
                line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90]))),
                line((xd[10], rect3(xd[10])), (xd[90], rect3(xd[90])))
            )

            pendiente1 = p[4]
            pendiente2 = p[5]
            pendiente3 = p[6]


            index_inter1 = distance.cdist([intersection1], np.array(list(zip(xs, ys)))).argmin()
            index_inter2 = distance.cdist([intersection2], np.array(list(zip(xs, ys)))).argmin()

            closest_int1 = list(zip(xs, ys))[index_inter1]
            closest_int2 = list(zip(xs, ys))[index_inter2]

            if interactive:
                plt.plot(xs, ys, "o")
                print(['{}'.format(px) for px in p])
                print(e)


                plt.plot(xd, generate_function(p[4], p[0], p[3])(xd), color='red')
                plt.plot(xd, generate_function(p[5], p[1], p[3])(xd), color='blue')
                plt.plot(xd, generate_function(p[6], p[2], p[3])(xd), color='green')

                plt.plot([intersection1[0]], [intersection1[1]], 'rx')
                plt.plot([intersection2[0]], [intersection2[1]], 'bx')

                plt.plot([closest_int1[0]], closest_int1[1], 'r+')
                plt.plot([closest_int2[0]], closest_int2[1], 'b+')

                plt.show()

            return pendiente1, pendiente2, pendiente3, closest_int1, closest_int2
        
        except Exception as e:
            print(e)

            xs = np.log(np.array(self.dinamica['avance: tiempo'][1:]))

            p , e = optimize.curve_fit(piecewise_linear_double, xs, ys)
            xd = np.linspace(xs[0], xs[-1], 100)

            rect1 = generate_function(p[2], p[0], p[1])
            rect2 = generate_function(p[3], p[0], p[1])

            intersection1 = intersection(
                line((xd[10], rect1(xd[10])), (xd[90], rect1(xd[90]))),
                line((xd[10], rect2(xd[10])), (xd[90], rect2(xd[90])))
            )

            pendiente1 = p[2]
            pendiente2 = p[3]

            index_inter1 = distance.cdist([intersection1], np.array(list(zip(xs, ys)))).argmin()

            closest_int1 = list(zip(xs, ys))[index_inter1]

            if interactive:
                plt.plot(xs, ys, "o")
                print(['{}'.format(px) for px in p])
                print(e)


                plt.plot(xd, generate_function(p[2], p[0], p[1])(xd), color='red')
                plt.plot(xd, generate_function(p[3], p[0], p[1])(xd), color='blue')

                plt.plot([intersection1[0]], [intersection1[1]], 'rx')

                plt.plot([closest_int1[0]], closest_int1[1], 'r+')

                plt.show()

            return pendiente1, pendiente2, closest_int1


    #no entiendo pa que wea sirve esta funcion...claramente es un plot de desglose, pero pa que? esos datos meh.
    def plot_desglose(self, param1, param2):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.plot(self.flir['FLIR: rango 3'], self.flir['FLIR: rango 4'])
        fig.show()

    def __repr__(self):
        return self.files[0]

    def __str__(self):
        return self.files[0]

nameList = [
        '5cc50C',
        '1cc54C',
        '25cc54C',
        '10cc60C',
        '25cc60C',
        '1cc59C',
        '1cc47C',
        '1cc55C',
        '25cc58C',
        '25cc70C',
        '10cc67C',
        '10cc59C',
        '5cc63C',
        '5cc59C',
        '10cc49C',
        '25cc62C',
        '1cc62C',
        '5cc53C',
        '5a25cc56C',
        '5cc57C',
        '25cc50C',
        '10cc55C',
        '1cc24C',
        '10cc57C',
        '10cc24C',
        '5cc55C',
        '5cc58C',
        '1cc48C',
        '1cc51C',
        '5cc48C',
        '10cc50C',
        '1cc54,7C',
        '1cc52C',
]


Experimentos = [Experimento(filename, r'.\Data') for filename in nameList]
