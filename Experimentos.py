
# coding: utf-8

import pandas as pd
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

#lectura de archivos
FILE_ENDINGS = ['.csv', ' - dinamica.csv', ' - perfil.csv', ' - flir.csv']

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
            self.flir = pd.read_csv(os.path.join(folder, self.files[3]), **pandas_settings)
        except Exception as e:
            print(e)
            self.flir = None
        
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
        if self.flir is not None:
            self.flir.to_csv(os.path.join(folder, self.files[3]), sep=';')


    #a ver si ya se ha hecho este proceso con los archivos y generar una version .old    
    def undone(self):
        for file in self.files:
            os.rename('{}.old'.format(file), file)
    
    #funcion para generar rectas segun x datos de DF dinamica
    def fit_to_two_curves(self):
        def two_lines(x, a, b, c, d):
            one = a*x + b
            two = c*x + d
            return np.maximum(one, two)
        
        x = np.array(self.dinamica['avance: tiempo'])
        y = np.array(self.dinamica['avance: largo total'])
    
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

