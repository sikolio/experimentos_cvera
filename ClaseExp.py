import csv
import os
import pandas as pd

file_endings = ['.csv', ' - dinamica.csv', ' - perfil.csv', ' - flir.csv' ]

class Experimento(object):
    def __init__(self, file_name):
        self.files = ['{}{}'.format(file_name, file_end) for file_end in file_endings] #cada
        #elemento de file_endings es y, se suma a el filename y se lee todo

        panda_settings = { #armar diccionario con valores a mi gusto
            'delimiter': ';',
            'thousands': '.',
            'decimal': ',',
            'engine': 'python',
            'header' : 0 #usa la primera fila del archivo y los asigna como headers
        }

        #para que lean como les enseñamos con el diccionario
        self.param = pd.read_csv(self.files[0], **panda_settings)
        self.dinamica = pd.read_csv(self.files[1], **panda_settings)
        self.perfil = pd.read_csv(self.files[2], **panda_settings)
        self.flir = pd.read_csv(self.files[3], **panda_settings)

        #genero los datos importantes de dinamica en el csv de parametro poniendole nombre al header y
        #escogiendo el ultimo valor de las columnas con el header mencionado en self.dinamica (con [-1])
        self.param['result: largo final desde punto eyeccion'] = self.dinamica['avance: distancia desde punto eyeccion'].iloc[-1]
        self.param['result: largo total'] = self.dinamica['avance: largo total flujo'].iloc[-1]
        self.param['result: ancho maximo final'] = self.dinamica['avance: ancho maximo'].iloc[-1]
        self.param['result: tiempo total medicion'] = self.dinamica['avance: tiempo'].iloc[-1]
        self.param['result: espesor maximo'] = self.perfil['perfil: espesor'].max()
        self.param['result: distancia a espesor maximo'] = self.perfil['perfil: largo'].iloc[self.perfil['perfil: espesor'].argmax()]

        #paso de area en pixeles a metros cuadrados
        self.flir['FLIR: area m2'] = self.flir['FLIR: area'] / self.param['pixeles'].iloc[0]

        #guardar valores nuevos en el excel
    def save_csv(self):
        for n_file in self.files:
            os.rename(n_file, n_file + '.original')
        self.param.to_csv(path_or_buf = self.files[0], sep = ';', decimal = ',')
        self.dinamica.to_csv(path_or_buf = self.files[1], sep = ';', decimal = ',')
        self.perfil.to_csv(path_or_buf = self.files[2], sep = ';', decimal = ',')
        self.flir.to_csv(path_or_buf = self.files[3], sep = ';', decimal = ',')
