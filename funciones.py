# -*- coding: utf-8 -*-
"""
Implementación de funciones útiles al procesado del audio, dentro de este
script se incluyen:

    - Funciones de aplicación y generación de los filtros de octavas y tercios
    de octavas, definidos según la Norma IEC 61620-1.

    - Implementación de funciones para la realización de ponderaciones,
    temporales y de frecuencia, a partir de la Norma IEC 61672-1.

    - Funciones varias para el cálculo simplificado de niveles sonoros,
    de acuerdo a requerimientos específicos.

    - Funciones de compensación y calibrado del dispositivo.

    - Funciones para el post-procesamiento y análisis de los datos ya
    obtenidos por el dispositivo.
"""

import os
import numpy as np
import sounddevice as sd
from scipy.signal import butter, zpk2sos, sosfilt, convolve
import datetime
import time
import pandas as pd
import warnings
import tables

# Parámetros por defecto
fs = 48000 # Frecuencia de Muestreo [Hz]
fr = 1000.0 # Frecuencia de Referencia [Hz]

root = '/home/pi/Documents/' # Carpeta utilizada por defecto.

"""
Filtrado de frecuencia por octavas y tercios de octavas.
"""

fto_nom = np.array([ 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
                    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
                    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
                    6300.0, 8000.0, 10000.0,])

foct_nom = fto_nom[1::3]

oct_ratio = np.around(10.0**(3.0/10.0), 5) # Según ecuación 1 en IEC 61672-1

def frec_a_num(frecs=fto_nom, frec_ref=fr, n_oct=3.0):
    """ Devuelve el número de banda para las frecuencias centrales ingresadas,
    correspondiendo la banda 0 a la frecuencia de referencia.
    n_oct representa el número de bandas por octava, siendo 1.0 para octavas
    y 3.0 para tercios de octava. """
    return np.round(n_oct*np.log2(frecs/frec_ref))

def frec_cen(nband=np.arange(-5, 4), frec_ref=fr, n_oct=1.0):
    """ Calcula las frecuencias centrales exactas para bandas de octava y
    tercios de octava. Recibe los números de banda a calcular, la
    frecuencia de referencia, y el número de bandas por octava. """
    return np.around(frec_ref*oct_ratio**(nband/n_oct), 5)

def frec_lim(frec_cen, n_oct=1.0):
    """ Devuelve las frecuencias lí­mites (inferior y superior), para
    bandas de tercio de octava y octavas, según los valores medios exactos
    de las bandas y el número de bandas por octava. """
    return np.around(frec_cen*oct_ratio**(-1/(2*n_oct)), 5), np.around(frec_cen*oct_ratio**(1/(2*n_oct)), 5)

def fto_lim(fto_nom=fto_nom):
    """ Entrega las frecuencias límite para el diseño de los filtros de
    tercio de octava.
    Permite el ingreso del valor nominal de las frecuencias centrales para las
    bandas deseadas, 'fto_nom'. Por defecto 25 Hz < f < 10 kHz. """
    fto_inf, fto_sup = frec_lim(frec_cen(frec_a_num(fto_nom, fr, 3.0), fr, 3.0), 3.0)
    return fto_inf, fto_sup

def foct_lim(foct_nom=foct_nom):
    """ Entrega las frecuencias límite para el diseño de los filtros de
    octava.
    Permite el ingreso del valor nominal de las frecuencias centrales para las
    bandas deseadas, 'foct_nom'. Por defecto 31.5 Hz < f < 8 kHz. """
    foct_inf, foct_sup = frec_lim(frec_cen(frec_a_num(foct_nom, fr, 1.0), fr, 1.0), 1.0)
    return foct_inf, foct_sup

def but_pb(inf, sup, fs=fs, order=4):
    """ Obtención de los coeficientes para el diseño del filtro.
    Siendo estos filtros Butterworth pasa banda. """
    nyq = 0.5*fs
    low = inf/nyq
    high = sup/nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def but_pb_filt(x, inf, sup, fs=fs, order=4):
    """ Filtrado de la señal. """
    sos = but_pb(inf, sup, fs=fs, order=order)
    return sosfilt(sos, x)

def filt_to(x, fto_nom=fto_nom):
    """ Filtrado de la señal de entrada 'x' por bandas de tercio de octava.
    Permite el ingreso del valor nominal de las frecuencias centrales para las
    bandas deseadas, 'fto_nom'. Por defecto 25 Hz < f < 10 kHz. """
    fto_inf, fto_sup = fto_lim(fto_nom)
    y = np.empty([len(fto_nom),len(x)])
    for i in range(len(fto_nom)):
        y[i] = np.reshape(but_pb_filt(x, fto_inf[i], fto_sup[i]), -1)
    return y

def filt_oct(x, foct_nom=foct_nom):
    """ Filtrado de la señal de entrada 'x' por bandas de octava.
    Permite el ingreso del valor nominal de las frecuencias centrales para las
    bandas deseadas, 'foct_nom'. Por defecto 31.5 Hz < f < 8 kHz. """
    foct_inf, foct_sup = foct_lim(foct_nom)
    y = np.empty([len(foct_nom),len(x)])
    for i in range(len(foct_nom)):
        y[i] = np.reshape(but_pb_filt(x, foct_inf[i], foct_sup[i]), -1)
    return y

"""
Funciones para ponderación temporal y de frecuencia.
"""

# Coeficientes calculados a partir de las ecuaciones en IEC 61672-1
z_c = np.array([0, 0])
p_c = np.array([-2*np.pi*20.598997057568145, -2*np.pi*20.598997057568145,
    -2*np.pi*12194.21714799801, -2*np.pi*12194.21714799801])
k_c = (10**(0.062/20))*p_c[3]**2

z_a = np.append(z_c, [0,0])
p_a = np.array([-2*np.pi*20.598997057568145, -2*np.pi*20.598997057568145,
     -2*np.pi*107.65264864304628, -2*np.pi*737.8622307362899,
     -2*np.pi*12194.21714799801, -2*np.pi*12194.21714799801])
k_a = (10**(2/20))*p_a[4]**2

def zpk_bil(z, p, k, fs=fs):
    """ Devuelve los parametros para un filtro digital a partir de un analógico,
    a partir de la transformada bilineal. Transforma los polos y ceros del
    plano 's' al plano 'z'. """
    deg = len(p) - len(z)
    fs2 = 2.0*fs
    # Transformación bilineal de polos y ceros:
    z_b = (fs2 + z)/(fs2 - z)
    p_b = (fs2 + p)/(fs2 - p)
    z_b = np.append(z_b, -np.ones(deg))
    k_b = k*np.real(np.prod(fs2 - z)/np.prod(fs2 - p))
    return z_b, p_b, k_b

sos_C = zpk2sos(*zpk_bil(z_c, p_c, k_c))
sos_A = zpk2sos(*zpk_bil(z_a, p_a, k_a))

def filt_A(x):
    """ Devuelve la señal posterior al proceso de filtrado según
    ponderación 'A'. Recibe la señal de entrada (en dimensión temporal),
    y la frecuencia de sampleo. """
    return sosfilt(sos_A, x)

def filt_C(x):
    """ Devuelve la señal posterior al proceso de filtrado según
    ponderación 'C'. Recibe la señal de entrada (en dimensión temporal). """
    return sosfilt(sos_C, x)

def gen_slow(fs=fs):
    """Generación del filtro de ponderación temporal 'Slow'. """
    b, a = bilinear(np.poly1d([1]), np.poly1d([1.0, 1]), fs)
    return b, a

def gen_fast(fs=fs):
    """Generación del filtro de ponderación temporal 'Fast'. """
    b, a = bilinear(np.poly1d([1]), np.poly1d([0.125, 1]), fs)
    return b, a

def slow(x, fs=fs):
    """ Aplicación de la ponderación temporal 'Slow'. """
    b, a = gen_slow(fs)
    return np.sqrt(lfilter(b, a, x**2))

def fast(x, fs=fs):
    """ Aplicación de la ponderación temporal 'Fast'. """
    b, a = gen_fast(fs)
    return np.sqrt(lfilter(b, a, x**2))

"""
Funciones útiles para obtención y cálculo de niveles.
"""
def wavread(x):
    """ Lectura y normalizado de archivos de audio en formato .wav,
    indicando su ubicación 'x'. """
    norm_fact = {'int16': (2**15)-1, 'int32': (2**31)-1, 'int64': (2**63)-1, 'float32': 1.0, 'float64': 1.0}
    fs, y = wavfile.read(x)
    y = np.float32(y)/norm_fact[y.dtype.name]
    return fs, y

def rms(x):
    """ Cálculo de nivel RMS para la señal de entrada 'x'. """
    return np.sqrt(np.sum(x**2)/len(x))

def rms_t(x, t=1.0, fs=fs):
    """ Cálculo de niveles RMS para la señal de entrada 'x', fragmentando
    de acuerdo al intervalo de tiempo 't' indicado. """
    N = int(np.floor(t*fs))
    if x.ndim == 1 :
        p_inic = np.arange(0, len(x), N)
        p_fin = np.zeros(len(p_inic), dtype="int32")
        p_fin[0:-1] = p_inic[1:]
        p_fin[-1] = len(x)
        y= np.empty(len(p_inic))
        for i in np.arange(len(p_inic)):
            y[i] = rms(x[p_inic[i]:p_fin[i]])
    else:
        p_inic = np.arange(0, x.shape[1], N)
        p_fin = np.zeros(len(p_inic), dtype="int32")
        p_fin[0:-1] = p_inic[1:]
        p_fin[-1] = x.shape[1]
        y= np.empty([x.shape[0], len(p_inic)])
        for i in np.arange(x.shape[0]):
            for j in np.arange(len(p_inic)):
                y[i, j] = rms(x[i, p_inic[j]:p_fin[j]])
    return y

def niveles(x, cal, ncal):
    """ Obtención de los niveles de presión sonora [dBSPL], para la señal de
    entrada 'x'. Debe ingresarse el valor eficaz de la calibración 'cal' y
    su nivel medido por un sonómetro de referencia. """
    return 20*np.log10(x/cal)+ncal

def sum_db(x):
    """ Suma energética de niveles en dB. """
    return 10*np.log10(np.sum(10**(x/10)))

def mean_db(x):
    """ Promedio enrgético de niveles en dB. """
    return 10*np.log10(np.mean(10**(x/10)))

def mean_t(x):
    """Promedio energético de todo el tiempo analizado para
    cada banda fraccional de octava o para el nivel global."""
    return np.apply_along_axis(mean_db,1,x)

def sum_f(x):
    """Suma energética de todos los niveles en frecuencia,
    para cada instante temporal."""
    return np.apply_along_axis(sum_db,0,x)

def ajustar(x):
    """ Ajuste de los niveles de entrada 'x'. """
    try:
        ajuste = busca_ajuste()
    except:
        print('No se pudo encontrar el ajuste.')
        return
    x_aj = np.zeros(x.shape)
    for i in np.arange(x.shape[0]):
        x_aj[i,:] = x[i,:] + ajuste[i]
    return x_aj

def leqas_oct(x, cal, ncal, fs, aj='False'):
    """ Niveles equivalentes por segundo con ponderación 'A' filtrados
    por banda de octava.
    Recibe la señal de entrada 'x', el nivel
    eficaz de la calibración 'cal' y su valor registrado por el
    sonómetro de referencia 'ncal'. Para aplicar ajuste se debe
    indicar "aj=True". """
    niv_x = niveles(rms_t(filt_oct(filt_A(x)), fs=fs), cal, ncal)
    if aj == True:
        try:
            return ajustar(niv_x)
        except:
            pass
    return niv_x

def leqaf_oct(x, cal, ncal, fs, aj='False'):
    """ Niveles equivalentes a intervalos de 0.125 s con ponderación 'A'
    filtrados por banda de octava.
    Recibe la señal de entrada 'x', el nivel
    eficaz de la calibración 'cal' y su valor registrado por el
    sonómetro de referencia 'ncal'. Para aplicar ajuste se debe
    indicar "aj=True". """
    niv_x = niveles(rms_t(filt_oct(filt_A(x)), t=0.125, fs=fs), cal, ncal)
    if aj == True:
        try:
            return ajustar(niv_x)
        except:
            pass
    return niv_x

def las_oct(x, cal, ncal, fs, aj='False'):
    """ Niveles con ponderación temporal 'Slow' y ponderación
    frecuencial 'A', filtrados por banda de octava.
    Recibe la señal de entrada 'x', el nivel
    eficaz de la calibración 'cal' y su valor registrado por el
    sonómetro de referencia 'ncal'. Para aplicar ajuste se debe
    indicar "aj=True". """
    niv_x = niveles(rms_t(slow(filt_oct(filt_A(x))), fs=fs), cal, ncal)
    if aj == True:
        try:
            return ajustar(niv_x)
        except:
            pass
    return niv_x

def laf_oct(x, cal, ncal, fs, aj='False'):
    """ Niveles con ponderación temporal 'Fast' y ponderación
    frecuencial 'A', filtrados por banda de octava.
    Recibe la señal de entrada 'x', el nivel
    eficaz de la calibración 'cal' y su valor registrado por el
    sonómetro de referencia 'ncal'. Para aplicar ajuste se debe
    indicar "aj=True". """
    niv_x = niveles(rms_t(slow(filt_oct(filt_A(x))), t=0.125, fs=fs), cal, ncal)
    if aj == True:
        try:
            return ajustar(niv_x)
        except:
            pass
    return niv_x

"""
Calibrado y compensación del dispositivo.
"""

def cal():
    """ Ejecución de la calibración del dispositivo. Una vez ejecutado el
    código luego de 5 segundos se comienza a grabar la entrada.
    Debe ser reproducido un tono de 94 dBSPL y 1 kHz, registrándolo
    a su vez en un sonómetro.
    Finalmente, debe indicarse el nivel obtenido en este último. """
    print("\nComenzando la calibración en 5 segundos.")
    time.sleep(5)
    print("\nGrabando calibración ... ")
    sd.default.device = 'snd_rpi_simple_card'
    x = sd.rec(int(5*fs), 48000, 1, blocking=True)
    vcal = rms(x)
    print("\nGrabación finalizada.")
    ncal = float(input("\nIngrese valor de la calibración: "))
    np.savetxt(root + 'cal.csv', (vcal, ncal))
    print("\nSe escribió el archivo 'cal.csv'")
    return
    return vcal, ncal

def busca_cal(root=root):
    """ Obtención del valor eficaz de la calibración grabada y su valor
    en NPS obtenido por un sonómetro calibrado. """
    vcal, ncal = np.loadtxt(root + 'cal.csv', unpack=True)
    return vcal, ncal

def busca_ajuste(root=root):
    """ Obtención de los coeficientes de compensación para el ajuste del
    dispositivo. """
    ajuste = np.loadtxt(root + 'ajuste.csv', unpack=False)
    return ajuste

"""
Funciones utilizadas para el manejo y análisis de los datos de salida.
"""

def guardado(yy, mn):
    """ Función wrapper. Busca todos los arrays de '.npy' creados para el mes
    ingresado y los guarda en dataframes sobre archivos '.h5'.
    Los datos ingresados deben estar en formato str. """
    path = root + str(yy) + '_' + str(mn) + '/'
    dias_npy = []
    for items in os.listdir(path):
        if items[-4:] == '.npy':
            if items[1] == '.':
                dias_npy.append(items[0])
            else:
                dias_npy.append(items[:2])
    dias_npy.sort(key=int)
    guardar_h5(yy, mn, dias_npy)
    return

def guardar_h5(yy, mn, dd, datos=0):
    """ Función wrapper para la conversión del archivo de volcado '.npy' en un
    dataframe de pandas y su posterior guardado en HDF5 (Hierarchical Data
    Format 5 File). """
    warnings.simplefilter('ignore', tables.NaturalNameWarning)
    path = root + str(yy) + '_' + str(mn) + '/'
    path_datos = path + 'datos.h5'
    if type(datos) == int:
        datos = npy_a_df(yy, mn, dd)
    if type(datos) == pd.core.frame.DataFrame:
        datos = {dd: datos}
    if type(datos) == dict:
        store = pd.HDFStore(path_datos)
        if not os.path.isfile(path_datos):
            for keys in datos:
                store.put(keys, datos[keys])
                os.remove(path + keys + '.npy')
        else:
            dias_datos = sorted(list(datos.keys()), key=int)
            dias_store = []
            for keys in list(store.keys()):
                if keys[-2] == '/':
                    dias_store.append(keys[-1])
                else:
                    dias_store.append(keys[-2:])
            dias_store.sort(key=int)
            for keys in dias_datos:
                if keys in dias_store:
                    aux = store[keys]
                    data_n = pd.concat((aux, datos[keys]))
                    store.put(keys, data_n)
                    os.remove(path + keys + '.npy')
                else:
                    store.put(keys, datos[keys])
                    os.remove(path + keys + '.npy')
        store.close()
    return

def npy_a_df_dias(yy, mn, dd):
    """ Función utilizada dentro de la ejecución de npy_a_df.
    Utilizada para convertir el archivo de volcado '.npy' a un dataframe de
    pandas. """
    dds = sorted(dd, key=int)
    datos_dict = {}
    for day in dds:
        try:
            datos_dict[day] = npy_a_df(yy, mn, day)
        except:
            continue
    return datos_dict

def npy_a_df(yy, mn, dd):
    """ Busca datos en el archivo de volcado '.npy' de manera recursiva para
    cada uno de los días indicados, devolviendo un dataframe con el período
    de medición y sus resultados.
    En caso de ingresar varios días, la función devuelve un diccionario
    con un dataframe para cada día.
    Los datos ingresados deben estar en formato str. """

    Headers = ['31.5 Hz', '63 Hz', '125 Hz', '250 Hz', '500 Hz', '1 kHz',
    '2 kHz', '4 kHz', '8 kHz', 'Global']
    path = root + str(yy) + '_' + str(mn) + '/'
    if not type(dd) == str:
        datos_dict = npy_a_df_dias(yy, mn, dd)
        return datos_dict
    else:
        try:
            data = np.load(path + str(dd) + '.npy')
        except:
            print('No hay datos disponibles para el día ' + str(dd) + '.')
            return
        niv = data[:, 3:]
        tts = data[:, :3]
        # Check for time breaks:
        dif_t = tts-np.vstack((tts[0,:],tts[:-1,:]))
        aux = np.where(np.abs(dif_t[:,2])>3)[0]
        cortes = []
        ti = tts[0, :]
        for i in np.arange(len(aux)):
            if dif_t[aux[i],0] == 0 and dif_t[aux[i],1] == 1 and ((dif_t[aux[i],2]+60) <= 2):
                continue
            cortes.append(aux[i])
        cortes.append(dif_t.shape[0])
        didx = pd.date_range(start=datetime.datetime(year=int(yy), month=int(mn),
                day=int(dd), hour=int(ti[0]), minute=int(ti[1]), second=int(ti[2])),
                freq='1S', periods=cortes[0])
        datos_df = pd.DataFrame(niv[0:cortes[0],:], columns=Headers, index=didx)
        for i in np.arange(len(cortes)-1):
            didx = pd.date_range(start=datetime.datetime(year=int(yy), month=int(mn),
                    day=int(dd), hour=int(tts[cortes[i],0]), minute=int(tts[cortes[i],1]), second=int(tts[cortes[i],2])),
                    freq='1S', periods=cortes[i+1]-cortes[i])
            aux_df = pd.DataFrame(niv[cortes[i]:cortes[i+1],:], columns=Headers, index=didx)
            datos_df = pd.concat([datos_df, aux_df]).round(2)
        return datos_df

def escr_arr(yy, mn, dd, mat):
    """ Guardado de datos obtenidos e información horaria en un array
    de numpy '.npy'. """
    path = root + str(yy) + "_" + str(mn) + "/"
    if not os.path.exists(str(path)):
        # Creación de carpeta
        os.makedirs(str(path))
        print("\nSe creó la carpeta : '" + str(path) + "'")
    if os.path.isfile(str(path) + str(dd) + ".npy"):
        file = np.load(path + str(dd) + ".npy")
        if file.size == 0:
            file = mat
        else:
            file = np.vstack((file, mat))
        np.save(str(path) + str(dd) + ".npy", file)
    else:
        np.save(str(path) + str(dd) + ".npy", mat)
    return
