# -*- coding: utf-8 -*-
"""
Descripción.
"""
try:
    import numpy as np
    import sounddevice as sd
    import threading
    from funciones import (filt_A, nb_oct, foct_sup, foct_inf, nb_oct, sum_db,
                          busca_cal, busca_ajuste, niveles, rms_t,
                          escr_arr as escribir)
    from time import localtime, time as tiempo

    print("Inicializando dispositivo.")
    sd.stop()

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Parámetros por defecto
    fs = 48000
    sd.default.samplerate = fs
    sd.default.channels = 1
    sd.default.device = 'snd_rpi_simple_card'
    dur = 30.0  # duración del ciclo de ejecución en segundos #
    durN = int(fs*dur)  # duración en muestras

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Creación de arreglos para los datos de salida del callback
    mat = np.array([])
    t = np.array([])
    niv_y = np.array([])
    y = np.zeros(len(nb_oct))
    buffer = np.array([])

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Recuperación de valores de calibración y compensación del dispositivo.
    vcal, ncal = busca_cal()
    ajuste = busca_ajuste()

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Funciones útiles para la ejecución de los ciclos de callback.
    def agregar_t(t, n=10):
        """ Recibe un arreglo lineal comprendiendo una hora, minuto y segundo,
        y la cantidad de filas a entregar, y entrega a la salida un arreglo
        en cuya primera fila está el arreglo ingresado, y en cada una de las
        subsiguientes se agrega un segundo. """
        t_agr = np.zeros([n, 3])
        t_agr[0,:] = t
        for i in np.arange(n-1):
            if not t_agr[i,2] == 59:
                t_agr[i+1,2] = t_agr[i,2]+1
                t_agr[i+1,1] = t_agr[i,1]
                t_agr[i+1,0] = t_agr[i,0]
            else:
                t_agr[i+1,2] = 0
                if not t_agr[i,1] == 59:
                    t_agr[i+1,1] = t_agr[i,1]+1
                    t_agr[i+1,0] = t_agr[i,0]
                elif not t_agr[i,0] == 23:
                    t_agr[i+1,1] = 0
                    t_agr[i+1,0] = t_agr[i,0]+1
                else:
                    t_agr[i+1,1] = 0
                    t_agr[i+1,0] = 0
        return t_agr

    def separar(yy, mn, dd, t_fin, mat):
        """ Escritura de datos por separado de cambiar el día durante la ejecución.
        Recibe el año, mes y día de inicio de la grabación, la estructura de
        tiempos del final de la grabación y la matriz con los datos a exportar. """
        idx = int(np.min(np.where(mat[:,0]==0)[0]))
        mat_prim = mat[:idx,:]
        mat_sec = mat[idx:,:]
        escribir(yy, mn, dd, mat_prim)
        escribir(t_fin.tm_year, t_fin.tm_mon, t_fin.tm_mday, mat_sec)
        return

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Funciones específicas de ejecución del dispositivo.
    def callback(indata, frames, time, status):
        """ Función principal para la ejecución continua del dispositivo
        utilizando threading. """
        global buffer
        buffer = indata[fs:]
        thr = threading.Thread(name='Procesado por threading', target=procesado, args=[buffer])
        thr.start()
        buffer = np.array([])
        return

    def procesado(x):
        """ Función principal para el procesado del audio a analizar para cada
        ciclo de ejecución, 'x'. """
        # Manejo de datos temporales.
        ahora = tiempo() # instante de arranque para obtener la duración del ciclo
        print('\nCalculando ...')
        inicio = localtime(tiempo()-len(x)/fs) # tiempo de inicio del audio a analizar
        fin = localtime() # tiempo final del audio a analizar
        t_inic = np.array([inicio.tm_hour, inicio.tm_min, inicio.tm_sec])
        dd = inicio.tm_mday
        mn = inicio.tm_mon
        yy = inicio.tm_year
        # Procesamiento de la señal de audio.
        xA = filt_A(x.reshape(-1)) # ponderación 'A'
        xA_filt = filt_oct(xA) # filtrado por octavas
        # Cálculo de niveles de presión con compensación.
        niv_y = np.transpose(niveles(rms_t(xA_filt), vcal, ncal)) + ajuste
        # Agregado de niveles globales.
        glob_y = np.apply_along_axis(sum_db, 1, niv_y).reshape(niv_y.shape[0],1)
        niv_y = np.hstack((niv_y, glob_y))
        # Agregado de tiempos de inicio para la matriz de salida.
        t = agregar_t(t_inic)
        mat = np.hstack((t, niv_y))
        # Escritura de la matriz de salida en el archivo de volcado '.npy'.
        if not inicio.tm_mday == fin.tm_mday: # si hubo cambio de día, separo los resultados
            separar(yy, mn, dd, fin, mat)
        else:
            escribir(yy, mn, dd, mat) # si no escribo directo la matriz
        # Impresión en consola de tiempo de ejecución y nivel global del ciclo.
        print('\nEjecución de ciclo completa. \nTiempo de ejecución : ' + str(np.around(tiempo() - ahora, 2)) + ' s\n')
        print('Nivel global = ' + str(np.around(glob_y, 2)) +' dBA')
        return

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Inicialización del stream de audio con sounddevice.
    with sd.InputStream(callback=callback, blocksize=durN):
        print("\nEjecutando. Presione 'q' para finalizar.")
        while True:
            response = input()
            if response in ('q', 'Q'):
                print("\nEjecución finalizada.")
                break

except KeyboardInterrupt:
    print('\nInterrupción de teclado. Ejecución finalizada.')
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
