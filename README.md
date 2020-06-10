# RAMON: RaspberryPi Acoustic Monitor.

Estación de monitoreo acústico urbano. Capaz de proveer monitoreo continuo, remoto y autónomo. El desarrollo se basa en un micrófono MEMS digital. Realiza las siguientes funciones:

* Grabación de ruido urbano de a ciclos de 30 s.
* Aplicación de filtro de ponderación "A".
* Ajuste compensatorio de la respuesta del micrófono.
* Análisis del audio por segundo.
* Cáculo de niveles equivalentes globales y por octava.

## ramon.py

Código principal de ejecución, diseñado para la ejecución continua del dispositivo. Capta el audio a la entrada y luego ejecuta el procesamiento de este en un nuevo *thread*, continuando así el proceso de grabación de y procesamiento de manera recursiva y simultánea.

## funciones.py

En este script se incluyen las definiciones útiles para la ejecución del dispositivo. Se incluyen funciones para la realización de:

* Filtrado por bandas de octava y tercio de octava.
* Filtros de ponderación de frecuencia A y C.
* Lectura de archivos de audio.
* Ponderación temporal *Slow* y *Fast*.
* Cálculo de nivel RMS.
* Cálculo de niveles sonoros, al ingresar el valor RMS de la calibración y su nivel en $dB_{SPL}$.
* Ejecución de la calibración del dispositivo.
* Aplicación de ajuste a los niveles por banda de octava o tercio de octava.
* Guardado de los niveles sonoros en archivos *.npy*, visualización y manejo de los datos guardado en un *DataFrame*, y guardado de estos en archivos *.h5*.

## ramon_GUI.py

GUI para el manejo de los datos obtenidos por la estación de monitoreo.

Tiene como finalidad:

* El procesamiento del archivo de volcado en formato .npy de la estación, generando y guardando en un archivo .hd5 la información en formato pandas dataframe.
* Generación de índice del tipo datetime para los dataframe.
* Elección del período a visualizar según día/s y hora/s.
* Integración temporal de acuerdo al intervalo elegido (en horas, minutos o segundos).
* Cálculo de percentiles, valores máximo y mínimo.
* Cálculo del niveles sonoro continuo equivalente día-tarde-noche $L_{den}$.
* Generación de gráficos según el análisis elegido.
* Guardado de la información en un archivo .xlsx.
