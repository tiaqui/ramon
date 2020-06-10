import tkinter as tk
import tkinter.scrolledtext, tkinter.messagebox, tkinter.filedialog
from tkinter import ttk
from funciones import mean_db, guardar_h5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import numpy as np
import os

def ayuda():
    text = """GUI creada para el manejo de la estación de monitoreo.\n
El dispositivo una vez conectado realiza análisis y almacenamiento de niveles por segundo, actualizando los datos
cada 10 segundos.\n
Para comenzar a operar presione el boton 'Cargar', una vez realizado esto se mostrarán los archivos
disponibles para ser procesados y luego analizados.\n\n
Luego, es posible graficar los mismos y almacenarlos en formato .xlsx"""
    tk.messagebox.showinfo('Ayuda', text)
    return

def cargar(r='C:/Users/tomia/Documents/Python/Notebooks/'):
    global root
    root = r
    folders = []
    npy = []
    h5 = []
    Archivos.configure(state='normal')
    Archivos.delete(1.0, tk.END)
    try:
        for item in os.listdir(root):
            if item[:2] == '20' and item[4] == '_':
                folders.append(item)
        if folders == []:
            text = """No se encontraron datos.\n
Ingrese a continuación la ubicación donde se encuentran
los archivos registrados por la estación de monitoreo."""
            tk.messagebox.showwarning('Cargar', text)
            root = tk.filedialog.askdirectory() + '/'
            cargar(root)
        for folder in folders:
            for item in os.listdir(root+folder):
                if item[-4:] == '.npy':
                    npy.append(folder+'/'+item)
                if item[-3:] == '.h5':
                    h5.append(folder+'/'+item)
        # Sin procesar:
        if npy != []:
            Archivos.insert('insert', 'A procesar:\n')
            for arch in npy:
                # Año
                Archivos.insert('insert', '\n' + arch[:4] + '/')
                # Mes
                if arch[6] == '/':
                    Archivos.insert('insert', arch[5] + '/')
                else:
                    Archivos.insert('insert', arch[5:7] + '/')
                # Día
                if arch[-6] == '/':
                    Archivos.insert('insert', arch[-5])
                else:
                    Archivos.insert('insert', arch[-6:-4])
            Archivos.insert('insert', '\n\n' + 12*' -' + '\n')
        else:
            Procesar.configure(state='disabled')
        # Procesados:
        if h5 != []:
            Archivos.insert('insert', 'Procesados:\n')
            for arch in h5:
                store = pd.HDFStore(root + arch, 'r')
                for key in store.keys():
                    hs = set(store[key].index.hour)
                    # if len(hs) == 23 o 24 ---->  'Completo' o ''.
                    # Año
                    iter_yy = arch[:4]
                    Archivos.insert('insert', '\n' + iter_yy + '/')
                    # Mes
                    if arch[6] == '/':
                        iter_mn = arch[5]
                    else:
                        iter_mn = arch[5:7]
                    Archivos.insert('insert', iter_mn + '/')
                    # Día
                    iter_dd = key[1:]
                    Archivos.insert('insert', iter_dd + ':' + '\n')
                    # Horas
                    Archivos.insert('insert', str(hs)[1:-1] + ' hs.\n')
        store.close()
        Archivos.configure(state='disabled')
        Mn_fin.configure(state='normal')
        Dd_fin.configure(state='normal')
        Mn_inic.delete(0, tk.END)
        Mn_fin.delete(0, tk.END)
        Dd_inic.delete(0, tk.END)
        Dd_fin.delete(0, tk.END)
        Mn_inic.insert(tk.END, iter_mn)
        Mn_fin.insert(tk.END, iter_mn)
        Dd_inic.insert(tk.END, iter_dd)
        Dd_fin.insert(tk.END, iter_dd)
        if mas1d.get() == 0:
            Mn_fin.configure(state='disabled')
            Dd_fin.configure(state='disabled')
    except Exception as e:
        text = """No se encontraron datos.\n
Ingrese a continuación la ubicación donde se encuentran
los archivos registrados por la estación de monitoreo."""
        tk.messagebox.showwarning('Cargar', text)
        root = tk.filedialog.askdirectory() + '/'
        #cargar(root)
        print(e)
    return npy, root

def procesar():
    s_proc, root = cargar()
    if s_proc == []:
        tk.messagebox.showwarning('Procesar', 'No hay datos a procesar')
    else:
        #folders = np.zeros(shape=(len(s_proc),2))
        for arch in s_proc:
            yy = arch[:4]
            if arch[6] == '/':
                mn = arch[5]
            else:
                mn = arch[5:7]
            if arch[-6] == '/':
                dd = arch[-5]
            else:
                dd = arch[-6:-4]
            guardar_h5(yy, mn, dd, root)
        cargar(root)
    return

def hab_porh():
    if porh.get() == 1:
        Hh_a.configure(state='readonly')
        Hh_b.configure(state='readonly')
    if porh.get() == 0:
        Hh_a.configure(state='disabled')
        Hh_b.configure(state='disabled')
    return

def hab_mas1d():
    if mas1d.get() == 1:
        Yy_fin.configure(state='normal')
        Mn_fin.configure(state='normal')
        Dd_fin.configure(state='normal')
    if mas1d.get() == 0:
        Yy_fin.configure(state='disabled')
        Mn_fin.configure(state='disabled')
        Dd_fin.configure(state='disabled')
    return

def get_days_df(root, a, b):
    try:
        daterange = pd.date_range(a, b)
        yy_idx = list(daterange.year)
        mn_idx = list(daterange.month)
        iter_yy = [yy_idx[0]]
    except:
        daterange = pd.date_range(b, a)
        yy_idx = list(daterange.year)
        mn_idx = list(daterange.month)
        iter_yy = [yy_idx[0]]
    iter_mn = [mn_idx[0]]
    j = 0
    aux_df = []
    proc_df = []
    for i in np.arange(len(mn_idx)):
        if iter_mn[j] != mn_idx[i]:
            iter_mn.append(mn_idx[i])
            iter_yy.append(yy_idx[i])
            j += 1
    for i in np.arange(len(iter_mn)):
        try:
            store = pd.HDFStore(root + str(iter_yy[i]) + '_' + str(iter_mn[i]) + '/datos.h5', 'r')
            for key in store.keys():
                aux_df = store[key]
                if isinstance(proc_df, pd.DataFrame):
                    proc_df = pd.concat([proc_df, aux_df])
                else:
                    proc_df = aux_df.copy()
            store.close()
        except Exception as e:
            print(e)
            pass
    return proc_df

def lden(df_hs):
    lden_day = df_hs.between_time('07:00', '19:00', include_end=False).apply(mean_db).round(2)
    lden_eve = df_hs.between_time('19:00', '23:00', include_end=False).apply(mean_db).round(2)
    lden_nig = df_hs.between_time('23:00', '07:00', include_end=False).apply(mean_db).round(2)
    lden_df = pd.DataFrame(index=['Día', 'Tarde', 'Noche', 'Lden'], columns=df_hs.columns)
    if not any(lden_day.isna()) and not any(lden_eve.isna()) and not any(lden_nig.isna()):
        lden_lden = 10*np.log10((12*(10**((lden_day)/10)) + 4*(10**((lden_eve + 5)/10)) + 8*(10**((lden_nig + 10)/10)))/24)
    elif not any(lden_day.isna()) and not any(lden_eve.isna()):
        lden_lden = 10*np.log10((12*(10**((lden_day)/10)) + 4*(10**((lden_eve + 5)/10)))/16)
    elif not any(lden_day.isna()) and not any(lden_nig.isna()):
        lden_lden = 10*np.log10((12*(10**((lden_day)/10)) + 8*(10**((lden_nig + 10)/10)))/20)
    elif not any(lden_eve.isna()) and not any(lden_nig.isna()):
        lden_lden = 10*np.log10((4*(10**((lden_eve + 5)/10)) + 8*(10**((lden_nig + 10)/10)))/12)
    elif not any(lden_day.isna()):
        lden_lden = lden_day
    elif not any(lden_eve.isna()):
        lden_lden = lden_eve
    elif not any(lden_nig.isna()):
        lden_lden = lden_nig
    else:
        print('No hay datos disponibles para el cálculo del Lden.')
    lden_df.loc['Día'] = lden_day
    lden_df.loc['Tarde'] = lden_eve
    lden_df.loc['Noche'] = lden_nig
    lden_df.loc['Lden'] = lden_lden.round(2)
    return lden_df

def graficar(df, quant, Lden):
    %matplotlib qt
    pd.plotting.register_matplotlib_converters()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    style.use('ggplot')
    fig, ax = plt.subplots(figsize=[15, 5])
    ax.plot(df['Global'], color='r', label='Valores globales')
    ax.set_ylabel('$L_{eqA} [dBA]$')
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if isinstance(quant, pd.DataFrame) or isinstance(Lden, pd.DataFrame):
        fig2, ax2 = plt.subplots(figsize=[15, 5])
        ax2.plot(df['Global'], color='r', label='Valores globales', zorder=1)
        lim_l = df.index[0]
        lim_r = df.index[-1]
        if isinstance(quant, pd.DataFrame):
            i = 0
            for q in quant.iterrows():
                col_set = 'm'
                if i == 1:
                    col_set = 'c'
                i += 1
                ax2.hlines(q[1]['Global'], xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], linestyles='-.', color=col_set, label='Percentil ' + str(int(q[0]*100)), zorder=2)
        if isinstance(Lden, pd.DataFrame):
            ax2.hlines(Lden.loc['Lden']['Global'], xmin=ax2.get_xlim()[0], xmax=ax2.get_xlim()[1], linestyles='--', color='g', label='$L_{den}$', zorder=2)
        ax2.xaxis.set_major_locator(locator)
        ax2.xaxis.set_major_formatter(formatter)
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=lim_l, right=lim_r)
        ax2.legend()
    return

def guardar_xlsx(df, quant, Lden):
    savefile = tkinter.filedialog.asksaveasfilename(filetypes=[('Excel files', '*.xlsx')])
    with pd.ExcelWriter(savefile + '.xlsx') as writer:
        df.to_excel(writer, sheet_name='Resultados')
        if not isinstance(quant, list):
            quant.to_excel(writer, sheet_name='Percentiles')
        if not isinstance(Lden, list):
            Lden.to_excel(writer, sheet_name='Lden')
    return

def ejecutar():
    ex_df = []
    proc_df = []
    Mn_i = ''
    Dd_i = ''
    if len(Mn_inic.get()) == 2 and Mn_inic.get()[0] == '0':
        Mn_i = Mn_inic.get()[1]
    else:
        Mn_i = Mn_inic.get()
    if len(Dd_inic.get()) == 2 and Dd_inic.get()[0] == '0':
        Dd_i = Dd_inic.get()[1]
    else:
        Dd_i = Dd_inic.get()
    date_1 = (Yy_inic.get(), Mn_i, Dd_i)
    Interv = N_interv.get() + U_interv.get()[0]
    if 'M' in Interv:
        Interv += 'in'
    Perc = []
    quant = []
    Lden = []
    try:
        Perc.append(int(ex_perc1.get())/100)
    except:
        pass
    try:
        Perc.append(int(ex_perc2.get())/100)
    except:
        pass
    if mas1d.get() == 0:
        try:
            store = pd.HDFStore(root + date_1[0] + '_' + date_1[1] + '/datos.h5', 'r')
            if porh.get() == 1:
                proc_df = store[date_1[2]].between_time(Hh_a.get() + ':00', Hh_b.get() + ':00')
                if len(proc_df.index) == 0:
                    text = """No hay datos disponibles.
Revise el rango horario ingresado."""
                    tkinter.messagebox.showwarning('Ejecutar', text)
            else:
                proc_df = store[date_1[2]]
            store.close()
            if not Interv == '1S':
                ex_df = proc_df.resample(Interv).apply(mean_db).round(2)
            else:
                ex_df = proc_df.round(2)
        except Exception as e:
            text = """Error en la carga de datos.
Revise la fecha y el intervalo ingresado."""
            tkinter.messagebox.showerror('Ejecutar', text)
            print(e)
    elif mas1d.get() == 1:
        Mn_f = ''
        Dd_f = ''
        if len(Mn_fin.get()) == 2 and Mn_fin.get()[0] == '0':
            Mn_f = Mn_fin.get()[1]
        else:
            Mn_f = Mn_fin.get()
        if len(Dd_fin.get()) == 2 and Dd_fin.get()[0] == '0':
            Dd_f = Dd_fin.get()[1]
        else:
            Dd_f = Dd_fin.get()
        date_b = Yy_fin.get() + '/' + Mn_f + '/' + Dd_f
        date_a = date_1[0] + '/' + date_1[1] + '/' + date_1[2]
        proc_df = get_days_df(root, date_a, date_b)
        if porh.get() == 1:
            proc_df = get_days_df(root, date_a, date_b).between_time(Hh_a.get() + ':00', Hh_b.get() + ':00')
        else:
            proc_df = get_days_df(root, date_a, date_b)
        if not Interv == '1S':
            ex_df = proc_df.resample(Interv).apply(mean_db).round(2)
        else:
            ex_df = proc_df.round(2)
    if len(Perc) >= 1:
        quant = ex_df.quantile(Perc).round(2) # aplica percentil al dataframe procesado
    if ex_lden.get() == 1:
        Lden = lden(proc_df)
    if ex_plot.get() == 1:
        graficar(ex_df, quant, Lden)
    if ex_xlsx.get() == 1:
        guardar_xlsx(ex_df.dropna(), quant.dropna(), Lden)
    return proc_df, Interv

#####################################################################################################################
##   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ####   ##   ##
#####################################################################################################################

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Ventana y frames

window = tk.Tk()
window.title('ramon')
frame_A = tk.Frame(window)
frame_B = tk.Frame(frame_A)
frame_C = ttk.LabelFrame(window, text='Opciones de análisis', relief=tk.RAISED)
frame_D = ttk.LabelFrame(window, text='Rango de análisis', relief=tk.RAISED)
frame_E = tk.Frame(window)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Botones
tk.Button(frame_A, text='Cargar', font=('Arial bold', 10), height=1, width=8, command=cargar).grid(column=0, row=0)
tk.Button(frame_A, text='Ayuda', font=('Arial bold', 10), height=1, width=8, command=ayuda).grid(column=0, row=2)
Procesar = tk.Button(frame_A, text='Procesar', font=('Arial bold', 10), height=1, width=8, command=procesar)
Procesar.grid(column=0, row=1)
tk.Button(frame_E, text='Ejecutar', font=('Arial bold', 11), height=1, width=12, command=ejecutar).grid(column=0, row=6, pady=20, sticky=tk.E)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Texto de archivos sin procesar y procesados
Archivos = tk.scrolledtext.ScrolledText(frame_B, wrap='word', font=('courier new bold', 10), width=25, height=10, state='disabled')
Archivos.grid(column=0, row=0, padx=20)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Día para ejecución
tk.Label(frame_D, text='Mes').grid(column=0, row=2)
tk.Label(frame_D, text='Año').grid(column=0, row=1)
tk.Label(frame_D, text='Día').grid(column=0, row=3)
Yy_inic = tk.Entry(frame_D, width=4)
Mn_inic = tk.Entry(frame_D, width=4)
Dd_inic = tk.Entry(frame_D, width=4)
Yy_inic.grid(column=1, row=1)
Mn_inic.grid(column=1, row=2)
Dd_inic.grid(column=1, row=3)
Yy_inic.insert(tk.END, pd.datetime.now().year)
Mn_inic.insert(tk.END, pd.datetime.now().month)
Dd_inic.insert(tk.END, pd.datetime.now().day)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Opción de ajustes por hora
porh = tk.IntVar()
tk.Checkbutton(frame_D, text='Ajuste por hora', variable=porh, onvalue=1, offvalue=0, command=hab_porh).grid(column=2, row=0, columnspan=2, padx=20)
tk.Label(frame_D, text='Hora inicial').grid(column=2, row=2, sticky=tk.E)
tk.Label(frame_D, text='Hora final').grid(column=2, row=3, sticky=tk.E)
Hh_a = ttk.Combobox(frame_D, values=list(np.arange(0, 24)), width=2)
Hh_b = ttk.Combobox(frame_D, values=list(np.arange(0, 24)), width=2)
Hh_a.grid(column=3, row=2, sticky=tk.W)
Hh_b.grid(column=3, row=3, sticky=tk.W)
Hh_a.current(0)
Hh_b.current(23)
Hh_a.configure(state='disabled')
Hh_b.configure(state='disabled')

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Opción más de un día
mas1d = tk.IntVar()
tk.Checkbutton(frame_D, text='Más de un día', variable=mas1d, onvalue=1, offvalue=0, command=hab_mas1d).grid(column=4, row=0, columnspan=2)
tk.Label(frame_D, text='Año').grid(column=4, row=1, sticky=tk.E)
tk.Label(frame_D, text='Mes').grid(column=4, row=2, sticky=tk.E)
tk.Label(frame_D, text='Día').grid(column=4, row=3, sticky=tk.E)
Yy_fin = tk.Entry(frame_D, width=4)
Mn_fin = tk.Entry(frame_D, width=4)
Dd_fin = tk.Entry(frame_D, width=4)
Yy_fin.grid(column=5, row=1, sticky=tk.W)
Mn_fin.grid(column=5, row=2, sticky=tk.W)
Dd_fin.grid(column=5, row=3, sticky=tk.W)
Yy_fin.insert(tk.END, pd.datetime.now().year)
Mn_fin.insert(tk.END, pd.datetime.now().month)
Dd_fin.insert(tk.END, pd.datetime.now().day)
Yy_fin.configure(state='disabled')
Mn_fin.configure(state='disabled')
Dd_fin.configure(state='disabled')

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Intervalo para el análisis
tk.Label(frame_C, text='Intervalo:').grid(column=1, row=1, sticky=tk.W, pady=5)
N_interv = tk.Entry(frame_C, width=3)
N_interv.grid(column=2, row=1, sticky=tk.W)
N_interv.insert(tk.END, 1)
U_interv = tk.StringVar()
U_interv = ttk.Combobox(frame_C, values=['Horas', 'Minutos', 'Segundos'], width=9, state='readonly')
U_interv.grid(column=2, row=1, sticky=tk.E)
U_interv.current(0)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Opciones de ejecución
ex_xlsx = tk.IntVar()
ex_plot = tk.IntVar(None, 1)
tk.Checkbutton(frame_C, text='Graficar', variable=ex_plot, onvalue=1).grid(column=3, row=4, sticky=tk.W)
tk.Checkbutton(frame_C, text='Guardar en .xlsx', variable=ex_xlsx, onvalue=1).grid(column=2, row=4, sticky=tk.W)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Cálculos para ejecución
ex_lden = tk.IntVar()
ex_perc = tk.IntVar()
ex_perc1 = ttk.Combobox(frame_C, values=list(np.arange(0, 105, 5)), width=4, state='readonly')
ex_perc2 = ttk.Combobox(frame_C, values=list(np.arange(0, 105, 5)), width=4, state='readonly')
ex_perc1.grid(column=2, row=2)
ex_perc2.grid(column=2, row=3)
tk.Label(frame_C, text='Calcular percentiles').grid(column=1, row=2, sticky=tk.E)
tk.Checkbutton(frame_C, text='Calcular Lden', variable=ex_lden, onvalue=1).grid(column=1, row=4, pady=7)

# - - - - - - - - - - - - - - - - - - - - - - - - - -
# Ajuste de frames
frame_A.grid(column=0, row=0, columnspan=2)
frame_B.grid(column=1, row=0, rowspan=3, ipady=0)
frame_C.grid(column=0, row=1, padx=10, pady=10)
frame_D.grid(column=1, row=1, ipady=10, padx=5)
frame_E.grid(column=0, row=3, columnspan=4)

window.mainloop()
