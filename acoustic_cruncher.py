from AJ_FFT_filter import FFT_transform as fftt
import numpy as np
import pandas as pd
import base64

from AJ_analisi_fotoacustica import fotoacustica as fa

import streamlit as st
import zipfile
from scipy.optimize import curve_fit
from bokeh.plotting import figure
from bokeh.palettes import all_palettes

# ███████ ██ ██      ███████
# ██      ██ ██      ██
# █████   ██ ██      █████
# ██      ██ ██      ██
# ██      ██ ███████ ███████

@st.cache()
def load_func(uploadfile):
    zf = zipfile.ZipFile(uploadfile)

    files = dict()
    for i, name in enumerate(zf.namelist()):
        files[i] = pd.read_csv(zf.open(name))
    return files, zf.namelist()

def download_file(data, filename):
    testo = 'Download '+filename+'.csv'
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">'+testo+'</a>'
    st.markdown(href, unsafe_allow_html=True)

fft = st.sidebar.radio('Calculate the spectrum:', [False, True])

down_file = st.text_input('Name on download file', 'sample')
pump = int(st.text_input('pump frquenze (Hz)',10000))
num_rep = int(st.text_input('number of armonics to read', 10))

uploadfile = st.file_uploader('load zip file here', 'zip')
if uploadfile:
    files, files_names = load_func(uploadfile)

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    # ████████ ███████ ███████ ████████
    #    ██    ██      ██         ██
    #    ██    █████   ███████    ██
    #    ██    ██           ██    ██
    #    ██    ███████ ███████    ██
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    test = st.selectbox('type of analysis', ('', 'test', 'mean', 'lockin', 'compare'))
    stop_time_correct = 64516

    @st.cache()
    def test_func(files):
        hz = []
        point = []
        num = []

        for i in files:
            temp_x, temp_y, temp_fft = fa().read_df(files[i])
            hz.append(round(1/((temp_x[1]-temp_x[0])*2)))
            point.append(len(temp_x))
            num.append(i)

        df_test = pd.DataFrame()
        df_test['Hz'] = hz
        df_test['point'] = point
        df_test['num'] = num
        return df_test

    if test == 'test':
        st.write('Files evaluation')
        df_test = test_func(files)
        st.write(df_test)

    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
            # ███    ███ ███████  █████  ███    ██
            # ████  ████ ██      ██   ██ ████   ██
            # ██ ████ ██ █████   ███████ ██ ██  ██
            # ██  ██  ██ ██      ██   ██ ██  ██ ██
            # ██      ██ ███████ ██   ██ ██   ████
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################

    @st.cache()
    def fft_func(files, fft):
        df_time = pd.DataFrame()
        tempo = dict()
        df_fft = pd.DataFrame()
        j = 0

        for i in files:
            times, volt, freq = fa().read_df(files[i])
            if len(times) == stop_time_correct:
                df_time = pd.DataFrame()
                df_time['time (s)'] = times
                df_time['volt'] = volt
                tempo[j] = df_time
                df_fft[j] = freq
                j = j + 1

        trasf_x, trasf_medio = 0, 0

        if fft:
            df_trasf_y = pd.DataFrame()
            trasf_x = fftt(tempo[0]['time (s)'].to_numpy(), tempo[0]['volt'].to_numpy()).trasformata()[0]
            for i in range(len(tempo)):
                df_trasf_y[i] = fftt(tempo[i]['time (s)'].to_numpy(), tempo[i]['volt'].to_numpy()).trasformata()[1]
            trasf_medio = df_trasf_y.mean(axis=1)

        return trasf_x, trasf_medio, df_fft, tempo

    if test == 'mean':
        divisore = float(st.text_input('Calibration (1->60kHz, 2->30kHz, 4->15kHz, 0.5->120KHz, 0.4->156kHz, 0.25->240KHz, 0.2->313KHz )', 0))

        if divisore != 0:
            trasf_x, trasf_medio, df_fft, tempo =  fft_func(files, fft)

            # # # ███████ ███████ ████████
            # # # ██      ██         ██
            # # # █████   █████      ██
            # # # ██      ██         ██
            # # # ██      ██         ██

            if fft:
                # #######################################################
                p = figure(title='FFT estimate', x_axis_label='Hz', y_axis_label='')
                p.line(trasf_x, trasf_medio, line_width=2)
                st.bokeh_chart(p, use_container_width=True)
                # #######################################################


            # ███████ ██████  ███████  ██████  ██    ██ ███████ ███    ██  ██████ ██    ██
            # ██      ██   ██ ██      ██    ██ ██    ██ ██      ████   ██ ██       ██  ██
            # █████   ██████  █████   ██    ██ ██    ██ █████   ██ ██  ██ ██        ████
            # ██      ██   ██ ██      ██ ▄▄ ██ ██    ██ ██      ██  ██ ██ ██         ██
            # ██      ██   ██ ███████  ██████   ██████  ███████ ██   ████  ██████    ██
            #                             ▀▀

            fft_medio = df_fft.mean(axis=1)
            step_new = 0.9536070185476565/divisore
            #######################################################
            #allineamento
            fft_x = np.linspace(0, step_new*df_fft.shape[0], df_fft.shape[0])/1000
            #######################################################

            # #######################################################
            p1 = figure(title='frequenzy trace (linear scale)', x_axis_label='kHz', y_axis_label='')
            p1.line(fft_x, (fft_medio - np.mean(fft_medio)) - np.min(fft_medio - np.mean(fft_medio)), line_width=2)
            st.bokeh_chart(p1, use_container_width=True)
            df_to_save = pd.DataFrame()
            df_to_save['x'] = fft_x
            df_to_save['y'] = (fft_medio - np.mean(fft_medio)) - np.min(fft_medio - np.mean(fft_medio))
            download_file(df_to_save, down_file+' Linear scale')

            fft_log = np.log(fft_medio)

            # yline = [2.7025,   0.18433]
            # xline = [30000.95, 150000]

            # yline = [2.1578,   0.6255]
            # xline = [6000, 19000]
            #
            #
            # m = (yline[0] - yline[1])/(xline[0] - xline[1])
            # st.write(m)

            df_fft = pd.DataFrame(fft_x)
            df_fft.columns = ['x']
            df_fft['y'] = fft_log - np.mean(fft_log) - np.min(fft_log - np.mean(fft_log))


            x_peaks = []
            y_peaks = []
            for rep in range(1,num_rep):
                x_temp = []
                y_temp = []
                for i in range(10):
                    x_temp.append(df_fft['x'].iloc[int(pump*rep/step_new)-5+i])
                    y_temp.append(df_fft['y'].iloc[int(pump*rep/step_new)-5+i])
                df_temp = pd.DataFrame(x_temp)
                df_temp.columns = ['x']
                df_temp['y'] = y_temp
                x_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['x'])
                y_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['y'])
            # st.write(x_peaks, y_peaks)
            x_peaks = np.array(x_peaks)
            y_peaks = np.array(y_peaks)

            def retta(x, p0, p1):
                return p0*x + p1

            par1, par2 = curve_fit(retta, x_peaks, y_peaks)
            yfit = retta(x_peaks, par1[0], par1[1])
            m = par1[0]
            q = par1[1]

            st.write('m:', m)

            p3 = figure(title='frequenzy trace (logaritmic scale)', x_axis_label='kHz', y_axis_label='')
            p3.line(fft_x, (fft_log - np.mean(fft_log)) - np.min(fft_log - np.mean(fft_log)), line_width=2)
            # p3.line(x_peaks, yfit, line_width=2, color='red')
            st.bokeh_chart(p3, use_container_width=True)

            df_to_save = pd.DataFrame()
            df_to_save['x'] = fft_x
            df_to_save['y'] = (fft_log - np.mean(fft_log)) - np.min(fft_log - np.mean(fft_log))
            download_file(df_to_save, down_file+' Log scale')

            timei = st.slider('Select the time region', 0, len(tempo), 0)
            p2 = figure(title='time trace', x_axis_label='sec', y_axis_label='V', x_range=(0,0.1))
            p2.line(tempo[timei]['time (s)'].to_numpy(), tempo[timei]['volt'].to_numpy(), line_width=2)
            st.bokeh_chart(p2, use_container_width=True)
            # #######################################################


            # ██       ██████   ██████ ██   ██ ██ ███    ██
            # ██      ██    ██ ██      ██  ██  ██ ████   ██
            # ██      ██    ██ ██      █████   ██ ██ ██  ██
            # ██      ██    ██ ██      ██  ██  ██ ██  ██ ██
            # ███████  ██████   ██████ ██   ██ ██ ██   ████


    if test == 'lockin':
        for file in files:
            files[file]['Time (s)'] = files[file]['Time (s)'] - files[file]['Time (s)'].min()

        for i in range(1, len(files)):
            files[i]['Time (s)'] = files[i]['Time (s)'] + files[i-1]['Time (s)'].max()

        p = figure(title='', x_axis_label='Time (s)', y_axis_label='V')
        ave = []
        for file in files:
            ave.append(files[file]['1 (VOLT)'].mean())
        ave_ave = np.array(ave).mean()
        st.write(ave_ave)
        for file in files:
            p.line(files[file]['Time (s)'], files[file]['1 (VOLT)'].ewm(span = 5).mean(), line_width=2)
            p.line((files[file]['Time (s)'].iloc[0], files[file]['Time (s)'].iloc[0]), (0, ave_ave+ave_ave*0.3), line_width=2, color = 'red')
        st.bokeh_chart(p, use_container_width=True)

    if test == 'compare':
        colori = all_palettes['Category20'][20]
        p3 = figure(title='compare frequenzy data', x_axis_label='Hz', y_axis_label='')#, x_axis_type = 'log')
        p4 = figure(title='compare frequenzy data normalized by pump frequenzy', x_axis_label='Hz/Hz', y_axis_label='')#, x_axis_type = 'log')

        def name_to_num(text):
            x = text.split('.')
            return int(x[0])
        pump_list = list(map(name_to_num, files_names))

        for i, file in enumerate(files):

            x_peaks = []
            y_peaks = []

            files[file]['x'] = files[file]['x']/1000
            pump_list[i] = pump_list[i]/1000

            # files[file]['x'] = np.log(files[file]['x'])
            # pump_list[i] = np.log(pump_list[i])


            step = files[file]['x'].loc[1]
            xcoord = int(round(pump_list[i]/step))

            for rep in range(1,num_rep):
                x_temp = []
                y_temp = []
                for j in range(30):
                    x_temp.append(files[file]['x'].iloc[int(xcoord*rep)-15+j])
                    y_temp.append(files[file]['y'].iloc[int(xcoord*rep)-15+j])
                df_temp = pd.DataFrame(x_temp)
                df_temp.columns = ['x']
                df_temp['y'] = y_temp
                x_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['x'])
                y_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['y'])
            x_peaks = np.array(x_peaks)
            y_peaks = np.array(y_peaks)

            # x_peaks_story = np.zeros(3*len(x_peaks))
            # y_peaks_story = np.zeros(3*len(y_peaks))
            # for j in range(len(x_peaks)):
            #     x_peaks_story[0+3*j] = x_peaks[j]
            #     x_peaks_story[1+3*j] = x_peaks[j]
            #     x_peaks_story[2+3*j] = x_peaks[j]
            #
            #     y_peaks_story[0+3*j] = np.min(y_peaks)
            #     y_peaks_story[1+3*j] = y_peaks[j]
            #     y_peaks_story[2+3*j] = np.min(y_peaks)

            p3.line(files[file]['x'], files[file]['y'], line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))
            # p3.line(x_peaks_story, y_peaks_story, line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))
            p3.line(x_peaks, y_peaks, line_width=2, color = colori[i], legend_label=str(pump_list[i]))

            p4.line(x_peaks/pump_list[i], y_peaks, line_width=2, color = colori[i], legend_label=str(pump_list[i]))

        p3.legend.click_policy="hide"
        st.bokeh_chart(p3, use_container_width=True)

        p4.legend.click_policy="hide"
        st.bokeh_chart(p4, use_container_width=True)
