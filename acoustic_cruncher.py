from AJ_FFT_filter import FFT_transform as fftt
import numpy as np
import pandas as pd

from AJ_analisi_fotoacustica import fotoacustica as fa

import streamlit as st
import zipfile

from bokeh.plotting import figure

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
    return files

uploadfile = st.file_uploader('load zip file here', 'zip')
if uploadfile:
    files = load_func(uploadfile)

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

    test = st.selectbox('type of analysis', ('', 'test', 'mean', 'lockin'))
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
    def fft_func(files):
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

        df_trasf_y = pd.DataFrame()
        trasf_x = fftt(tempo[0]['time (s)'].to_numpy(), tempo[0]['volt'].to_numpy()).trasformata()[0]

        for i in range(len(tempo)):
            df_trasf_y[i] = fftt(tempo[i]['time (s)'].to_numpy(), tempo[i]['volt'].to_numpy()).trasformata()[1]
        trasf_medio = df_trasf_y.mean(axis=1)
        return trasf_x, trasf_medio, df_fft, tempo

    if test == 'mean':
        divisore = float(st.text_input('Calibration (1->60kHz, 2->30kHz, 4->15kHz, 0.5->120KHz, 0.25->240KHz, 0.2->313KHz )', 0))

        if divisore != 0:
            trasf_x, trasf_medio, df_fft, tempo =  fft_func(files)

            # # # ███████ ███████ ████████
            # # # ██      ██         ██
            # # # █████   █████      ██
            # # # ██      ██         ██
            # # # ██      ██         ██

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

            fft_x = np.linspace(0, trasf_x[-1], df_fft.shape[0])
            fft_medio = df_fft.mean(axis=1)

            step_new = 0.9536070185476565/divisore
            # st.write(step_new*len(fft_x))
            #######################################################
            #allineamento
            fft_x = np.linspace(0, step_new*len(fft_x), len(fft_x))
            #######################################################

            # #######################################################
            p1 = figure(title='frequenzy trace', x_axis_label='Hz', y_axis_label='')
            p1.line(fft_x, (fft_medio - np.mean(fft_medio))*1000, line_width=2)
            st.bokeh_chart(p1, use_container_width=True)

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
