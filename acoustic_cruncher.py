from AJ_FFT_filter import FFT_transform as fftt
import numpy as np
import pandas as pd
import base64

from AJ_analisi_fotoacustica import fotoacustica as fa

import streamlit as st
import zipfile
from bokeh.plotting import figure
from bokeh.palettes import all_palettes
from scipy.interpolate import interp1d

from scipy import integrate

# ███████ ██ ██      ███████
# ██      ██ ██      ██
# █████   ██ ██      █████
# ██      ██ ██      ██
# ██      ██ ███████ ███████

@st.cache(allow_output_mutation=True)
def load_func(uploadfile):
    zf = zipfile.ZipFile(uploadfile)

    files = dict()
    for i, name in enumerate(zf.namelist()):
        files[i] = pd.read_csv(zf.open(name))
        if name == 'spectrum.txt':
            files[i] = files[i][16:-1]
            colonna = files[i].columns.tolist()[0]
            files[i]['x'] = files[i][colonna].str.split(expand = True)[0]
            files[i]['y'] = files[i][colonna].str.split(expand = True)[1]
            files[i] = files[i].drop([colonna], axis = 1)
            files[i] = files[i].astype({'x': 'float', 'y':'float'})

# self.data_row['upload year'] = self.data_row['DOI'].str.split(".", expand = True)[3]

    return files, zf.namelist()

def download_file(data, filename):
    testo = 'Download '+filename+'.csv'
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">'+testo+'</a>'
    st.markdown(href, unsafe_allow_html=True)

colori = all_palettes['Category20'][20]

fft = st.sidebar.radio('Calculate the spectrum:', [False, True])
num_spec = int(st.sidebar.text_input('num of spectrum',1))
second_filter = int(st.sidebar.text_input('second filter',700))
second_filter2 = int(st.sidebar.text_input('second filter',810))



down_file = st.text_input('Name on download file', 'sample')
pump = int(st.text_input('pump frquenze (Hz)',10000))
num_rep = int(st.text_input('number of armonics to read', 10))

if st.checkbox('Type of analysis Legend'):
    st.markdown('**test**: test the length of all the file in the zip, for the mean analysis. It takes a zip where all the files have 3 columns: the time, the volt and the frequency.')
    st.markdown('**mean**: plot the average frequence spectrum from all the files in the zip. It takes a zip where all the files have 3 columns: the time, the volt and the frequency.')
    st.markdown('**lock in**: Plot the time trace from a zip file, pasting together all the time traces in the zip file. It take a zip where each file has 2 columns: the time and the volt.')
    st.markdown('**compare**: plot together the average spectrum of different experiments. It needs a zip where each file has 2 coulms: x and y. Those file are produced by this app using the mean function and Downloading the results at the end. Each file must have as name the frequenzy of the pump in Hz.')
    st.markdown('**spectrum**: Plot the intesity of the lock int overinpose on the spectrum of the lamp. It takes a zip with each file has 2 columns: the time and the volt. An additional file called spectrum.txt must be present and with the spectral information of the pump. The first zip to be loaded has to be the background, then the menu to load the rest appears. For the background is not required the spectrum file.')


uploadfile = st.file_uploader('load zip file here', 'zip')
if uploadfile:
    files, files_names = load_func(uploadfile)

    for file in files:
        new_cols = []
        for name in files[file].columns:
            if 'VOLT' in name:
                name = '1 (VOLT)'
            new_cols.append(name)
        files[file].columns = new_cols

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

    test = st.selectbox('type of analysis', ('', 'test', 'mean', 'lockin', 'compare', 'spectrum'))
    if test == 'compare' or test == 'mean':
        scale_logy = st.radio('scale Y:', ['linear', 'log'])
        scale_logx = st.radio('scale X:', ['linear', 'log'])
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
                df_time['1 (VOLT)'] = volt
                tempo[j] = df_time
                df_fft[j] = freq
                j = j + 1

        trasf_x, trasf_medio = 0, 0

        if fft:
            df_trasf_y = pd.DataFrame()
            trasf_x = fftt(tempo[0]['time (s)'].to_numpy(), tempo[0]['1 (VOLT)'].to_numpy()).trasformata()[0]
            for i in range(len(tempo)):
                df_trasf_y[i] = fftt(tempo[i]['time (s)'].to_numpy(), tempo[i]['1 (VOLT)'].to_numpy()).trasformata()[1]
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
            p1 = figure(title='frequenzy trace', x_axis_label='kHz', y_axis_label='', x_axis_type = scale_logx, y_axis_type = scale_logy)
            p1.line(fft_x, (fft_medio - np.mean(fft_medio)) - np.min(fft_medio - np.mean(fft_medio) + 1e-7), line_width=2)

            df_to_save = pd.DataFrame()
            df_to_save['x'] = fft_x
            df_to_save['y'] = (fft_medio - np.mean(fft_medio)) - np.min(fft_medio - np.mean(fft_medio))
            download_file(df_to_save, down_file+' Linear scale')

            df_fft = pd.DataFrame(fft_x)
            df_fft.columns = ['x']
            df_fft['y'] = (fft_medio - np.mean(fft_medio)) - np.min(fft_medio - np.mean(fft_medio) + 1e-7)

            x_peaks = []
            y_peaks = []
            step = df_fft['x'].loc[1]
            xcoord = int(round((pump/1000)/step))
            for rep in range(1,num_rep):
                x_temp = []
                y_temp = []
                for j in range(30):
                    x_temp.append(df_fft['x'].iloc[int(xcoord*rep)-15+j])
                    y_temp.append(df_fft['y'].iloc[int(xcoord*rep)-15+j])
                df_temp = pd.DataFrame(x_temp)
                df_temp.columns = ['x']
                df_temp['y'] = y_temp
                x_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['x'])
                y_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['y'])
            x_peaks = np.array(x_peaks)
            y_peaks = np.array(y_peaks)

            df_spec = pd.DataFrame(df_fft['x'])
            df_spec.columns = ['x']
            df_spec['y'] = df_fft['y']
            df_peaks = np.zeros_like(df_spec)
            df_peaks = pd.DataFrame(df_peaks)
            df_peaks.columns = ['x','y']
            df_peaks['x'] = df_spec['x']

            for peak in range(len(x_peaks)):
                ind_peak = df_spec[df_spec['x'] == x_peaks[peak]].index.tolist()[0]
                df_peaks.iloc[ind_peak]['y'] = y_peaks[peak]

            trasf_peaksx, trasf_peaksy, trasf_tot_peaksx, trasf_tot_peaksy =  fftt(df_peaks['x'].to_numpy()*1000, df_peaks['y'].to_numpy()).trasformata()
            trasf_peaks = trasf_tot_peaksx, trasf_tot_peaksy

            p1.line(x_peaks, y_peaks, line_width=2, legend_label='trend')
            p1.legend.click_policy="hide"
            st.bokeh_chart(p1, use_container_width=True)

            p3 = figure(title='Peaks', x_axis_label='kHz', y_axis_label='', x_axis_type = scale_logx, y_axis_type = scale_logy)
            p3.line(df_peaks['x'].to_numpy(), df_peaks['y'].to_numpy(), line_width=2, legend_label='peaks')
            p3.line(x_peaks, y_peaks, line_width=2, legend_label='trend')
            p3.legend.click_policy="hide"
            st.bokeh_chart(p3, use_container_width=True)

            p4 = figure(title='pulse', x_axis_label='sec', y_axis_label='')
            p4.line(trasf_peaks[0], trasf_peaks[1], line_width=2)
            st.bokeh_chart(p4, use_container_width=True)

            f = interp1d(x_peaks, y_peaks)
            x_peaks_continued = np.linspace(x_peaks[0], x_peaks[-1], num=fft_x.shape[0], endpoint=True)
            y_peaks_continued = f(x_peaks_continued)
            _, _, trasf_cint_peaksx, trasf_cont_peaksy = fftt(x_peaks_continued*1000, y_peaks_continued).trasformata()
            trasf_cont_peaks = trasf_cint_peaksx, trasf_cont_peaksy

            p7 = figure(title='back transform trend', x_axis_label='sec', y_axis_label='')
            p7.line(trasf_cont_peaks[0], trasf_cont_peaks[1], line_width=2)
            st.bokeh_chart(p7, use_container_width=True)

            timei = st.slider('Select the time region', 0, len(tempo), 0)
            p2 = figure(title='time trace', x_axis_label='sec', y_axis_label='V', x_range=(0,0.1))
            p2.line(tempo[timei]['time (s)'].to_numpy(), tempo[timei]['1 (VOLT)'].to_numpy(), line_width=2)
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


        #  ██████  ██████  ███    ███ ██████   █████  ██████  ███████
        # ██      ██    ██ ████  ████ ██   ██ ██   ██ ██   ██ ██
        # ██      ██    ██ ██ ████ ██ ██████  ███████ ██████  █████
        # ██      ██    ██ ██  ██  ██ ██      ██   ██ ██   ██ ██
        #  ██████  ██████  ██      ██ ██      ██   ██ ██   ██ ███████


    if test == 'compare':
        p3 = figure(title='compare frequenzy data', x_axis_label='kHz', y_axis_label='', x_axis_type = scale_logx, y_axis_type = scale_logy)
        p5 = figure(title='compare frequenzy data', x_axis_label='kHz', y_axis_label='')
        p6 = figure(title='compare back transform pulses', x_axis_label='sec', y_axis_label='')
        p7 = figure(title='compare back transform trend', x_axis_label='sec', y_axis_label='')
        p4 = figure(title='compare frequenzy data normalized by pump frequenzy', x_axis_label='Hz/Hz', y_axis_label='',  x_axis_type = scale_logx, y_axis_type = scale_logy)

        def name_to_num(text):
            x = text.split('.')
            return float(x[0])/1000
        pump_list = list(map(name_to_num, files_names))

        for i, file in enumerate(files):

            x_peaks = []
            y_peaks = []

            files_x = files[file]['x'].copy()/1000
            files_y = files[file]['y'].copy()
            files_y = files_y - np.min(files_y) + 1e-7

            step = files_x.loc[1]
            xcoord = int(round(pump_list[i]/step))

            for rep in range(1,num_rep):
                x_temp = []
                y_temp = []
                for j in range(30):
                    x_temp.append(files_x.iloc[int(xcoord*rep)-15+j])
                    y_temp.append(files_y.iloc[int(xcoord*rep)-15+j])
                df_temp = pd.DataFrame(x_temp)
                df_temp.columns = ['x']
                df_temp['y'] = y_temp
                x_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['x'])
                y_peaks.append(df_temp[df_temp['y'] == df_temp['y'].max()].iloc[0]['y'])
            x_peaks = np.array(x_peaks)
            y_peaks = np.array(y_peaks)

            df_spec = pd.DataFrame(files_x)
            df_spec.columns = ['x']
            df_spec['y'] = files_y

            df_peaks = np.zeros_like(df_spec)
            df_peaks = pd.DataFrame(df_peaks)
            df_peaks.columns = ['x','y']
            df_peaks['x'] = df_spec['x']

            for peak in range(len(x_peaks)):
                ind_peak = df_spec[df_spec['x'] == x_peaks[peak]].index.tolist()[0]
                df_peaks.iloc[ind_peak]['y'] = y_peaks[peak]

            _, _, trasf_tot_peaksx, trasf_tot_peaksy = fftt(df_peaks['x'].to_numpy()*1000, df_peaks['y'].to_numpy()).trasformata()
            trasf_peaks = trasf_tot_peaksx, trasf_tot_peaksy

            f = interp1d(x_peaks, y_peaks)
            x_peaks_continued = np.linspace(x_peaks[0], x_peaks[-1], num=files_x.shape[0], endpoint=True)
            y_peaks_continued = f(x_peaks_continued)

            _, _, trasf_cint_peaksx, trasf_cont_peaksy = fftt(x_peaks_continued*1000, y_peaks_continued).trasformata()
            trasf_cont_peaks = trasf_cint_peaksx, trasf_cont_peaksy

            p3.line(files_x, files_y, line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))
            p3.line(x_peaks, y_peaks, line_width=2, color = colori[i], legend_label=str(pump_list[i]))
            p4.line(x_peaks/pump_list[i], y_peaks, line_width=2, color = colori[i], legend_label=str(pump_list[i]))
            p5.line(df_peaks['x'], df_peaks['y'], line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))
            p6.line(trasf_peaks[0], trasf_peaks[1], line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))
            p7.line(trasf_cont_peaks[0], trasf_cont_peaks[1], line_width=2, color = colori[i], legend_label='spectr '+str(pump_list[i]))

        p3.legend.click_policy="hide"
        st.bokeh_chart(p3, use_container_width=True)

        p4.legend.click_policy="hide"
        st.bokeh_chart(p4, use_container_width=True)

        p5.legend.click_policy="hide"
        st.bokeh_chart(p5, use_container_width=True)

        p6.legend.click_policy="hide"
        st.bokeh_chart(p6, use_container_width=True)

        p7.legend.click_policy="hide"
        st.bokeh_chart(p7, use_container_width=True)


        # ███████ ██████  ███████  ██████ ████████ ██████  ██    ██ ███    ███
        # ██      ██   ██ ██      ██         ██    ██   ██ ██    ██ ████  ████
        # ███████ ██████  █████   ██         ██    ██████  ██    ██ ██ ████ ██
        #      ██ ██      ██      ██         ██    ██   ██ ██    ██ ██  ██  ██
        # ███████ ██      ███████  ██████    ██    ██   ██  ██████  ██      ██


    if test == 'spectrum':
        uploadfile2 = dict()
        for i in range(num_spec):
            uploadfile2[i] = st.file_uploader('load zip spectrum file '+str(i)+' here', 'zip')

        if uploadfile2[num_spec-1]:

            media_dict = dict()
            media_vet_dict = dict()
            media_x_dict = dict()
            files2_dict = dict()
            integrale_dict = dict()

            for i in range(num_spec):
                files2, files_names = load_func(uploadfile2[i])
                media_vet = []

                files2[0]['y'] = files2[0]['y'] - files2[0]['y'].mean()

                for j in range(1,len(files2)):
                    media_vet.append(files2[j]['1 (VOLT)'].mean())
                    media = np.array(media_vet).mean()

                pump_region = files2[0][files2[0]['x'] > second_filter][files2[0]['x'] < second_filter2]
                pump_std = files2[0][files2[0]['x'] > 900][files2[0]['x'] < 1000]
                pump_trasmitted = pump_region[pump_region['y'] > pump_std['y'].mean()+5*pump_std['y'].std()]

                media_x = pump_trasmitted['x'].mean()
                integrale = integrate.trapz(pump_trasmitted['y'], x=pump_trasmitted['x'])

                media_dict[i] = media
                media_vet_dict[i] = media_vet
                media_x_dict[i] = media_x
                files2_dict[i] = files2[0]
                integrale_dict[i] = integrale

            media_vet_bkg = []
            for j in range(1,len(files)):
                media_vet_bkg.append(files[j]['1 (VOLT)'].mean())
                media_bkg = np.array(media_vet_bkg).mean()
            x_bkg = [i*3 for i in range(len(media_vet_bkg))]

            p1 = figure(title='Dark', x_axis_label='sec', y_axis_label='Volts')
            p1.scatter(x_bkg, media_vet_bkg, line_width=2)
            p1.line([x_bkg[0], x_bkg[-1]], [media_bkg,media_bkg], color = 'red', legend_label = 'average')
            st.bokeh_chart(p1, use_container_width=True)

            p1 = figure(title='filters', x_axis_label='sec', y_axis_label='Volts')
            for i in range(num_spec):
                p1.line(files2_dict[i]['x'], files2_dict[i]['y'], line_width=2, legend_label='filter '+str(i), color = colori[i])
            p1.legend.click_policy="hide"
            st.bokeh_chart(p1, use_container_width=True)

            def plot_spect(x_media, y_media, vet_media, spec_df, title):
                x_tot = []
                y_tot = []
                y_calib_max = []
                p1 = figure(title=title, x_axis_label='nm', y_axis_label='Volts', x_range = (second_filter, second_filter2))
                for i in range(num_spec):
                    p1.circle(x_media[i], y_media[i], line_width=2, legend_label='average', color = 'red', size=10)
                    x_media2 = np.zeros_like(vet_media[i]) + x_media[i]
                    p1.circle(x_media2, vet_media[i], line_width=2, legend_label='single scan', alpha=0.5, size = 7)

                    x_tot.append(x_media[i])
                    y_tot.append(y_media[i])
                    y_calib_max.append(spec_df[i]['y'].max())
                p1.line(x_tot, y_tot, line_width=2, legend_label='average line', color = 'red')
                for i in range(num_spec):
                    y_calib = (spec_df[i]['y']/max(y_calib_max))*max(y_tot)
                    p1.line(spec_df[i]['x'], y_calib, line_width=2, legend_label='filter', color = colori[i])
                p1.legend.click_policy="hide"
                st.bokeh_chart(p1, use_container_width=True)


            plot_spect(media_x_dict, media_dict, media_vet_dict, files2_dict, 'Absorption Spectrum')

            for i in range(num_spec):
                media_dict[i] = media_dict[i] - media_bkg
                media_vet_dict[i] = np.array(media_vet_dict[i]) - media_bkg

            plot_spect(media_x_dict, media_dict, media_vet_dict, files2_dict, 'Absorption Spectrum, dark subtraction')

            for i in range(num_spec):
                media_dict[i] = media_dict[i]/integrale_dict[i]
                media_vet_dict[i] = np.array(media_vet_dict[i])/np.array(integrale_dict[i])

            plot_spect(media_x_dict, media_dict, media_vet_dict, files2_dict, 'Absorption Spectrum normalized')
