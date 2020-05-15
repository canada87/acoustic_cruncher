import numpy as np
import streamlit as st
import pandas as pd

class fotoacustica():
    def __init__(self, file_name = ''):
        self.file_name = file_name

    def len_trace(self):
        nan_position = []

        dati = np.genfromtxt(fname = self.file_name, usecols=[0], delimiter=',')

        for i in range(len(dati)):
            if np.isnan(dati[i]):
                nan_position.append(i)

        return nan_position

    def read_df(self, df):
        df.columns = ['time (s)', 'Volt', 'Freq']
        return df['time (s)'].dropna(), df['Volt'].dropna(), df['Freq'].dropna()

    def read_file(self):

        # time_x = np.genfromtxt(fname = self.file_name, skip_header=1, usecols=[0], delimiter=',', max_rows=stop_time)
        # time_y = np.genfromtxt(fname = self.file_name, skip_header=1, usecols=[1], delimiter=',', max_rows=stop_time)
        #
        # fft_y = np.genfromtxt(fname = self.file_name, skip_header=1, usecols=[2], delimiter=',', max_rows=stop_fft)

        data_row = pd.read_csv(self.file_name, delimiter=',')
        data_row.columns = ['time (s)', 'Volt', 'Freq']
        # st.write(data_row['time (s)'].dropna())

        # offtime = time_x[0]
        # time_x = time_x - offtime
        #
        # # print(stop_time, time_x[-1])
        #
        # if np.any(np.isnan(time_x)):
        #     print('nan present timex')
        # if np.any(np.isnan(time_y)):
        #     print('nan present timey')
        # if np.any(np.isnan(fft_y)):
        #     print('nan present fft')

        return data_row['time (s)'].dropna(), data_row['Volt'].dropna(), data_row['Freq'].dropna()
