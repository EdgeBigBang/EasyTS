import os

from scipy.fftpack import fft
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots

# Convert sampling frequency to corresponding data points
Sampling_points_day = {"1h": 24, "15T": 96, "5T": 288}


def draw_day(file_path, frequency):
    '''Draw daily data

    Help understand the approximate cycle of the day

    Agrs:
        file_path(str): The path to read the dataset
        frequency(str): Sampling frequency of the dataset
    '''
    # Determine the number of sampling points
    points_number = Sampling_points_day[frequency]
    values = []
    Abnormal_date = []
    data_df = pd.read_csv(file_path, usecols=['OT','date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date',inplace=True)
    temp_date = pd.to_datetime(data_df.index.values[0])
    end_date = pd.to_datetime(data_df.index.values[-1])
    x = []
    count = 0
    fig = go.Figure( layout_title_text="Daily Data of Dataset" )
    while temp_date < end_date:
        count += 1
        next_date = temp_date + timedelta(days=1) - timedelta(minutes=1)
        data_of_each_day = data_df.loc[temp_date:next_date]

        if len(data_of_each_day) == points_number:
            if not x:
                x_index = data_of_each_day.index.tolist()
                x = [dt.strftime('%H:%M') for dt in x_index]
            values.append(list(data_of_each_day['OT']))
            y = (data_of_each_day['OT']).to_numpy()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name=str(temp_date.strftime('%Y-%m-%d'))))
        else:
            Abnormal_date.append(temp_date.strftime('%Y-%m-%d'))

        temp_date = temp_date + timedelta(days=1)
    mean_values = np.mean(values,axis=0)
    fig.add_trace(go.Scatter(x=x, y=mean_values, line=dict(color='royalblue', width=5, dash='dot'), name='mean'))
    if not x:
        print('Serious data missing')
        return

    # Get Dataset Name
    fig.show(renderer='browser')
    print(f'Total days: {count}.')
    print(f'Abnormal date: {Abnormal_date}.')


def draw_month(file_path):
    '''Draw monthly data

    Help understand the approximate cycle of the month

    Agrs:
        file_path(str): The path to read the dataset

    '''

    values = []
    data_df = pd.read_csv(file_path, usecols=['OT','date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date',inplace=True)
    temp_date = pd.to_datetime(data_df.index.values[0])
    end_date = pd.to_datetime(data_df.index.values[-1])

    count = 0
    fig = go.Figure( layout_title_text="Monthly Data of Dataset" )
    while temp_date < end_date:
        count += 1
        next_date = temp_date + relativedelta(months=1) - timedelta(minutes=1)
        test = data_df.loc[temp_date:next_date]
        y = test['OT'].to_numpy()

        x_index = test.index.tolist()
        x = [dt.strftime('%d %H:%M') for dt in x_index]
        values.append(list(test['OT']))
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=str(temp_date)))
        temp_date = next_date
    fig.show(renderer='browser')

    print(f'Total months: {count}.')

# Help understand extreme values at different frequencies
def draw_frequency_max(file_path, frequency):
    '''draw the max of frequency

        Help understand extreme values at different frequencies

        Agrs:
            file_path(str): The path to read the dataset
            frequency(str): Sampling frequency of the dataset

    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)
    # resample
    daily_max = data_df.resample(frequency).max()
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "Max of" + " " + frequency + " for " + dataset_name

    fig = px.line(daily_max,x=daily_max.index, y='OT', title=title)

    fig.show(renderer='browser')


def draw_box(file_path):
    '''Draw box

        Help to quickly understand the magnitude of a dataset

        Agrs:
            file_path(str): The path to read the dataset
    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "Box of " + dataset_name
    fig = px.box(data_df, y="OT", title=title)
    fig.show(renderer='browser')


def draw_histogram(file_path):
    '''
        Quickly understand the approximate distribution of data

        Agrs:
            file_path(str): The path to read the dataset
    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "Histogram of " + dataset_name
    fig = px.histogram(data_df, x="OT", color = 'OT', title=title)
    fig.show(renderer='browser')


def draw_line_with_slider(file_path):
    '''

        Help provide an overview of the entire data situation

        Agrs:
            file_path(str): The path to read the dataset
    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    fig = px.line(data_df, x='date', y='OT', title='Time Series with Rangeslider')
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show(renderer='browser')


def hide_weekend_or_workday(file_path, sign="workday"):
    '''

        Help hide weekend or weekday data

        Agrs:
            file_path(str): The path to read the dataset
            sign(str): Decide whether to hide weekdays or weekends
    '''

    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)

    if sign == 'weekend':
        data_df['weekday'] = data_df.index.weekday
        filtered_df = data_df[(data_df['weekday'] != 5) & (data_df['weekday'] != 6)]
        data_df = filtered_df.drop('weekday', axis=1)
    elif sign == "workday":
        data_df['weekday'] = data_df.index.weekday
        filtered_df = data_df[(data_df['weekday'] == 5) | (data_df['weekday'] == 6)]
        data_df = filtered_df.drop('weekday', axis=1)

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "Remove weekday data of" + dataset_name
    fig = px.scatter(data_df, y='OT', title=title)
    fig.show(renderer='browser')

#
def hide_hour(file_path, start=23, end=16):
    '''
        Help hide hourly data

        Agrs:
            file_path(str): The path to read the dataset
            start(int): Start time that needs to be hidden
            end(int): End time that needs to be hidden
    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)

    start_hour = start
    end_hour = end

    if end > start:
        data_df = data_df[~((data_df.index.hour >= start_hour) & (data_df.index.hour < end_hour))]
    elif end < start:
        data_df = data_df[~((data_df.index.hour >= start_hour) | (data_df.index.hour < end_hour))]

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "Hide work time data of" + dataset_name

    fig = px.scatter(data_df, y='OT',
                     title=title)
    fig.show(renderer='browser')

def draw_FFT_count(file_path, window_size=288, m=6):
    '''
        Help to calculate the frequency of Fourier main components in time series segments

        Agrs:
            file_path(str): The path to read the dataset
            window_size(int): Fragment size for FFT
            m(int): How many Fourier components to choose
    '''
    data_df = pd.read_csv(file_path, usecols=['OT'])
    start_index = 0
    all_major = []
    while start_index < len(data_df):
        y = np.array(data_df)[start_index:start_index + window_size]
        data_fft = fft(y)
        data = abs(data_fft).flatten()
        # Remove the 0 value, which is usually the larger value for better observation
        data = data[1:]
        major_frequency = np.argsort(-data)[:m]
        all_major.extend(major_frequency.tolist())

        start_index = start_index + window_size
    major_count = Counter(all_major)
    df = pd.DataFrame(major_count.items(), columns=['Index', 'Count'])
    # Add missing frequency components
    for i in range(window_size):
        if i not in df['Index']:
            df = df._append({'Index':i,'Count':0},ignore_index=True)

    # Sort
    df = df.sort_values(by='Index')

    fig = px.bar(df, x='Index', y='Count', color = 'Count', color_continuous_scale='oryel',title="Statistics of Main Fourier Components in Window Fragments")
    fig.show(renderer='browser')


def draw_ma_decomp(file_path, window_size=24):
    '''
        Using MA for temporal data decomposition

         Agrs:
            file_path(str): The path to read the dataset
            window_size(int): Window size of MA
    '''
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df = data_df[:3000]

    data_df['Moving Average'] = data_df['OT'].rolling(window=window_size).mean()

    # Calculate residuals
    data_df['Residual'] = data_df['OT'] - data_df['Moving Average']

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Raw", "Trend", "Residual"),shared_xaxes=True)

    fig.add_trace(go.Scatter(
        x=data_df['date'],
        y=data_df['OT'], mode="lines", name="Raw"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data_df['date'],
        y=data_df['Moving Average'], mode="lines", name="Trend"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=data_df['date'],
        y=data_df['Residual'], mode="lines", name="Residual"
    ), row=3, col=1)

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    title = "STL of " + dataset_name

    fig.update_layout(height=500, width=600, title_text=title)
    fig.show(renderer='browser')


if __name__ == "__main__":
    ## Dataset Path
    # file_path = "../Electricity/OFFICEh.csv"
    file_path = "../Electricity/OFFICEm.csv"

    ## Using different analytical methods
    # draw_month(file_path)
    # draw_day(file_path,'5T')
    # draw_frequency_max(file_path, '1D')
    # draw_histogram(file_path)
    # draw_box(file_path)
    # draw_line_with_slider(file_path)
    # hide_weekend_or_workday(file_path)
    # hide_hour(file_path)
    draw_FFT_count(file_path)
    # draw_ma_decomp(file_path)