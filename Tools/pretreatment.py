import os
import pandas as pd



def fill_and_smooth(file_path, frequency, max_missing=5):
    '''Smoothing missing values, discarding data if there are more than [max_missing] consecutive missing values
    
        Args:
            file_path(str): Dataset file path
            frequency(str): Dataset sampling frequency
            max_missing(int): Discard threshold, maximum number of consecutive misses
    
    '''
    # Determine the number of sampling points
    data_df = pd.read_csv(file_path, usecols=['OT', 'date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date', inplace=True)
    temp_date = pd.to_datetime(data_df.index.values[0])
    end_date = pd.to_datetime(data_df.index.values[-1])
    full_day_index = pd.date_range(start=temp_date, end=end_date, freq=frequency)
    data_df = data_df[~data_df.index.duplicated()]
    data_df = data_df.reindex(full_day_index)

    missing_temp_dates = []
    missing_dates = []
    discard_dates = []
    missing_count = 0

    for i in range(len(data_df)):
        if pd.isna(data_df.iloc[i]["OT"]):
            missing_date = pd.to_datetime(data_df.index[i]).date()
            missing_temp_dates.append(missing_date)
            missing_dates.append(missing_date)
            missing_count += 1
            if missing_count >= max_missing:
                for i in list(set(missing_temp_dates)):
                    discard_dates.append(i)
        else:
            missing_count = 0
            missing_temp_dates = []
    discard_dates = list(set(discard_dates))
    df_discard_dates = pd.DataFrame(discard_dates)
    file_name = file_path.split('/')[-1].split('.')[0]
    abnormal_file_path = file_path.replace(file_name, file_name + '_abnormal_days')
    df_discard_dates.to_csv(abnormal_file_path)

    missing_dates = list(set(missing_dates))
    df_missing_dates = pd.DataFrame(missing_dates)
    missing_file_path = file_path.replace(file_name, file_name + '_missing_days')
    df_missing_dates.to_csv(missing_file_path)

    data_df = data_df[[day not in discard_dates for day in data_df.index.date]]
    data_df = data_df.resample(frequency).mean().interpolate()
    # Restore the date column
    data_df = data_df.rename_axis('date')
    new_file_path = file_path.replace(file_name, file_name+'_pre')
    data_df.to_csv(new_file_path)


def change_sampling_frequency(file_path, frequency):
    '''Change the sampling frequency of the dataset

        Change the dataset to the sampling frequency specified by frequency
        and save it to the directory where the file is located

        Args:
            file_path(str): Dataset file path
            frequency(str): frequency：['15T','1D']

    '''

    data_df = pd.read_csv(file_path, usecols=['OT','date'])
    data_df['date'] = pd.to_datetime(data_df['date'])
    data_df = data_df.sort_values(by='date')
    data_df.set_index('date',inplace=True)

    # resample
    daily_max = data_df.resample(frequency).max()
    daily_max = daily_max.rename_axis('date')

    # Define a new file name
    directory = os.path.dirname(file_path)
    old_filename = os.path.basename(file_path)
    new_filename = old_filename.split('.')[0] + "_" + frequency + ".csv"
    save_path = os.path.join(directory, new_filename)

    daily_max.to_csv(save_path, mode='w', encoding='UTF-8')

if __name__ == "__main__":
    file_path = "../Electricity/htyqAzbdz_YC0037_m5_pre.csv"
    # fill_and_smooth(file_path, '1h')
    change_sampling_frequency(file_path,"15T")