import pandas as pd
import numpy as np
df = pd.read_csv('C:/Users/ELCOT/Desktop/mapup_assesment_ans/MapUp-DA-Assessment-2024/datasets/dataset-2.csv')
print("\n Dataset head:")
print(df.head())
def calculate_distance_matrix(df) -> pd.DataFrame:
    locations = pd.unique(df[['id_start', 'id_end']].values.ravel('k'))
    locations.sort()
    n = len(locations)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(distance_matrix.values, 0)
    #populate the matrix with given distances
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist #symmetric for bidirectional travel
    #apply floyd-warshall algorithm
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i,k] + distance_matrix.at[k,j]:
                    distance_matrix.at[i,j] = distance_matrix.at[i,k] + distance_matrix.at[k,j]
    return distance_matrix
#Calculate the cumulative distance matrix
cumulative_distance_matrix = calculate_distance_matrix(df)
print("\n 9.Cumulative distance matrix:")
print(cumulative_distance_matrix)     
    

def unroll_distance_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    unrolled = (
        matrix
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'id_start', 'level_1': 'id_end', 0: 'distance'})
    )
    #filter out row
    unrolled = unrolled[unrolled['id_start'] != unrolled['id_end']]
    return unrolled
unrolled_df = unroll_distance_matrix(cumulative_distance_matrix)
print("\n 10.Cumulative distance matrix:")
print(unrolled_df.head(10))


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    ref_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1
    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()
    ids_within_threshold = avg_distances[
        (avg_distances['distance'] >= lower_bound) &
        (avg_distances['distance'] <= upper_bound)
    ]['id_start'].sort_values()
    return ids_within_threshold.reset_index(drop=True)
#example case
result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, 1001400)
print("\n 11.Example: 10% threshold of reference ID 1001400:")
print(result_ids)


data = {
    'id_start': [1001400] * 10,
    'id_end': [1001402, 1001404, 1001406, 1001408, 1001410, 1001412, 1001414, 1001416, 1001418, 1001420],
    'moto': [7.76, 23.92, 36.72, 54.08, 62.96, 75.44, 90.00, 100.56, 111.44, 121.76],
    'car': [11.64, 35.88, 55.08, 81.12, 94.44, 113.16, 135.00, 150.84, 167.16, 182.64],
    'rv': [14.55, 44.85, 68.85, 101.40, 118.05, 141.45, 168.75, 188.55, 208.95, 228.30],
    'bus': [21.34, 65.78, 100.98, 148.72, 173.14, 207.46, 247.50, 276.54, 306.46, 334.84],
    'truck': [34.92, 107.64, 165.24, 243.36, 283.32, 339.48, 405.00, 452.52, 501.48, 547.92]
}
df = pd.DataFrame(data)
def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    df['toll_moto'] = df['moto'] * 0.8
    df['toll_car'] = df['car'] * 1.2
    df['toll_rv'] = df['rv'] * 1.5
    df['toll_bus'] = df['bus'] * 2.2
    df['toll_truck'] = df['truck'] * 3.6
    return df
df_with_tolls = calculate_toll_rate(df)
print("\n 12.Dataframe with toll rates:")
print(df_with_tolls)


from datetime import datetime, time
data = {
    'id_start': [1001400, 1001400, 1001400, 1001400, 1001408, 1001408, 1001408, 1001408],
    'id_end': [1001402, 1001402, 1001402, 1001402, 1001410, 1001410, 1001410, 1001410],
    'distance': [9.7, 9.7, 9.7, 9.7, 11.1, 11.1, 11.1, 11.1],
    'start_day': ['Monday', 'Tuesday', 'Wednesday', 'Saturday', 'Monday', 'Tuesday', 'Wednesday', 'Saturday'],
    'start_time': ['00:00:00', '10:00:00', '18:00:00', '00:00:00', '00:00:00', '10:00:00', '18:00:00', '00:00:00'],
    'end_day': ['Friday', 'Saturday', 'Sunday', 'Sunday', 'Friday', 'Saturday', 'Sunday', 'Sunday'],
    'end_time': ['10:00:00', '18:00:00', '23:59:59', '23:59:59', '10:00:00', '18:00:00', '23:59:59', '23:59:59'],
    'moto': [6.21, 9.31, 6.21, 5.43, 7.10, 10.66, 7.10, 6.22],
    'car': [9.31, 13.97, 9.31, 8.15, 10.66, 15.98, 10.66, 9.32],
    'rv': [11.64, 17.46, 11.64, 10.19, 13.32, 19.98, 13.32, 11.66],
    'bus': [17.07, 25.61, 17.07, 14.94, 19.54, 29.30, 19.54, 17.09],
    'truck': [27.94, 41.90, 27.94, 24.44, 31.97, 47.95, 31.97, 27.97]
}
df = pd.DataFrame(data)
def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    def apply_discount(start_day, start_time, vehicle_rates):
        weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekend = ['Saturday', 'Sunday']
        if start_day in weekday:
            if time(0, 0, 0) <= start_time <= time(10, 0, 0):
                discount_factor = 0.8
            elif time(10, 0, 0) < start_time <= time(18, 0, 0):
                discount_factor = 1.2
            else:
                discount_factor = 0.8
        else:
            discount_factor = 0.7
        return {vehicle: rate * discount_factor for vehicle,rate in vehicle_rates.items()}
    df['toll_moto'] = 0
    df['toll_car'] = 0
    df['toll_rv'] = 0
    df['toll_bus'] = 0
    df['toll_truck'] = 0
    for index, row in df.iterrows():
        start_day = row['start_day']
        start_time = datetime.strptime(row['start_time'], "%H:%M:%S").time()
        vehicle_rates = {
            'moto': row['moto'],
            'car': row['car'],
            'rv': row['rv'],
            'bus': row['bus'],
            'truck': row['truck'],
        }
        adjusted_rates = apply_discount(start_day, start_time, vehicle_rates)
        df.at[index, 'toll_moto'] = adjusted_rates['moto']
        df.at[index, 'toll_car'] = adjusted_rates['car']
        df.at[index, 'toll_rv'] = adjusted_rates['rv']
        df.at[index, 'toll_bus'] = adjusted_rates['bus']
        df.at[index, 'toll_truck'] = adjusted_rates['truck']
    return df
df_with_time_based_tolls = calculate_time_based_toll_rates(df)
print("\n 13.Time-based toll rates:")
print(df_with_time_based_tolls)