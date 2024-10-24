from typing import Dict, List, Any
import pandas as pd

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    result = []
    for i in range(0, len(lst), n):
        #manually reverse the current chunk of size n
        chunk = lst[i:i + n]
        #swap elements in the chunk
        for j in range(len(chunk)//2):
            chunk[j], chunk[len(chunk) -1 -j] = chunk[len(chunk) -1 -j], chunk[j]
        #add the reversed chunk to the result
        result.extend(chunk)
    return result
#test cases
print('\n 1.Reverse by n elements:')
print(reverse_by_n_elements([1,2,3,4,5,6,7,8], 3))
print(reverse_by_n_elements([1,2,3,4,5], 2))    
print(reverse_by_n_elements([10,20,30,40,50,60,70], 4))


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    result={}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    #sort the Dict
    sorted_result=dict(sorted(result.items()))
    return sorted_result
#test cases
print("\n 2.Group the strings by their length:")
print(group_by_length(["apple","bat","car","elephant","dog","bear"]))
print(group_by_length(["one","two","three","four"]))


def flatten_dict(nested_dict: Dict[str, Any], sep: str='.') -> Dict[str,Any]:
    def _flatten(current_item: Any, parent_key: str='') -> None:
        #traverse the dict or list recursively
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                _flatten(value, new_key)
        elif isinstance(current_item, list):
            for i, item in enumerate(current_item):
                new_key = f"{parent_key}[{i}]"
                _flatten(item, new_key)
        else:
            flatten_dict[parent_key] = current_item #for non-dict, non list items add to result
    #initiate empty dict to hold flattenes structure
    flatten_dict={}
    _flatten(nested_dict)
    return flatten_dict            
nested_dict={
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id":1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}
import json
result = flatten_dict(nested_dict)
print("\n 3.Flatten a nested dict into a single dict:")
formatted_result = json.dumps(result, indent=4)
print(formatted_result)


def unique_permutataions(nums: List[int]) -> List[List[int]]:
    def backtrack(permutation: List[int], used: List[bool]) -> None:
        if len(permutation) == len(nums):
            result.append(permutation[:])
            return
        for i in range(len(nums)):
            #skip used elements or duplicates
            if used[i]:         
                continue
            if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                continue
            #Mark the element as used and add
            used[i] = True
            permutation.append(nums[i])
            #Recurse to generate permutation for futher
            backtrack(permutation, used)
            #Backtrack: undo the choice
            used[i] = False
            permutation.pop()
    #sort the input list for handle duplicates
    nums.sort()
    result = []
    used = [False]* len(nums)
    backtrack([], used)
    return result
#Generate unique permutations
permutations = unique_permutataions([1, 1, 2])
print("\n 4.All unique permutations:")
print("[")
for i,perm in enumerate(permutations):
    if i<len(permutations)-1:
        print(f"    {perm},")
    else:
        print(f"    {perm}")    
print("]")


import re
def find_all_dates(text: str) -> List[str]:
    #date formats
    date_patterns =[
        r'\b\d{2}-\d{2}-\d{4}\b',   #dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',   #mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  #yyyy.mm.dd
    ]            
    #initialize a list to store
    matches = []
    #find all matches for each patterns
    for pattern in date_patterns:
        matches.extend(re.findall(pattern,text))
    return matches
text = "I was born on 22-08-1994, my friend on 08/23/1994, and another one on 1994.08.23"
print("\n 5.List of valid date formats:")
print(find_all_dates(text))


import polyline
import numpy as np
#haversine formula for calculate diatance b/w two lat/lon
def haversine(lat1, lon1, lat2, lon2):
    R=6371000 #radius of earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 -lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2* np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
#function to decode polyline and convert to df with distances
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    coords = polyline.decode(polyline_str)
    df = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    distances = [0]
    for i in range(1, len(coords)):
        lat1, lon1 = coords[i-1]
        lat2, lon2 = coords[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    df['distance'] = distances
    return df
#example
polyline_str = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
result_df = polyline_to_dataframe(polyline_str)
print("\n 6.Decode polyline and covert to df the distances:")
print(result_df.head())    


def rotate_and_multiply_matrix(matrix):
    rotated_matrix = np.rot90(matrix, -1).tolist()
    n = len(matrix)
    transformed_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum
    return rotated_matrix, transformed_matrix
#test cases
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rotated_matrix, transformed_matrix = rotate_and_multiply_matrix(matrix)
print(f"\n 7.Rotated Matrix = {rotated_matrix}")
print(f"\n Transformed Matrix = {transformed_matrix}")


df = pd.read_csv('C:/Users/ELCOT/Desktop/mapup_assesment_ans/MapUp-DA-Assessment-2024/datasets/dataset-1.csv')
print("\n 8.Dataset head:")
print(df.head())
#convert day names to numbers
day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
#add numerical rep of days to df
df['startDayNum'] = df['startDay'].map(day_mapping)
df['endDayNum'] = df['endDay'].map(day_mapping)
def time_check(df: pd.DataFrame) -> pd.Series:
    df_grouped = df.groupby(['id', 'id_2'])
    coverage_status = pd.Series(index=df_grouped.groups.keys(), dtype=bool)
    for (id_val, id_2_val), group in df_grouped:
        day_coverage = set()
        full_24_hour_days = True
        for _, row in group.iterrows():
            for day in range(row['startDayNum'], row['endDayNum'] + 1):
                day_coverage.add(day)
                #check the day is missing
                if day == row['startDayNum']:
                    if row['startTime'] != '00:00:00':
                        full_24_hour_days = False
                if day == row['endDayNum']:
                    if row['endDayNum'] != '23:59:59':
                        full_24_hour_days = False
        #check all 7 days
        has_all_days = len(day_coverage) == 7
        is_complete = has_all_days and full_24_hour_days
        coverage_status[(id_val, id_2_val)] = not is_complete
    return coverage_status
#apply the time check function
time_check_result = time_check(df)
print("\n 8.Completeness of time coverage:")
print(time_check_result.head())
                     