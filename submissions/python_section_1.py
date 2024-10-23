
# Question 1: Reverse List by N Elementsa

def reverse_in_groups(lst, n):
    
    result = []

    for i in range(0, len(lst), n):
        group = []
        
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        
        result.extend(group)
    
    return result
    
print("Reverse List by N Elementsa", reverse_in_groups([1, 2, 3, 4, 5, 6, 7, 8], 3))

# Question 2: Lists & Dictionaries

def group_by_length(strings):
    length_dict = {}

    for string in strings:
        length = len(string)
        
        if length not in length_dict:
            length_dict[length] = []
        
        length_dict[length].append(string)
    
    return dict(sorted(length_dict.items()))
    
print("Lists & Dictionaries",group_by_length(["one", "two", "three", "four"]))


# Question 3: Flatten a Nested Dictionary

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    
    for k, v in d.items():
        # Create new key by concatenating parent key and current key
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        
        elif isinstance(v, list):
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        
        else:
            items.append((new_key, v))
    
    return dict(items)

# Input from the user
import ast

# Prompt the user to enter a dictionary in string format
input_str = input("Enter a dictionary")

try:
    d = ast.literal_eval(input_str)
    if not isinstance(d, dict):
        raise ValueError("Input is not a valid dictionary.")

    # Call the flatten_dict function
    flattened_dict = flatten_dict(d)

    # Print the result
    print("Flattened dictionary:")
    print(flattened_dict)
except Exception as e:
    print(f"Error: {e}")


def unique_permutations(nums):
    def backtrack(path, remaining, result):
        if not remaining:
            result.append(path)
            return
        
        seen = set()
        for i in range(len(remaining)):
            if remaining[i] not in seen:
                seen.add(remaining[i])
                
                backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:], result)

    result = []
    nums.sort()
    backtrack([], nums, result)
    return result

# Input from the user
input_str = input("Enter a list of numbers separated by spaces (e.g., 1 1 2): ")
# Convert the input string into a list of integers
nums = list(map(int, input_str.split()))

# Call the function
permutations = unique_permutations(nums)

# Print the result
print("Unique permutations:")
for perm in permutations:
    print(perm)


# Question 5: Find All Dates in a Text


import re
text=input()
def find_all_dates(text):
    
    date_pattern = r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    # to find all matches of the date pattern in the text
    dates = re.findall(date_pattern, text)
    
    return dates
print(find_all_dates(text))


# Question 6: Decode Polyline, Convert to DataFrame with Distances



import pandas as pd
import math
import polyline

# Function to calculate the distance between two points using the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula to calculate the distance
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Distance in meters

# Function to decode the polyline and calculate distances
def decode_polyline_to_df(polyline_str):
    # Step 1: Decode the polyline string into a list of (latitude, longitude) coordinates
    coordinates = polyline.decode(polyline_str)
    
    # Step 2: Convert the coordinates into a DataFrame
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Step 3: Initialize a distance column with the first row having 0 distance
    df['distance'] = 0.0
    
    # Step 4: Calculate the Haversine distance for each successive pair of coordinates
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude']
        lat2, lon2 = df.loc[i, 'latitude'], df.loc[i, 'longitude']
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df

# Example of polyline string
polyline_str = '_^&@hhsu&i|U_kahhkdnnheyq`@'

# Decode polyline and calculate distances
df_result = decode_polyline_to_df(polyline_str)

# Display the resulting DataFrame
print(df_result)



# Question 7: Matrix Rotation and Transformation


import numpy as np

def rotate_and_transform(matrix):
    n = len(matrix)
    
    # Step 1: Rotate the matrix 90 degrees clockwise
    rotated_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]

    # Step 2: Replace each element with the sum of all elements in its row and column, excluding itself
    final_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Sum of the row (excluding current element)
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            # Sum of the column (excluding current element)
            col_sum = sum(rotated_matrix[x][j] for x in range(n)) - rotated_matrix[i][j]
            # Set the element in final matrix as the sum of row_sum and col_sum
            final_matrix[i][j] = row_sum + col_sum

    return final_matrix

# Example matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Call the function and display the result
result = rotate_and_transform(matrix)
print(np.array(result))

# Question 8: Time Check


def check_time_coverage(df_group):
    full_week = set(range(7))  # Days from Monday (0) to Sunday (6)
    covered_days = set()

    for _, row in df_group.iterrows():
        # Get the start and end days
        start_day = row['startDay_num']
        end_day = row['endDay_num']
        
        # Add all days between start_day and end_day to the covered_days set
        day_range = range(start_day, end_day + 1) if start_day <= end_day else range(start_day, 7)
        covered_days.update(day_range)
        
        # Check if the time spans a full day (00:00:00 to 23:59:59)
        if not (row['startTime'] == pd.Timestamp("00:00:00").time() and row['endTime'] == pd.Timestamp("23:59:59").time()):
            return False  # Time period for this entry doesn't cover the full 24 hours
    
    # Ensure all 7 days are covered
    return full_week.issubset(covered_days)

import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\raki_singh\Downloads\dataset-1.csv')

# Make sure the necessary preprocessing is done (e.g., mapping days and converting times)
day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

# Apply day mapping to startDay and endDay
df['startDay_num'] = df['startDay'].map(day_map)
df['endDay_num'] = df['endDay'].map(day_map)

# Convert startTime and endTime into time objects for comparison
df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time

# Group the dataframe by (id, id_2) and apply the coverage check function
coverage_check = df.groupby(['id', 'id_2']).apply(check_time_coverage)

# Output the result
print(coverage_check)






