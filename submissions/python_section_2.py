import pandas as pd
import numpy as np
from datetime import time

# Load the dataset to inspect its contents
dataset = pd.read_csv(r'dataset-2.csv')


# Extract unique IDs from the dataset to create the distance matrix
unique_ids = np.unique(dataset[['id_start', 'id_end']])
num_ids = len(unique_ids)

# Create a DataFrame for the distance matrix, initialized with infinity values
distance_matrix = pd.DataFrame(np.inf, index=unique_ids, columns=unique_ids)

# Fill in the direct distances from the dataset
for _, row in dataset.iterrows():
    start, end, dist = row['id_start'], row['id_end'], row['distance']
    distance_matrix.at[start, end] = dist
    distance_matrix.at[end, start] = dist  # Symmetric entry

# Set the diagonal to 0 (distance from a node to itself)
np.fill_diagonal(distance_matrix.values, 0)

# Apply the Floyd-Warshall algorithm to calculate cumulative distances
for k in unique_ids:
    for i in unique_ids:
        for j in unique_ids:
            distance_matrix.at[i, j] = min(distance_matrix.at[i, j], distance_matrix.at[i, k] + distance_matrix.at[k, j])

# Display the resulting distance matrix
print("Display the resulting distance matrix", distance_matrix.head())


# Question 10: Unroll Distance Matrix


def unroll_distance_matrix(distance_matrix):
    
    # Get the index and column names (unique IDs) from the distance matrix
    id_pairs = [
        {'id_start': i, 'id_end': j, 'distance': distance_matrix.at[i, j]}
        for i in distance_matrix.index
        for j in distance_matrix.columns
        if i != j
    ]
    
    # Create a DataFrame from the list of dictionaries
    unrolled_df = pd.DataFrame(id_pairs)
    return unrolled_df

# Unroll the previously computed distance matrix
unrolled_df = unroll_distance_matrix(distance_matrix)

# Display the first few rows of the unrolled DataFrame
print("Display the first few rows of the unrolled DataFrame unrolled_df",unrolled_df.head())



#Display the first few rows of the unrolled DataFrame


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    
    # Filter the DataFrame for the reference id_start
    reference_rows = unrolled_df[unrolled_df['id_start'] == reference_id]
    
    # Calculate the average distance for the reference ID
    reference_avg = reference_rows['distance'].mean()
    
    # Define the 10% threshold range
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1
    
    # Calculate the average distance for all other id_start values
    id_avg_distances = unrolled_df.groupby('id_start')['distance'].mean()
    
    # Filter IDs that have their average distances within the 10% threshold
    ids_within_threshold = id_avg_distances[(id_avg_distances >= lower_bound) & (id_avg_distances <= upper_bound)].index.tolist()
    
    # Sort the result
    ids_within_threshold_sorted = sorted(ids_within_threshold)
    
    return ids_within_threshold_sorted

# Example usage: find IDs within 10% threshold of a reference value (e.g., 1001400)
reference_id = 1001400
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

# Display the result
print("Display the first few rows of the unrolled DataFrame",ids_within_threshold)


# Question 12: Calculate Toll Rate


# Load the uploaded dataset
df = pd.read_csv(r'dataset-2.csv')

# Display the first few rows of the dataset to understand its structure
df.head()

# Define the function to calculate toll rates
def calculate_toll_rate(df):
    # Rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Add columns to calculate toll rates for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate
    
    return df

# Apply the function to the dataset
df_with_toll_rates = calculate_toll_rate(df)

# Display the updated DataFrame with toll rates
print("Display the updated DataFrame with toll rates", df_with_toll_rates.head())





# Question 13: Calculate Time-Based Toll Rates



# Create a list of days in a week for easy reference
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Create a function to generate a 24-hour period split into time intervals
def generate_time_intervals():
    # We will generate start and end times as datetime.time objects
    intervals = [
        (time(0, 0, 0), time(10, 0, 0)),  # 00:00:00 to 10:00:00
        (time(10, 0, 0), time(18, 0, 0)), # 10:00:00 to 18:00:00
        (time(18, 0, 0), time(23, 59, 59)) # 18:00:00 to 23:59:59
    ]
    return intervals

# Define the function to calculate time-based toll rates
def calculate_time_based_toll_rates(df):
    # Create columns for start_day, start_time, end_day, and end_time
    results = []

    # Define discount factors for weekdays and weekends
    weekday_discounts = [0.8, 1.2, 0.8]  # For each time interval on weekdays
    weekend_discount = 0.7  # Constant discount factor for weekends
    
    # Loop through each row of the dataframe
    for _, row in df.iterrows():
        distance = row['distance']
        
        # Loop through all 7 days of the week
        for i, day in enumerate(days_of_week):
            # Determine if it's a weekday or weekend
            if day in days_of_week[:5]:  # Monday to Friday
                discounts = weekday_discounts
            else:  # Saturday and Sunday
                discounts = [weekend_discount] * 3  # Apply constant discount factor
                
            # Loop through each time interval in a 24-hour day
            for (start, end), discount in zip(generate_time_intervals(), discounts):
                # Create new rows for each day/time interval
                result_row = {
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'start_day': day,
                    'start_time': start,
                    'end_day': day,
                    'end_time': end
                }
                
                # Calculate discounted toll rates for each vehicle type
                result_row['moto'] = distance * 0.8 * discount
                result_row['car'] = distance * 1.2 * discount
                result_row['rv'] = distance * 1.5 * discount
                result_row['bus'] = distance * 2.2 * discount
                result_row['truck'] = distance * 3.6 * discount
                
                # Append the row to the results list
                results.append(result_row)
    
    # Convert the results list to a DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

# Apply the function to calculate time-based toll rates
df_time_based_tolls = calculate_time_based_toll_rates(df_with_toll_rates)

# Display the first few rows of the resulting DataFrame
print("Display the first few rows of the resulting DataFrame", df_time_based_tolls.head())





