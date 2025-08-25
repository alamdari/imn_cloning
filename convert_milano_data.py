import pandas as pd
import numpy as np

def convert_milano_data(input_file, output_file):
    """
    Convert Milano2007_ORIGINAL.csv to the required format for IMN generation.
    
    The input file has space-separated values with columns:
    Vehicle_ID, Date, Time, Latitude_x10^6, Longitude_x10^6, Speed, Direction, Quality, Engine_status, Distance
    """
    
    print("Reading Milano2007_ORIGINAL.csv...")
    
    # Read the space-separated file
    # The file has no headers and is space-separated
    columns = [
        'Vehicle_ID', 'Date', 'Time', 'Latitude_x10^6', 'Longitude_x10^6',
        'Speed', 'Direction', 'Quality', 'Engine_status', 'Distance'
    ]
    
    # Read with whitespace separator and no header
    df = pd.read_csv(input_file, sep=r'\s+', names=columns, header=None)
    
    print(f"Loaded {len(df)} rows")
    print("Sample data:")
    print(df.head())
    
    # Convert coordinates from x10^6 format to decimal degrees
    df['latitude'] = df['Latitude_x10^6'] / 1000000.0
    df['longitude'] = df['Longitude_x10^6'] / 1000000.0
    
    # Combine date and time into timestamp and convert to Unix timestamp
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['timestamp_unix'] = df['timestamp'].astype(np.int64) // 10**9  # Convert to Unix timestamp (seconds)
    
    # Rename Vehicle_ID to id
    df['id'] = df['Vehicle_ID']
    
    # Select only the required columns
    result_df = df[['id', 'longitude', 'latitude', 'timestamp_unix']].copy()
    result_df = result_df.rename(columns={'timestamp_unix': 'timestamp'})
    
    # Sort by id and timestamp
    result_df = result_df.sort_values(['id', 'timestamp'])
    
    print(f"\nFinal data shape: {result_df.shape}")
    print("Final columns:", result_df.columns.tolist())
    print("\nSample of converted data:")
    print(result_df.head(10))
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    result_df.to_csv(output_file, index=False)
    print("Conversion completed successfully!")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total records: {len(result_df)}")
    print(f"Unique users: {result_df['id'].nunique()}")
    print(f"Date range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
    print(f"Latitude range: {result_df['latitude'].min():.6f} to {result_df['latitude'].max():.6f}")
    print(f"Longitude range: {result_df['longitude'].min():.6f} to {result_df['longitude'].max():.6f}")

if __name__ == "__main__":
    input_file = "data/Milano2007_ORIGINAL.csv"
    output_file = "data/Milano2007_ORIGINAL_combined.csv"
    
    convert_milano_data(input_file, output_file)


