from math import nan
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

# --- Parsers (No changes needed, they are correct) ---

def convert_ddmm_to_decimal(ddmm_str: str, direction: str) -> float:
    try:
        ddmm_float = float(ddmm_str)
        degrees = int(ddmm_float / 100)
        minutes = ddmm_float - (degrees * 100)
        decimal_degrees = degrees + (minutes / 60)
        if direction.upper() in ['S', 'W']:
            decimal_degrees *= -1
        return decimal_degrees
    except (ValueError, TypeError):
        return None

def parse_bnrdd_log(file_path: str) -> pd.DataFrame:
    column_names = [
        "Identifier", "Device_ID", "Timestamp_str", "CPM", "CPS", "Total_Counts",
        "Counts_Value", "Latitude_ddmm", "Latitude_Dir", "Longitude_ddmm",
        "Longitude_Dir", "Altitude_m", "GPS_Value", "GPS_HDOP", "Checksum"
    ]
    data_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('$BNRDD,'):
                clean_line = line.strip().split('*')[0][1:]
                parts = clean_line.split(',')
                if len(parts) == len(column_names):
                    data_rows.append(parts)
    if not data_rows: return pd.DataFrame()
    df = pd.DataFrame(data_rows, columns=column_names)
    df['timestamp'] = pd.to_datetime(df['Timestamp_str'], utc=True, errors='coerce')
    df['latitude'] = df.apply(lambda row: convert_ddmm_to_decimal(row['Latitude_ddmm'], row['Latitude_Dir']), axis=1)
    df['longitude'] = df.apply(lambda row: convert_ddmm_to_decimal(row['Longitude_ddmm'], row['Longitude_Dir']), axis=1)
    numeric_cols = ['CPM', 'CPS', 'Total_Counts', 'Altitude_m', 'GPS_HDOP']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.rename(columns={'Altitude_m': 'altitude'})
    final_cols = ['timestamp', 'latitude', 'longitude', 'altitude', 'CPM', 'CPS']
    df = df[final_cols].copy()
    df.dropna(inplace=True)
    # Remove rows where the timestamp is 2000-01-01 (likely invalid data)
    df = df[df['timestamp'].dt.date != pd.to_datetime("2000-01-01").date()]
    df.dropna(subset=['timestamp'], inplace=True)
    df["dose_rate"]= df["CPM"]/334
    
    return df

def parse_flight_kml(kml_path: str) -> pd.DataFrame:
    namespaces = {'kml': 'http://www.opengis.net/kml/2.2', 'gx': 'http://www.google.com/kml/ext/2.2'}
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error reading or parsing KML file {kml_path}: {e}")
        return pd.DataFrame()
    track_element = root.find('.//gx:Track', namespaces)
    if track_element is None: return pd.DataFrame()
    timestamps = [elem.text for elem in track_element.findall('kml:when', namespaces)]
    coords_str = [elem.text for elem in track_element.findall('gx:coord', namespaces)]
    flight_data = []
    for i in range(min(len(timestamps), len(coords_str))):
        lon, lat, alt = map(float, coords_str[i].split())
        flight_data.append({
            'timestamp': timestamps[i],
            'longitude': lon,
            'latitude': lat,
            'altitude': alt
        })
    df = pd.DataFrame(flight_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.sort_values('timestamp', inplace=True)
    return df

def parse_Flight_radar_csv(csv_filepath):
    """
    Parse a CSV file with flight data and convert it to a DataFrame with UTC, Latitude, Longitude, and Altitude columns.
    
    Expected CSV format:
    Timestamp,UTC,Callsign,Position,Altitude,Speed,Direction
    1752351184,2025-07-12T20:13:04Z,DLH576,"50.048195,8.565812",0,1,337
    
    Args:
        csv_filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns: UTC, Latitude, Longitude, Altitude
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Original columns: {list(df.columns)}")
        
        # Check if required columns exist
        required_columns = ['UTC', 'Position', 'Altitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Create a new DataFrame with the desired structure
        result_df = pd.DataFrame()
        
        # Convert UTC column to datetime
        result_df['timestamp'] = pd.to_datetime(df['UTC'], utc=True)
        
        # Extract latitude and longitude from Position column
        # Position format: "50.048195,8.565812"
        def extract_coordinates(position_str):
            try:
                # Remove quotes and split by comma
                clean_pos = position_str.strip('"')
                lat_str, lon_str = clean_pos.split(',')
                return float(lat_str), float(lon_str)
            except (ValueError, AttributeError) as e:
                print(f"Warning: Could not parse position '{position_str}': {e}")
                return None, None
        
        # Apply coordinate extraction
        coordinates = df['Position'].apply(extract_coordinates)
        latitudes = [coord[0] for coord in coordinates if coord[0] is not None]
        longitudes = [coord[1] for coord in coordinates if coord[1] is not None]
        
        # Check if we have valid coordinates
        if len(latitudes) != len(df):
            print(f"Warning: Only {len(latitudes)} out of {len(df)} rows have valid coordinates")
        
        result_df['latitude'] = latitudes
        result_df['longitude'] = longitudes
        
        # Add altitude column
        result_df['altitude'] = pd.to_numeric(df['Altitude'], errors='coerce')/3.28084  # Convert feet to meters
        
        # Remove rows with missing coordinates or altitude
        original_count = len(result_df)
        result_df = result_df.dropna(subset=['latitude', 'longitude', 'altitude'])
        final_count = len(result_df)
        
        if final_count < original_count:
            print(f"Removed {original_count - final_count} rows with missing data")
        
        # Sort by UTC timestamp
        #result_df.sort_values('timestamp').reset_index(drop=True)

        return result_df
        
    except FileNotFoundError:
        print(f"Error: File '{csv_filepath}' not found")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: File '{csv_filepath}' is empty")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return pd.DataFrame()

def parse_radiacode(file_path):
    df = pd.read_csv(file_path, delimiter=";")

    # Some files may have inconsistent column names (extra spaces, hidden chars), so clean them
    df.columns = df.columns.str.strip()

    # Select only the desired columns
    columns_to_keep = [
        "Time",
        "Count Rate, cps",
        "Dose rate, Sv/h",
        "Latitude",
        "Longitude",
        "Altitude" if "Altitude" in df.columns else None  # keep altitude if it exists
    ]
    columns_to_keep = [col for col in columns_to_keep if col is not None]

    df = df[columns_to_keep]
    # Rename columns to standard names for downstream processing
    rename_map = {
        "Time": "timestamp",
        "Count Rate, cps": "CPS",
        "Dose rate, Sv/h": "dose_rate",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Altitude": "altitude"
    }
    # Only keep renames for columns that exist in the dataframe
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    df["timestamp"] = pd.to_datetime(df['timestamp'], utc=True)
    df["CPM"] = df["CPS"]*60
    df["dose_rate"] = df["dose_rate"]*1000000
    # Drop None in case "Altitude" isn't present


    # Show or export
    return df

def parse_gmc500(file_path):
    """
    Parse a GMC Data Viewer CSV file and return a DataFrame with timestamp, CPM, and dose_rate (uSv/h).
    """
    import pandas as pd
    import csv

    # Find the header row and get all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Date Time"):
            header_idx = i
            break
    
    if header_idx is None:
        print("Error: Could not find header row in GMC500 CSV.")
        return pd.DataFrame()

    # Manually parse the data to avoid pandas parsing issues
    data = []
    header_line = lines[header_idx].strip()
    
    # Split header and find the columns we want (first 4 columns)
    header_parts = header_line.split(',')
    column_names = [part.strip() for part in header_parts[:4]]

    
    # Process data rows
    for line in lines[header_idx + 1:]:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Split the line and take only the first 4 columns
        parts = line.split(',')
        if len(parts) >= 4:
            row_data = [part.strip() for part in parts[:4]]
            data.append(row_data)
    
    if not data:
        print("Error: No data rows found.")
        return pd.DataFrame()
    
    # Create DataFrame with only the columns we want
    df = pd.DataFrame(data, columns=column_names)


    # Keep only relevant columns
    columns_to_keep = ["Date Time", "uSv/h", "CPM"]
    df = df[columns_to_keep]

    # Rename columns for consistency
    df = df.rename(columns={
        "Date Time": "timestamp",
        "uSv/h": "dose_rate",
        "CPM": "CPM"
    })

    # Parse timestamp and convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce', utc=True)
    df["dose_rate"] = pd.to_numeric(df["dose_rate"], errors='coerce')
    df["CPM"] = pd.to_numeric(df["CPM"], errors='coerce')

    # Drop rows with missing values
    df = df.dropna(subset=["timestamp", "dose_rate", "CPM"])

    # Optionally, add CPS column (CPM/60)
    df["CPS"] = df["CPM"] / 60.0

    return df

def processCSV(filepath):
    from llm_csv_handler import process_csv_file

    df = process_csv_file(filepath)
    first_common = find_most_common_by_sampling(df, "timestamp", Date=True)
    df = filter_by_flight_date(df, first_common)
    return df

def opensky():
    from opensky_api import OpenSkyApi
    from datetime import datetime, timezone,timedelta
    api = OpenSkyApi()
    import requests

    r = requests.get("https://opensky-network.org/api/states/all")
    print(r.status_code)
    #print(r.json() if r.ok else r.text)

    # --- PART 4: Main execution with plotting ---
    import logging
    #logging.basicConfig(level=logging.DEBUG)
    result = api.get_states()
    print(len(result.states), result.states[:5])

    icao24 = "e8027d"  # e.g. Lufthansa A320
    now = datetime.now(timezone.utc)
    begin = int((now - timedelta(days=1)).timestamp())
    end   = int(now.timestamp())

    # 2. Retrieve recent flights
    flights = api.get_flights_by_aircraft(icao24, begin, end)
    if not flights:
        print("No flights found for the past 24 hours.")
    else:
        flight = flights[0]
        print("Flight detected:", flight.icao24, "from", flight.estDepartureAirport, "at", datetime.fromtimestamp(flight.firstSeen, tz=timezone.utc))
        
        # 3. Pick a timestamp 10 minutes into the flight
        timestamp = flight.firstSeen + 600

        # 4. Retrieve the track for that timestamp
        track = api.get_track_by_aircraft(icao24, t=timestamp)
        print("Track from", datetime.fromtimestamp(track.startTime, tz=timezone.utc), "to", datetime.fromtimestamp(track.endTime, tz=timezone.utc))
        for wp in track.path[:10]:  # print first 10 waypoints
            print(wp.time, wp.latitude, wp.longitude, wp.baro_altitude)

# --- Flight Analysis Functions ---

def identify_takeoff_landing(df):
    """Identify takeoff and landing points to filter out ground operations"""
    if df.empty or 'altitude_interp' not in df.columns:
        return df
    
    # Define thresholds
    MIN_FLIGHT_ALTITUDE = 1000  # meters - altitude must exceed this to be considered "in flight"
    MIN_FLIGHT_DURATION = 300   # seconds - flight phase must last at least 5 minutes
    
    # Sort by timestamp to ensure proper order
    # df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Find points above minimum flight altitude
    in_flight = df['altitude_interp'] > MIN_FLIGHT_ALTITUDE
    
    if not in_flight.any():
        print("Warning: No flight data found above minimum altitude threshold")
        return df
    
    # Find the first and last sustained flight points
    flight_indices = df[in_flight].index
    
    # Find takeoff: first point where we stay above threshold for MIN_FLIGHT_DURATION
    takeoff_idx = None
    for idx in flight_indices:
        # Check if we stay above threshold for the next 5 minutes
        future_time = df.loc[idx, 'timestamp'] + pd.Timedelta(seconds=MIN_FLIGHT_DURATION)
        future_data = df[(df['timestamp'] >= df.loc[idx, 'timestamp']) & 
                        (df['timestamp'] <= future_time)]
        
        if len(future_data) > 0 and (future_data['altitude_interp'] > MIN_FLIGHT_ALTITUDE).all():
            takeoff_idx = idx
            break
    
    # Find landing: last point where we were above threshold for MIN_FLIGHT_DURATION before
    landing_idx = None
    for idx in reversed(flight_indices):
        # Check if we were above threshold for the previous 5 minutes
        past_time = df.loc[idx, 'timestamp'] - pd.Timedelta(seconds=MIN_FLIGHT_DURATION)
        past_data = df[(df['timestamp'] >= past_time) & 
                      (df['timestamp'] <= df.loc[idx, 'timestamp'])]
        
        if len(past_data) > 0 and (past_data['altitude_interp'] > MIN_FLIGHT_ALTITUDE).all():
            landing_idx = idx
            break
    
    if takeoff_idx is None or landing_idx is None:
        print("Warning: Could not identify clear takeoff/landing points")
        return df
    
    # Filter data to flight period only
    flight_data = df.loc[takeoff_idx-100:landing_idx+100].copy()
    
    takeoff_time = df.loc[takeoff_idx, 'timestamp']
    landing_time = df.loc[landing_idx, 'timestamp']
    flight_duration = (landing_time - takeoff_time).total_seconds() / 3600  # hours
    
    # print(f"Flight Analysis:")
    print(f"   Takeoff: {takeoff_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"   Landing: {landing_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    # print(f"   Flight Duration: {flight_duration:.2f} hours")
    # print(f"   Data points: {len(df)} → {len(flight_data)} (removed {len(df) - len(flight_data)} ground points)")
    
    return flight_data.reset_index(drop=True)

def round_coordinates(df, decimal_places=4):
    """Round latitude and longitude to specified decimal places"""
    coord_columns = ['latitude', 'longitude', 'latitude_interp', 'longitude_interp']
    
    for col in coord_columns:
        if col in df.columns:
            df[col] = df[col].round(decimal_places)
    
    print(f"Coordinates rounded to {decimal_places} decimal places")
    return df

def filter_by_flight_date(df, flight_date):
    """
    Filter dataframe to keep only entries with timestamps on the specified flight date.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'timestamp' column
        flight_date (str): Flight date in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Filtered dataframe with only entries from the specified date
    """
    if df.empty or 'timestamp' not in df.columns:
        print("Warning: DataFrame is empty or missing 'timestamp' column")
        return df
    
    # Convert flight_date string to datetime.date object
    try:
        target_date = pd.to_datetime(flight_date,utc=True).date()
    except ValueError as e:
        print(f"Error: Invalid date format '{flight_date}'. Use 'YYYY-MM-DD' format.")
        return df
    
    # Ensure timestamp column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        except Exception as e:
            print(f"Error converting timestamp column to datetime: {e}")
            return df

    # Filter to keep only rows where the timestamp date matches the flight date or the day after
    original_count = len(df)
    prev_day = target_date - pd.Timedelta(days=1)
    next_day = target_date + pd.Timedelta(days=1)
    df_filtered = df[df['timestamp'].dt.date.isin([target_date, next_day,prev_day])].copy()
    filtered_count = len(df_filtered)
    # Print summary of filtering
    removed_count = original_count - filtered_count
    # print(f"Date Filtering Results:")
    # print(f"   Target date: {flight_date}")
    # print(f"   Original entries: {original_count}")
    # print(f"   Entries after filtering: {filtered_count}")
    print(f"Filtering and Removed entries: {removed_count}")
    
    if removed_count > 0:
        # Show some examples of removed timestamps
        removed_timestamps = df[df['timestamp'].dt.date != target_date]['timestamp'].dt.date.unique()
        print(f"   Removed dates: {sorted(removed_timestamps)}")
    
    return df_filtered

def find_most_common_by_sampling(df, column_name, sample_size=400, random_state=None, Date= False):
    """
    Find the most common value in a DataFrame column by sampling a subset of rows.
    
    Args:
        df (pd.DataFrame): DataFrame to sample from
        column_name (str): Name of the column to analyze
        sample_size (int): Number of rows to sample (default: 100)
        random_state (int, optional): Random seed for reproducible sampling
    
    Returns:
        tuple: (most_common_value, count, total_sampled) or (None, 0, 0) if error
    """
    if df.empty or column_name not in df.columns:
        print(f"Warning: DataFrame is empty or missing column '{column_name}'")
        return None, 0, 0


    actual_sample_size = min(sample_size, len(df))
    
    if actual_sample_size < len(df):
        # Sample the DataFrame
        df_sample = df.sample(n=actual_sample_size, random_state=random_state)
        print(f"Sampling {actual_sample_size} rows from {len(df)} total rows")
    else:
        # Use entire DataFrame if sample size >= total rows
        df_sample = df
        print(f"Using entire DataFrame ({len(df)} rows)")
    # If the column is datetime, use only the date part for value_counts
    col = df_sample[column_name]
    
    # If the column is datetime-like or string that looks like datetime, convert to date for value_counts
    if Date:
        # If Date is True, convert the column to date only (no time component)
        col = col.apply(lambda d: d.date() if hasattr(d, 'date') else str(d).split("T")[0].split(" ")[0] if isinstance(d, str) else d)
        # col = col.apply(lambda d: d.date() if hasattr(d, 'date') else str(d).split("T")[0] if isinstance(d, str) and "T" in d else d)
    # Determine actual sample size (don't sample more than available)

    
    # Count values in the sampled data
    value_counts = col.value_counts()
    
    if value_counts.empty:
        print(f"No data found in column '{column_name}'")
        return None, 0, 0
    
    # Get the most common value and its count
    most_common_value = value_counts.index[0]
    most_common_count = value_counts.iloc[0]
    
    # Calculate percentage of the most common value in the sample
    percentage = (most_common_count / actual_sample_size) * 100
    
    print(f"Most common value in '{column_name}': {most_common_value}")
    print(f"Count in sample: {most_common_count}/{actual_sample_size} ({percentage:.1f}%)")
    
    # Show top 3 values if available
    # if len(value_counts) > 1:
    #     print(f"   Top 3 values:")
    #     for i, (value, count) in enumerate(value_counts.head(3).items()):
    #         pct = (count / actual_sample_size) * 100
    #         print(f"     {i+1}. {value}: {count} ({pct:.1f}%)")
    
    return most_common_value

def interpolate(logdf, radardf, compareCols = ['latitude','longitude','altitude']):
    
    x_new = logdf['timestamp'].astype('int64') / 10**9
    x_known = radardf['timestamp'].astype('int64') / 10**9

    # Interpolate each coordinate
    df_final = logdf.copy()

    for column in compareCols:
        interp = np.interp(x_new, x_known, radardf[column])
        df_final[f'{column}_interp'] = interp

    return df_final

def extract_working_gps(flight_data, measured_data, compareCols=['altitude', 'latitude', 'longitude'], step_range= None):
    
    if step_range ==None:
        delta =  flight_data["timestamp"].iloc[0].tz_localize(None) - measured_data["timestamp"].iloc[0].tz_localize(None) 
        hour_diff = int(delta.total_seconds()/3600)
        delta =  flight_data["timestamp"].iloc[-1].tz_localize(None) - measured_data["timestamp"].iloc[-1].tz_localize(None) 
        hour_end = int(delta.total_seconds()/3600)
        if hour_end>hour_diff:
            step_range = range(hour_diff,hour_end+1)
        else:
            step_range = range(hour_end,hour_diff+1)

        print(step_range)
    steps = step_range
    best = float('inf')
    best_step = None
    columns = []
    for col in compareCols:
        if col in measured_data.columns:
            columns.append(col)

    if columns==[]:
        return None
    for step in steps:
        # Shift measured_data timestamps by 'step' hours
        shifted = measured_data.copy()
        shifted['timestamp'] = pd.to_datetime(shifted['timestamp'],utc=True) + pd.Timedelta(hours=step)

        inter = interpolate(shifted, flight_data, compareCols)
        
        # Filter out data points outside the flight data time range
        flight_start = flight_data['timestamp'].min()
        flight_end = flight_data['timestamp'].max()
  
        inter = inter[(inter['timestamp'] >= flight_start) & (inter['timestamp'] <= flight_end)]
        
        if len(inter) ==0:
            continue
        score = 0

        for _, n in inter.iterrows():
            add =0
            for col in columns:
                add += abs(n[col] - n[f"{col}_interp"])
            add = add/len(columns)
            score+=add
        score = score / (len(inter)**2)
        if score < best:
            best = score
            best_step = step
    if best_step ==None:
        print("error in ",", ".join(columns),"returning none")
        return None
    # Display the timezone offset relative to UTC
    if best_step == 0:
        tz_str = "UTC"
    else:
        sign = "+" if best_step > 0 else "-"
        tz_str = f"UTC{sign}{abs(best_step):02d}:00"

    print(f"Best score: {best:.4f} at offset {best_step} hours")
    print(f"Measured data timezone appears to be: {tz_str} (relative to UTC flight data)")

    return best_step

def find_takeoff_point(df, column, threshold, min_duration=3600, timestamp_col='timestamp', order=1):
    """
    Find the takeoff point where a column exceeds a threshold for a sustained period.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Flight data
    column : str
        Column name to monitor for takeoff (e.g., 'altitude', 'speed', 'climb_rate')
    threshold : float
        Value that must be exceeded to indicate takeoff
    min_duration : int, default=600
        Minimum duration (seconds) to stay above threshold
    timestamp_col : str, default='timestamp'
        Name of the timestamp column
    
    Returns:
    --------
    pandas.Timestamp or None
        Timestamp of the takeoff point, or None if no takeoff found
    """
    if df.empty or column not in df.columns or timestamp_col not in df.columns:
        print(f"Warning: DataFrame empty or missing columns '{column}' or '{timestamp_col}'")
        return None
    
    # Sort by timestamp
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    
    # State variables for tracking sustained period
    candidate_start_idx = None
    candidate_start_time = None
    
    # Loop through each row once
    if order == 1:
        iterator = df.iterrows()
    elif order == -1:
        iterator = reversed(list(df.iterrows()))
    else:
        raise ValueError("order must be 1 (forward) or -1 (backward)")

    for idx, row in iterator:
        current_value = row[column]
        current_time = row[timestamp_col]
        
        if current_value > threshold:
            # If we're not currently tracking a candidate, start tracking
            if candidate_start_idx is None:
                candidate_start_idx = idx
                candidate_start_time = current_time
            
            # Check if we've sustained above threshold for min_duration
            elapsed_time = ((current_time - candidate_start_time)*order).total_seconds()
            if elapsed_time >= min_duration:
                print(f"Takeoff detected at {candidate_start_time}: {column}={df.loc[candidate_start_idx, column]:.1f} > {threshold}")
                return candidate_start_time
        
        else:
            # Value dropped below threshold - reset if we were tracking
            
            
            if candidate_start_idx is not None:

                candidate_start_idx = None
                candidate_start_time = None
    
    # If we get here, no sustained takeoff was found
    if candidate_start_idx is not None:
        elapsed_time = (df.iloc[-1][timestamp_col] - candidate_start_time).total_seconds()
        print(f"Warning: Found candidate takeoff at {candidate_start_time} but only sustained for {elapsed_time:.1f}s < {min_duration}s")
    else:
        print(f"Warning: No sustained takeoff found for {column} > {threshold}")
    
    return None

def process_flight_file(flight_dict):
    """Parse and filter the flight data file (KML/CSV)."""
    if flight_dict.get("Flight_aware"):
        flight_df = parse_flight_kml(flight_dict.get("Flight_aware"))
        print("successfully parsed flightaware file")
    elif flight_dict.get("Flight_radar"):
        flight_df = parse_Flight_radar_csv(flight_dict.get("Flight_radar"))
        print("successfully parsed flight radar file")
    else:
        out = input("UNKNOWN flight data file, please input the file name")
        flight_df = processCSV(out)
    flight_df["timestamp"] = pd.to_datetime(flight_df["timestamp"],utc=True)
    return flight_df

def process_device_file(flight_key, flight_path):
    """Parse and filter the main device data file."""
    mainType = ""
    main_df = None
    if "bnrdd_log" ==flight_key :
        mainType = "bnrdd_log"
        print("processing", mainType)
        main_df = parse_bnrdd_log(flight_path)
    elif "Safecastzen" == flight_key :
        mainType = "Safecastzen"
        print("processing", mainType)
        main_df = parse_bnrdd_log(flight_path)
    elif "Radiacode" ==flight_key :
        mainType = "Radiacode"
        print("processing", mainType)
        main_df = parse_radiacode(flight_path)
    elif "log_file_GMC500" ==flight_key :
        mainType = "log_file_GMC500"
        print("processing", mainType)
        main_df = parse_gmc500(flight_path)

    else:
        print("processing with llm", flight_key)
        main_df = processCSV(flight_path)
        mainType = flight_key

    first_common = find_most_common_by_sampling(main_df, "timestamp", Date=True)
    main_df = filter_by_flight_date(main_df, first_common)
    return main_df

def correct_timezone(main_df,flight_df, cari7_df=None):

    dose_off =find_takeoff_point(main_df,"dose_rate", threshold=main_df['dose_rate'].mean())
    take_off= find_takeoff_point(flight_df,"altitude",threshold=flight_df['altitude'].mean())
    if dose_off is not None and take_off is not None:
        delta =  take_off -dose_off
        step = round(delta.total_seconds()/3600)
        print("Takeoff-based timezone correction of", step, "hours")
        return main_df, step
    #Trys to find landing point instead of takeoff
    dose_off =find_takeoff_point(main_df,"dose_rate", threshold=main_df['dose_rate'].mean(), order=-1)
    take_off= find_takeoff_point(flight_df,"altitude",threshold=flight_df['altitude'].mean(),order=-1)
    if dose_off is not None and take_off is not None:
        delta =  take_off -dose_off
        step = round(delta.total_seconds()/3600)
        print("Landing-based timezone correction of", step, "hours")
        return main_df, step

    step = int(input("Could not automatically determine timezone offset. Please enter the timezone offset in hours (e.g., -5 for UTC-5): "))
    print("Applying timezone correction of", step, "hours")
    main_df['timestamp'] = pd.to_datetime(main_df['timestamp'],utc=True) + pd.Timedelta(hours=step)
    
    return main_df, step

import requests

# def get_elevations_batch(lat_lon_list, dataset="srtm30m"):
#     """
#     Fetch elevations for multiple coordinates in a single API request.
#     lat_lon_list: list of (lat, lon) tuples
#     Returns: list of elevations in meters
#     """
#     if not lat_lon_list:
#         return []

#     base_url = "https://api.opentopodata.org/v1"
#     url = f"{base_url}/{dataset}"
    
#     # Build locations string for the API
#     locations_str = "|".join([f"{lat},{lon}" for lat, lon in lat_lon_list])
#     params = {"locations": locations_str}
    
#     try:
#         resp = requests.get(url, params=params, timeout=30)
#         resp.raise_for_status()
#         data = resp.json()
#         elevations = [res.get("elevation") for res in data.get("results", [])]
#         return elevations
#     except Exception as e:
#         print("Error fetching batch elevations:", e)
#         return [None] * len(lat_lon_list)

# --- Plotting Functions ---

def plot_3d_interactive(df_final, cari7=None, title="Flight Path Visualization", xyzc=['longitude','latitude','altitude','dose_rate']):
    """Create interactive 3D visualizations using Plotly with improved layout and logic"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import json
    import urllib.request
    
    # Validate inputs
    if df_final is None or len(df_final) == 0:
        print("Warning: Empty or None dataframe provided")
        return
    
    # 3D Flight path with radiation data
    fig1 = go.Figure()
    
    # Create interpolated column names for flight data
    interpkeys = []
    for item in xyzc:
        if item in ['longitude', 'latitude', 'altitude']:
            interpkeys.append(f'{item}_interp')
        else:
            interpkeys.append(item)
    
    # Check if required columns exist
    missing_cols = [col for col in interpkeys if col not in df_final.columns]
    if missing_cols:
        print(f"Warning: Missing columns in df_final: {missing_cols}")
        return
    dose_rate_data = df_final[interpkeys[3]]
    # Add FlightAware interpolated track (currently disabled in original code)
    if False:  # Enable flight track visualization
        # Calculate dose rate difference if cari7 is available
        dose_rate_data = df_final[interpkeys[3]]
        if cari7 is not None and xyzc[3] in cari7.columns:
            # Interpolate cari7 data to match df_final timestamps if needed
            dose_rate_diff = dose_rate_data - cari7[xyzc[3]].iloc[0]  # Simplified for now
        else:
            dose_rate_diff = dose_rate_data
        
        fig1.add_trace(go.Scatter3d(
            x=df_final[interpkeys[0]],
            y=df_final[interpkeys[1]], 
            z=df_final[interpkeys[2]],
            mode='markers+lines',
            marker=dict(
                size=3,
                color=dose_rate_diff,
                colorscale='Viridis',
                colorbar=dict(
                    title=f'{xyzc[3].replace("_", " ").title()}',
                    x=1.02,  # Move colorbar away from plot area
                    len=0.7,  # Make colorbar shorter to avoid overlap
                    thickness=15
                ),
                showscale=True,
                cmin=np.nanpercentile(dose_rate_diff, 5),  # Robust color scaling
                cmax=np.nanpercentile(dose_rate_diff, 95)
            ),
            line=dict(
                width=2,
                color=dose_rate_diff,
                colorscale='Viridis'
            ),
            name='Flight Track',
            hovertemplate='<b>Flight Data</b><br>' +
                        'Longitude: %{x:.4f}°<br>' +
                        'Latitude: %{y:.4f}°<br>' +
                        'Altitude: %{z:.0f}m<br>' +
                        f'{xyzc[3].replace("_", " ").title()}: %{{marker.color:.2f}}<br>' +
                        '<extra></extra>'
        ))

    # Add CARI7 reference track if available
    result = pd.merge(cari7, df_final, on='timestamp', how='inner')
    if cari7 is not None:
        # Validate cari7 has required columns
        missing_cari7_cols = [col for col in xyzc if col not in cari7.columns]
        if not missing_cari7_cols:
            cpm_diff = df_final[interpkeys[3]]
            fig1.add_trace(go.Scatter3d(
                x=df_final[interpkeys[0]],
                y=df_final[interpkeys[1]],
                z=df_final[interpkeys[2]],  # Convert to meters if needed
                mode='markers+lines',
                marker=dict(
                    size=3,
                    color=cpm_diff,
                    colorscale='Plasma',  # Different colorscale for distinction
                    colorbar=dict(
                        title='Dose Rate Safecast (uSv/h)',
                        x=1.15,  # Position further right
                        y= 0.8,
                        len=0.5,
                        thickness=15
                    ),
                    showscale=True,
                    cmin=np.nanmin(cpm_diff),
                    cmax=max(np.nanmax(cpm_diff), 3)
                ),
                line=dict(
                    width=2,
                    color=cpm_diff,
                    colorscale='Plasma'
                ),
                name='CARI7 Reference',
                hovertemplate='<b>CARI7 Track</b><br>' +
                              'Longitude: %{x:.4f}°<br>' +
                              'Latitude: %{y:.4f}°<br>' +
                              'Altitude: %{z:.0f}m<br>' +
                              'Dose rate Diff: %{marker.color:.3f}<br>'
                              '<extra></extra>'
            ))
        else:
            print(f"Warning: Missing columns in cari7: {missing_cari7_cols}")

    # Add Earth surface and coastlines with improved error handling
    add_earth_surface(fig1)
    
    # Calculate appropriate axis ranges
    max_altitude = float(np.nanmax(df_final[interpkeys[2]])) if len(df_final) > 0 else 10000.0
    lon_range = [np.nanmin(df_final[interpkeys[0]]), np.nanmax(df_final[interpkeys[0]])]
    lat_range = [np.nanmin(df_final[interpkeys[1]]), np.nanmax(df_final[interpkeys[1]])]
    min_val = min(lon_range[0], lat_range[0])
    max_val = max(lon_range[1], lat_range[1])
    # Expand ranges slightly for better visualization
    padding = (max_val- min_val) * 0.2
    
    fig1.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        scene=dict(
            xaxis=dict(
                title='Longitude (°)',
                range=[min_val - padding, max_val + padding]
            ),
            yaxis=dict(
                title='Latitude (°)', 
                range=[max(min_val - padding,-90), min(max_val+ padding, 90)]
            ),
            zaxis=dict(
                title='Altitude (m)',
                range=[0, max_altitude * 1.1]
            ),
            camera=dict(
                eye=dict(x=1.2, y=-1.5, z=0.8),  # Better default view showing map correctly
                center=dict(x=0, y=0, z=0.2),
                up=dict(x=0, y=0, z=1)  # Ensure Z is up
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)  # Better proportions
        ),
        width=1200,
        height=800,
        margin=dict(r=150),  # Extra margin for colorbars
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )
    )
    
    fig1.show()
    fig1.write_html('flight_path_3d.html')
    
    # Create multi-panel dashboard with improved logic
    # create_dashboard(df_final, cari7, interpkeys, xyzc)
    
    print("Interactive plots saved as:")
    print("- flight_path_3d.html")  
    print("- flight_dashboard.html")

def plot_dosage(plot_df, plotColumns, cari7_col=None, flightheading="Flight", Calibrated=False, foldername="", extrainfo=""):
    """
    Plot dosage data with interactive legend and multiple y-axes support.
    
    Parameters:
    - plot_df: DataFrame containing the data to plot
    - plotColumns: List of column names to plot
    - cari7_col: Reference column for calibration (optional)
    - flightheading: Title identifier for the plot
    - Calibrated: Boolean indicating whether to apply calibration
    - foldername: Folder name for saving (currently unused)
    - extrainfo: Additional information to display on plot
    """
    
    # Ensure we have a copy to avoid modifying the original DataFrame
    plot_df = plot_df.copy()
    
    # Handle timestamp conversion
    try:
        if 'timestamp' not in plot_df.columns:
            print("Warning: 'timestamp' column not found in DataFrame")
            return
        
        if not pd.api.types.is_datetime64_any_dtype(plot_df['timestamp']):
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'], utc=True, errors='coerce')
    except Exception as e:
        print(f"Error parsing 'timestamp' column to datetime: {e}")
        print("Plotting with raw 'timestamp' values.")

    # Remove rows with invalid timestamps
    plot_df = plot_df.dropna(subset=['timestamp'])
    
    if plot_df.empty:
        print("No valid data to plot after dropping missing timestamps.")
        return

    # Create figure and primary axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define more distinguishable colors with better contrast
    colors = ['#1f77b4', '#2ca02c', '#9e9229', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Store plot objects for toggling
    plot_objects = {}
    legend_to_plot = {}  # Map legend labels to plot objects
    annotations = []
    color_index = 0
    secondary_axis = None
    
    for col in plotColumns:
        if col not in plot_df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue
            
        print(f"Plotting column: {col}")
        color = colors[color_index % len(colors)]  # Cycle through colors if more columns than colors
        
        if "dose_rate" in col.lower():
            # Handle dose rate columns on primary axis
            
            # Apply calibration if requested and conditions are met
            if cari7_col is not None and col != cari7_col and Calibrated and cari7_col in plot_df.columns:
                try:
                    shift = get_calibration(plot_df, cari7_col, col)
                    
                    # Add calibration annotation
                    calibrated_annotation = ax.annotate(
                        f"{col} calibration: {shift:.2f}", 
                        xy=(0.7, 0.01 + len(annotations) * 0.03), 
                        xycoords='axes fraction',
                        fontsize=10, color='black', ha='right', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                    )
                    annotations.append(calibrated_annotation)
                    
                    # Plot calibrated data
                    calibrated_data = plot_df[col] * shift
                    calibrated_label = f"{col} calibrated"
                    calibrated_extra = ax.scatter(
                        plot_df['timestamp'], calibrated_data, 
                        marker='o', alpha=0.6, s=2, color=color, 
                        label=calibrated_label, edgecolors='none'
                    )
                    plot_objects[calibrated_label] = calibrated_extra
                    legend_to_plot[calibrated_label] = calibrated_extra
                    
                except NameError:
                    print(f"Warning: get_calibration function not defined. Skipping calibration for {col}")
                except Exception as e:
                    print(f"Error applying calibration to {col}: {e}")
            
            # Plot original data
            label = f"{col} calibrated" if Calibrated and cari7_col is not None and col != cari7_col else col
            extra_scatter = ax.scatter(
                plot_df['timestamp'], plot_df[col], 
                marker='o', alpha=0.6, s=2, color=color, 
                label=label, edgecolors='none'
            )
            plot_objects[label] = extra_scatter
            legend_to_plot[label] = extra_scatter
            
        else:
            # Handle non-dose columns (e.g., altitude) on secondary y-axis
            if secondary_axis is None:
                secondary_axis = ax.twinx()
                secondary_axis.spines["right"].set_position(("outward", 0))
                secondary_axis.set_ylabel("Altitude (m)", fontsize=12, color=color)
                secondary_axis.tick_params(axis='y', labelcolor=color)
            
            # Filter out NaN values for line plotting to create continuous lines
            valid_mask = plot_df[col].notna()
            valid_data = plot_df[valid_mask]
            
            if valid_data.empty:
                print(f"Warning: No valid data found for column '{col}'. Skipping.")
                continue
            
            # Plot other non-dose columns
            plot_line, = secondary_axis.plot(
                valid_data['timestamp'], valid_data[col], 
                alpha=0.6, color=color, label=f"{col} (Altitude axis)", linewidth=1.5,
                marker='o', markersize=2, markerfacecolor=color, markeredgewidth=0
            )
            plot_objects[col] = plot_line
            legend_to_plot[col] = plot_line
        
        color_index += 1
    
    # Configure primary axis
    ax.set_xlabel('Time (M-D-H)', fontsize=12)
    ax.set_ylabel(r'Dose rate ($\mu$Sv/h)', fontsize=12)
    ax.set_title(f'Dose Rate vs Time for {flightheading}', fontsize=14, fontweight='bold')
    
    # Format x-axis
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Create combined legend for both axes
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = [], []
    if secondary_axis is not None:
        lines_2, labels_2 = secondary_axis.get_legend_handles_labels()
    
    all_lines = lines_1 + lines_2
    all_labels = labels_1 + labels_2
    
    if all_lines:  # Only create legend if there are items to show
        # Position legend outside the plot area to avoid interference
        legend = ax.legend(
            all_lines, all_labels,
            bbox_to_anchor=(0.5, 0.08),  # Move legend higher above the x-axis
            loc='lower center',
            frameon=True, fancybox=True, shadow=True, fontsize=10, markerscale=3
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('gray')
        
        # Make legend markers more visible
        for handle in legend.legend_handles:
            if hasattr(handle, 'set_alpha'):
                handle.set_alpha(1.0)
            if hasattr(handle, 'set_sizes'):
                handle.set_sizes([50])
        
        # Interactive legend toggling functionality
        def on_pick(event):
            # Handle clicks on legend text
            if hasattr(event.artist, 'get_text'):
                label = event.artist.get_text()
            # Handle clicks on legend handles
            elif hasattr(event.artist, '_label'):
                label = event.artist._label
            else:
                return
            
            print(f"Clicked on: {label}")  # Debug print
            
            if label in legend_to_plot:
                plot_obj = legend_to_plot[label]
                current_visibility = plot_obj.get_visible()
                
                # Toggle visibility
                plot_obj.set_visible(not current_visibility)
                
                # Find the corresponding legend handle and update its appearance
                for i, legend_label in enumerate(all_labels):
                    if legend_label == label:
                        legend_handle = legend.legend_handles[i]
                        if current_visibility:  # Was visible, now hidden
                            legend_handle.set_alpha(0.3)
                        else:  # Was hidden, now visible
                            legend_handle.set_alpha(1.0)
                        break
                
                # Redraw the plot
                fig.canvas.draw()
        
        # Make legend items clickable
        for legend_text in legend.get_texts():
            legend_text.set_picker(True)
        
        for legend_handle in legend.legend_handles:
            legend_handle.set_picker(True)
            # Only set pickradius for handles that support it
            if hasattr(legend_handle, 'set_pickradius'):
                legend_handle.set_pickradius(10)
        
        # Connect the click event
        fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Apply layout and styling
    plt.tight_layout()
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add instruction text in upper left corner
    ax.text(0.02, 0.98, 'Click legend items to toggle visibility', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
    
    # Add extra info if provided
    if extrainfo != "":
        print("Extrainfo:", extrainfo)
        ax.text(0.02, 0.02, extrainfo, 
                transform=ax.transAxes, fontsize=10, 
                horizontalalignment='left', verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
    
    # Optional: Save figure (uncomment if needed)
    # if foldername:
    #     plt.savefig(f'{foldername}_dose_rate_vs_time.png', dpi=300, bbox_inches='tight')
    #     print(f"Plot saved as '{foldername}_dose_rate_vs_time.png'")
    
    plt.show()

def add_earth_surface(fig):
    """Add Earth surface and coastlines to 3D plot"""
    z0 = 0  # Sea level
    
    try:
        # Add semi-transparent ocean surface
        fig.add_trace(go.Surface(
            z=[[z0, z0], [z0, z0]],
            x=[-180, 180],
            y=[-90, 90],
            surfacecolor=[[0, 0], [0, 0]],
            colorscale=[[0, '#4da6dc'], [1, '#4da6dc']],  # Ocean blue
            showscale=False,
            opacity=0.3,
            name='Ocean Surface',
            hoverinfo='skip'
        ))

        # Download and add coastlines
        coastline_url = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson'
        import urllib.request
        with urllib.request.urlopen(coastline_url, timeout=10) as response:
            coastline_geojson = json.loads(response.read().decode('utf-8'))

        xs, ys, zs = [], [], []
        for feature in coastline_geojson.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'LineString':
                coords = geom.get('coordinates', [])
                for lon, lat in coords:
                    xs.append(lon)
                    ys.append(lat)
                    zs.append(z0 + 10)  # Slightly above surface
                # Add None to separate line segments
                xs.append(None)
                ys.append(None)
                zs.append(None)
            elif geom.get('type') == 'MultiLineString':
                for line in geom.get('coordinates', []):
                    for lon, lat in line:
                        xs.append(lon)
                        ys.append(lat)
                        zs.append(z0 + 10)
                    # Add None to separate line segments
                    xs.append(None)
                    ys.append(None)
                    zs.append(None)

        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(color='#2d5016', width=2),  # Dark green for coastlines
                name='Coastlines',
                showlegend=False,
                hoverinfo='skip'
            ))
            
    except Exception as e:
        print(f"Warning: Could not load coastlines ({e}). Using fallback grid.")
        # Fallback: simple lat/lon grid
        for lon in range(-180, 181, 60):  # Less dense grid
            fig.add_trace(go.Scatter3d(
                x=[lon, lon], y=[-90, 90], z=[z0, z0],
                mode='lines', line=dict(color='lightgray', width=1),
                showlegend=False, hoverinfo='skip'
            ))
        for lat in range(-60, 61, 30):  # Skip polar regions
            fig.add_trace(go.Scatter3d(
                x=[-180, 180], y=[lat, lat], z=[z0, z0],
                mode='lines', line=dict(color='lightgray', width=1),
                showlegend=False, hoverinfo='skip'
            ))

def create_kml_flight_tour(df,foldername):
    import pandas as pd
    from simplekml import Kml
    import numpy as np
    import os
    
    # Downsample for performance
    df = df.iloc[::5, :].reset_index(drop=True)
    
    # Calculate heading between consecutive points
    def calculate_heading(lat1, lon1, lat2, lon2):
        """Calculate bearing/heading from point 1 to point 2 in degrees"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        bearing = np.arctan2(y, x)
        return (np.degrees(bearing) + 360) % 360
    
    # Add heading column
    headings = []
    for i in range(len(df)):
        if i < len(df) - 1:
            heading = calculate_heading(
                df.iloc[i]["latitude_interp"],
                df.iloc[i]["longitude_interp"],
                df.iloc[i+1]["latitude_interp"],
                df.iloc[i+1]["longitude_interp"]
            )
            headings.append(heading)
        else:
            # Use the last calculated heading for the final point
            headings.append(headings[-1] if headings else 0)
    
    df["heading"] = headings
    
    # Initialize KML
    kml = Kml()
    
    # Create document
    doc = kml.document
    doc.name = "Flight Tour"
    doc.open = 1
    
    # Create tour
    tour = kml.newgxtour(name="Animated Flight Path")
    playlist = tour.newgxplaylist()
    
    for i, row in df.iterrows():
        lat = row["latitude_interp"]
        lon = row["longitude_interp"]
        alt = row["altitude_interp"]*20  # Scale altitude for better visibility
        heading = row["heading"]
        
        # Create flyto element with LookAt
        flyto = playlist.newgxflyto(gxduration=2.0)
        flyto.gxflytomode = "smooth"
        
        # Use LookAt instead of Camera
        flyto.lookat.longitude = lon
        flyto.lookat.latitude = lat
        flyto.lookat.altitude = alt
        flyto.lookat.heading = heading
        flyto.lookat.tilt = 90
        flyto.lookat.range = 0
        flyto.lookat.altitudemode = "absolute"
    kml.save(f"{foldername}\\flight_tour.kml")
    print(f"Flight tour created with {len(df)} waypoints")
    print("Open the KML file in Google Earth and click the play button to start the tour")

def flight_path(df, foldername):
    import pandas as pd
    import numpy as np

    # Adaptive sampling: more points for higher CPS values
    def adaptive_sampling(df, base_interval=5, max_interval=20):
        """
        Create adaptive sampling based on CPS values.
        Higher CPS = more points (smaller interval)
        Lower CPS = fewer points (larger interval)
        """
        # Normalize CPS to 0-1 range
        cps_min, cps_max = df['CPS'].min(), df['CPS'].max()
        cps_norm = (df['CPS'] - cps_min) / (cps_max - cps_min + 1e-9)
        
        # Calculate sampling intervals (inverse relationship with CPS)
        # High CPS (1.0) -> base_interval, Low CPS (0.0) -> max_interval
        intervals = base_interval + (max_interval - base_interval) * (1 - cps_norm)
        
        # Create sampling mask
        sampled_indices = []
        current_idx = 0
        
        while current_idx < len(df):
            sampled_indices.append(current_idx)
            # Move to next index based on current interval
            interval = int(intervals.iloc[current_idx])
            current_idx += max(1, interval)  # Ensure we always move forward
        
        return df.iloc[sampled_indices].reset_index(drop=True)

    # Apply adaptive sampling
    df_sampled = adaptive_sampling(df)
    print(f"Adaptive sampling: {len(df)} → {len(df_sampled)} points")

    # Normalize CPS for color mapping
    cps_min, cps_max = df_sampled['CPS'].min(), df_sampled['CPS'].max()
    df_sampled['cps_norm'] = (df_sampled['CPS'] - cps_min) / (cps_max - cps_min + 1e-9)

    # Color mapping function (blue -> green -> yellow -> red)
    def cps_to_color(value):
        """Convert normalized CPS (0-1) to KML color (AABBGGRR format)"""
        if value < 0.33:  # Low CPS: Blue to Green
            r = int(255 * (value / 0.33))
            g = int(255 * (value / 0.33))
            b = 255
        elif value < 0.66:  # Medium CPS: Green to Yellow
            r = 255
            g = 255
            b = int(255 * (1 - (value - 0.33) / 0.33))
        else:  # High CPS: Yellow to Red
            r = 255
            g = int(255 * (1 - (value - 0.66) / 0.34))
            b = 0
        
        # KML colors are AABBGGRR (alpha, blue, green, red)
        return f"ff{b:02x}{g:02x}{r:02x}"

    # Build KML with colored points
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2"
     xmlns:gx="http://www.google.com/kml/ext/2.2">
    <Document>
    <name>Flight Path - CPS Colored Points</name>
    <open>1</open>
    <description>Flight path with points colored by Counts Per Second (CPS)</description>
    
    <!-- Color scale reference -->
    <ScreenOverlay>
        <name>Color Scale</name>
        <overlayXY x="0" y="1" xunits="fraction" yunits="fraction"/>
        <screenXY x="0" y="1" xunits="fraction" yunits="fraction"/>
        <size x="200" y="100" xunits="pixels" yunits="pixels"/>
        <color>80ffffff</color>
    </ScreenOverlay>
    """

    # Add colored points
    for i, row in df_sampled.iterrows():
        lat = row["latitude_interp"]
        lon = row["longitude_interp"]
        alt = row["altitude_interp"]*20 
        cps = row["CPS"]
        cpm = row["CPM"]
        timestamp = row["timestamp"]
        
        # Parse timestamp for display
        dt = pd.to_datetime(timestamp,utc=True )
        
        # Get color based on CPS
        color = cps_to_color(row["cps_norm"])
        
        # Calculate point size based on CPS (higher CPS = larger points)
        point_size = 0.5 + (row["cps_norm"] * 1.5)  # 0.5 to 2.0 scale
        
        kml_content += f"""
    <Placemark>
        <description>
            <![CDATA[
            <b>Time:</b> {dt.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>
            <b>CPS:</b> {cps:.2f}<br/>
            <b>CPM:</b> {cpm:.1f}<br/>
            <b>Altitude:</b> {alt:.0f} m<br/>
            <b>Position:</b> {lat:.4f}, {lon:.4f}
            ]]>
        </description>
        <Point>
            <coordinates>{lon},{lat},{alt}</coordinates>
            <altitudeMode>absolute</altitudeMode>
        </Point>
        <Style>
            <IconStyle>
                <color>{color}</color>
                <scale>{point_size:.2f}</scale>
                <Icon>
                    <href>http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href>
                </Icon>
            </IconStyle>
            <LabelStyle>
                <scale>0.8</scale>
            </LabelStyle>
        </Style>
    </Placemark>
    """
    # Close KML
    kml_content += """
    </Document>
    </kml>
    """

    # Save to file
    with open(f"{foldername}\\flight_cps.kml", "w", encoding="utf-8") as f:
        f.write(kml_content)
    
    print(f"Created flight_path_cps_colored.kml with {len(df_sampled)} colored points")
    print(f"CPS range: {cps_min:.2f} - {cps_max:.2f}")
    print(f"Color mapping: Blue (low CPS) → Green → Yellow → Red (high CPS)")
    print(f"Point sizes: 0.5 - 2.0 (scaled by CPS)")
    
    return df_sampled

#--- Cari-7 fucntions ---

def make_default_inp(loc_filename, date_str, output_dir):
    contents = f"""0000/00/00
    0 
    0 
    4
    {loc_filename}
    """
    with open(os.path.join(output_dir, "DEFAULT.INP"), "w", newline="\r\n") as f:
        f.write(contents)

def convert_to_cari7_format(df, output, particle_type="Total", n=10):
    import subprocess
    def format_timestamp(ts):
        # Check if ts is already a datetime or Timestamp object
        if hasattr(ts, 'strftime'):
            dt = ts
        else:
            # Incoming format: '06-12 23:11:34+00:00'
            dt = datetime.strptime(ts.split('+')[0], "%m-%d %H:%M:%S")
            # Assume year 2024 (adjust if needed)
            # dt = dt.replace(year=2024)
        return dt.strftime("%Y/%m/%d"), dt.strftime("H%H"), dt.strftime("D%d")

    lines = []
    foldername =output
    if particle_type!="Total":
        output+= " "+particle_type
    i = 0
    for idx, row in df.iterrows():
        if idx % n != 0:
            continue
        lat = row['latitude_interp']
        lon = row['longitude_interp']
        alt = row['altitude_interp']
        ts = row['timestamp']

        date_str, hour_str, day_str = format_timestamp(ts)

        # Determine longitude direction and absolute value
        if lon >= 0:
            lon_dir = 'E'
        else:
            lon_dir = 'W'
        if lat >= 0:
            lat_dir = 'N'
        else:
            lat_dir = 'S'
        lon = abs(lon)
        lat = abs(lat)
        # Format line according to given structure
        feetalt = alt/1000
        if particle_type=="Charged Particles":
            for i in range(2,8):
                lines.append(f"{lat_dir}, {lat:.2f}, {lon_dir}, {lon:.2f}, K, {feetalt:.2f}, {date_str}, {hour_str}, D4, P{i}, C4, S0")
        elif particle_type=="Neutrons":
            line = f"{lat_dir}, {lat:.2f}, {lon_dir}, {lon:.2f}, K, {feetalt:.2f}, {date_str}, {hour_str}, D4, P1, C4, S0"
            lines.append(line)
        elif particle_type=="Photons":
            line = f"{lat_dir}, {lat:.2f}, {lon_dir}, {lon:.2f}, K, {feetalt:.2f}, {date_str}, {hour_str}, D4, P2, C4, S0"
            lines.append(line)
        elif particle_type=="Total":
            line = f"{lat_dir}, {lat:.2f}, {lon_dir}, {lon:.2f}, K, {feetalt:.2f}, {date_str}, {hour_str}, D4, P0, C4, S0"
            lines.append(line)
        else:
            print("error unkown type")
        i+=1
        

    work_dir = r"C:\Users\danie\Documents\project\CARI_7_DVD"
    output_txt = f"{work_dir}\\{output[:20]}.LOC"
    with open(output_txt, 'w') as f:
        f.write("START-------------------------------------------------\n")
        f.write("\n".join(lines))
        f.write("\nSTOP-------------------------------------------------")
    print("saved to",output_txt)
    loc_file = output_txt

    import shutil

    cari7_bin = r"C:\Users\danie\Documents\project\CARI_7_DVD\cari-7.exe"
    make_default_inp(f"{output[:20]}.LOC", "2024/06/12", work_dir)
    run_cari7(loc_file, cari7_bin, work_dir)

    # Move loc_file to a new directory named after the loc_file (without extension)
    folder = foldername
    loc_file_basename = os.path.basename(loc_file)
    loc_file_name, _ = os.path.splitext(loc_file_basename)
    target_dir = folder
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(folder, f"{output}.LOC")
    shutil.move(loc_file, target_path)
    shutil.move(loc_file.replace(".LOC",".ANS"), target_path.replace(".LOC",".ANS"))
    print(f"Moved {loc_file} to {target_path}")

def cari7_to_csv(foldername, df_final,m=10):
    cari7_total = pd.read_csv(os.path.join(foldername, f"{foldername}.ANS"), skiprows=1, header=None, names=['LAT', 'LON', 'ALTITUDE', 'ALTITUDE UNITS', 'DATE', 'HR', 'VCR(GV)', 'PARTICLE', 'DOSE RATE', 'UNIT', 'QUANTITY'])
    cari7_neutrons = pd.read_csv(os.path.join(foldername, f"{foldername} Neutrons.ANS"), skiprows=1, header=None, names=['LAT', 'LON', 'ALTITUDE', 'ALTITUDE UNITS', 'DATE', 'HR', 'VCR(GV)', 'PARTICLE', 'DOSE RATE', 'UNIT', 'QUANTITY'])
    cari7_diff = cari7_total.copy()
    cari7_diff['DOSE RATE'] = cari7_total['DOSE RATE'] - cari7_neutrons['DOSE RATE']
    
    # Insert ALTITUDEUNITS column after ALTITUDE column

    cari7_diff['timestamp']= pd.to_datetime(cari7_diff['DATE'], utc=True)
    cari7_df = cari7_diff
    cari7_time = []
    i =0
    for idx, row in df_final.iterrows():
        if idx%m==0:
            cari7_time.append(row["timestamp"])
        

    cari7_df["timestamp"] = cari7_time
    
    # Rename columns to match the required format
    column_mapping = {
        'LAT': 'latitude',
        'LON': 'longitude', 
        'ALTITUDE': 'altitude',
        'ALTITUDEUNIT': 'ALTITUDEUNIT',
        'DATE': 'DATE',
        'HR': 'HR',
        'VCR(GV)': 'VCR(GV)',
        'PARTICLE': 'PARTICLE',
        'DOSE RATE': 'dose_rate',
        'UNIT': 'UNIT',
        'QUANTITY': 'QUANTITY',
        'timestamp': 'timestamp'
    }
    # Apply the column renaming
    cari7_df = cari7_df.rename(columns=column_mapping)
    cari7_diff['timestamp']= pd.to_datetime(cari7_diff['timestamp'], utc=True)
    return cari7_df
    
def cari7_sum_charged_particles(foldername,df_final,m,soft_component_weight=0.5):
    """
    Reads a CARI-7 .ANS file, groups the data by unique location points,
    and sums the dose rates of the specified particle components.

    This correctly models the response of a detector that sees charged particles
    and photons but not neutrons.
    """
    old = "Charged Particles"
    # Define the path to the input file
    input_file = os.path.join(foldername, f"{foldername} Photons.ANS")
    # Read the data, skipping the first header line
    # Note: Column names are slightly simplified for clarity
    try:
        cari7_df = pd.read_csv(
            input_file,
            skiprows=1,
            header=None,
            names=['LAT', 'LON', 'ALTITUDE', 'ALT_UNITS', 'DATE', 'HR', 'VCR_GV', 'PARTICLE', 'DOSE RATE', 'UNIT', 'QUANTITY'],
            # Handle potential whitespace issues in the CSV
            skipinitialspace=True
        )
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return None

    # Convert the DOSE_RATE column to a numeric type, forcing errors to NaN
    cari7_df['DOSE RATE'] = pd.to_numeric(cari7_df['DOSE RATE'], errors='coerce')
    # Drop any rows where the conversion failed
    cari7_df.dropna(subset=['DOSE RATE'], inplace=True)

    # These columns uniquely identify a single measurement point in your flight
    grouping_columns = ['LAT', 'LON', 'ALTITUDE', 'DATE']

    soft_component_weight = 0

    def get_particle_category(particle_str):
        if particle_str in ['POS MUONS ', 'NEG MUONS ']:  # Muons (hard component)
            return 'hard'
        elif particle_str in ['ELECTRONS ','POSITRONS ']: # Photons, Electrons, Positrons (soft)
            return 'soft'
        elif particle_str in ['PHOTONS   ']: # Photons, Electrons, Positrons (soft)
            return 'photons'
        elif particle_str == 'PROTONS   ': # Protons
            return 'proton'
            
        else:
            return 'other'

    cari7_df['category'] = cari7_df['PARTICLE'].apply(get_particle_category)

    # Apply weights
    cari7_df['weighted_dose'] = cari7_df['DOSE RATE']
    # Reduce the contribution of the soft component
    cari7_df.loc[cari7_df['category'] == 'soft', 'weighted_dose'] *= 0
    cari7_df.loc[cari7_df['category'] == 'hard', 'weighted_dose'] *= 0
    cari7_df.loc[cari7_df['category'] == 'proton', 'weighted_dose'] *= 0
    cari7_df.loc[cari7_df['category'] == 'photons', 'weighted_dose'] *= 1
    # You could also add a weight for protons if desired
    # Group by the location columns and sum the DOSE_RATE for each group.
    # .agg() allows us to perform multiple operations. We sum the dose rate
    # and take the 'first' value for the other columns to keep them as metadata.
    summed_df = cari7_df.groupby(grouping_columns).agg(
        # The new summed dose rate column
        DOSE_RATE_SUM=('weighted_dose', 'sum'),
        # Keep the first value from these columns for context
        ALT_UNITS=('ALT_UNITS', 'first'),
        HR=('HR', 'first'),
        VCR_GV=('VCR_GV', 'first'),
        UNIT=('UNIT', 'first')
    ).reset_index()


    # Add a descriptive quantity name
    summed_df['QUANTITY'] = 'Summed non-neutron H*(10)'
    
    # Rename column for clarity
    summed_df.rename(columns={'DOSE_RATE_SUM': 'dose_rate'}, inplace=True)


    cari7_time = []
    for idx, row in df_final.iterrows():
        if idx%m==0:
          cari7_time.append(row["timestamp"])
    summed_df["timestamp"] = cari7_time
    return summed_df

def run_cari7(loc_filepath, cari7_executable, working_dir, ini_file=None):
    """
    Runs CARI-7 on a .LOC file.

    - loc_filepath: path to the .LOC input file
    - cari7_executable: path to the CARI-7 binary (e.g. 'cari7.exe' or './cari7')
    - working_dir: directory where run files (INI, INP) are located
    - ini_file: optional, alternate CARI.INI path
    """
    import subprocess
    cwd = working_dir or os.getcwd()
    cmd = [cari7_executable]
    if ini_file:
        # If supported, pass custom INI via command-line, or copy into cwd
        cmd += ['--ini', ini_file]  # adjust if CARI-7 accepts it

    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        print(f"CARI-7 completed for {loc_filepath}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running CARI-7: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

#--- Folder Anaylisis   ---

def classify_flight_folder(folder_path: str):
    """
    Takes a folder path, lists its files, and uses Gemini to classify them into a flight dictionary entry.
    """
    from google import genai
    folder_name = os.path.basename(folder_path)
    files = os.listdir(folder_path)

    # Create prompt for Gemini
    # prompt = f"""
    # You are given a folder containing flight-related files. 
    # The folder name is: "{folder_name}"
    # The files in it are:
    # {files}

    # Classify them into a structured dictionary with possible keys:

    # Flight_data:
    #     - Flight_radar
    #     - Flight_aware
    # Device_data:
    #     - Radiacode
    #     - Safecastzen
    #     - GMC500

    # If there are multiple files of the same type, list them all in a list.
    # If a type is not present, omit that key.

    # Return ONLY a valid JSON object with the folder name as the key.
    # Example format:
    #     "Paris Sanfran (AF82 24-06-2025)": {
    #     "Detectors": {"Safecastzen": ['12250624.log', '2025-06-24_1517.log', '23400624.LOG', '23430624.LOG', '32200624.log']},
    #     "Flight_data": {"type":"Flight_aware", 
    #                     "file":"FlightAware_AFR82_LFPG_KSFO_20250624.kml"}
    # }
    # """
   
    prompt2= f"""
        **Improved Prompt:**

        You are given a folder that contains flight-related files.

        * **Folder name:** `{folder_name}`
        * **Files inside:**

        ```
        {files}
        ```

        Your task is to classify these files into a structured **JSON object**.

        ### Classification Rules:

        * The top-level key must be the folder name.
        * Use the following categories if applicable:

        ```jsonc
        {{
        "Flight_data": {{
            "Flight_radar": [/* list of matching files */],
            "Flight_aware": [/* list of matching files */]
        }},
        "Device_data": {{
            "Radiacode": [/* list of matching files */],
            "Safecastzen": [/* list of matching files */],
            "GMC500": [/* list of matching files */]
        }}
        }}
        ```

        ### Requirements:

        * If multiple files belong to the same type, list them all in an array.
        * If a type or category has no files, omit that key entirely.
        * Ensure the result is **valid JSON**, no extra text or commentary.

        ### Example Output:

        ```json
        "Paris Sanfran (AF82 24-06-2025)": {{
            "Device_data": {{
            "Safecastzen": [ #always .log
                "12250624.log",
                "2025-06-24_1517.log",
                "23400624.LOG",
                "23430624.LOG",
                "32200624.log"
            ]
            }},
            "Flight_data": {{
            "Flight_aware": [ #always .kml
                "FlightAware_AFR82_LFPG_KSFO_20250624.kml"
            ]
            }}
        }}
        ```

        Return **only the JSON object** — no explanations.

"""
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=gemini_api_key)
    
    # Retry logic for transient errors
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt2
            )
            if response.text is None:
                raise ValueError("Gemini response is None.")
            return response.text
        except Exception as e:
            if attempt == 2:
                raise

def get_calibration(df_final, cari7_col, column ="dose_rate"):
    ratio= []
    for idx, row in df_final.iterrows():
        # Find the index in df_final where the timestamp matches cari7_df's timestamp
        matching_idx = df_final.index[(df_final['timestamp'] == row['timestamp']) & (~df_final[cari7_col].isna())]
        if not matching_idx.empty:
            if df_final.loc[matching_idx[0], column] > 0.5 and row[cari7_col] > 0.5:
                ratio.append(row[cari7_col] / df_final.loc[matching_idx[0], column])
    ratio = np.array(ratio)
    ratio = np.mean(ratio)
    return ratio

def process_single_flight(device_path, device_type, flight_path):
    print("Processing single flight with device:", device_path,"\n","*"*10)
    df_final = None
    flight_df = process_flight_file(flight_path)
    device_df = process_device_file(device_type, device_path)
    device_df, step = correct_timezone(device_df, flight_df)
    df_final = interpolate(device_df, flight_df)
    df_final = identify_takeoff_landing(df_final)
    print("Successfully processed:", device_path,"\n","-"*10)
    return df_final

def merge_plots(df_final, df_merge, mergeName, threshold=pd.Timedelta("1min")):

    columns_to_merge = ['timestamp']
    for col in ['dose_rate', 'CPM', 'CPS']:
        if col in df_merge.columns:
            columns_to_merge.append(col)

    df_merge_subset = df_merge[columns_to_merge]
    initial_rename_map = {c: f"{c}_{mergeName}" for c in df_merge_subset.columns if c != 'timestamp'}
    df_merge_ren = df_merge_subset.rename(columns=initial_rename_map)
    
    # Step 2: asof merge to align timestamps within 2 minutes
    merged = pd.merge_asof(
        df_final,
        df_merge_ren,
        on="timestamp",
        direction="nearest",
        tolerance=threshold,
        suffixes=("", f"_{mergeName}")
    )
    keys = initial_rename_map.keys()  # Use initial_rename_map instead of rename_map

    # Remove rows where any of the merged columns (except timestamp) are all NaN (i.e., unmatched)
    merged_cols = [col for col in df_merge_ren.columns if col != "timestamp"]
    # Keep only rows where at least one merged column is not NaN
    matched = merged[merged[merged_cols].notna().any(axis=1)].reset_index(drop=True)
    print("match",matched)
    interp = interpolate(matched, df_merge, keys)
    keys_interp = [f"{c}_interp" for c in keys]
    interp_to_final_rename_map = {ci: c for c, ci in zip(keys_interp, keys)}  # Use different variable name
    # Select all columns except those in 'keys'
    df_merge_subset = interp[[col for col in interp.columns if col not in keys]]
    df_merged = df_merge_subset.rename(columns=interp_to_final_rename_map)

    print("df_merged",df_merged)
    # Alternative approach: Use indicator to find truly unmatched rows
    # This finds rows from df_merge that have no corresponding row in df_final within tolerance

    df_final_with_marker = df_final[['timestamp']].copy()
    df_final_with_marker['_marker'] = 1

    merge_with_indicator = pd.merge_asof(
        df_merge_ren,
        df_final_with_marker, 
        on="timestamp",
        direction="nearest",
        tolerance=threshold,
        suffixes=("", "_final")
    )
    
    # Rows where timestamp_final is NaN are truly unmatched
    unmatched_mask = merge_with_indicator['_marker'].isna()
    truly_unmatched = df_merge_ren[unmatched_mask]
    if not truly_unmatched.empty:
        print("ajdsf;lkasdjf")
        # Create rows for unmatched data with NaN for df_final columns
        unmatched_rows = truly_unmatched.copy()
        
        # Add NaN columns for all df_final columns (except timestamp)
        df_final_cols = [col for col in df_final.columns if col != 'timestamp']
        for col in df_final_cols:
            unmatched_rows[col] = pd.NA
            
        # Add interpolated versions with NaN as well
        for key in keys:
            interp_col = f"{key}_interp" 
            if interp_col in df_merged.columns:
                final_col = interp_to_final_rename_map.get(interp_col, interp_col)  # Use the new variable name
                unmatched_rows[final_col] = pd.NA
        
        # Reorder columns to match df_merged - but handle duplicates first
        unique_columns = list(dict.fromkeys(df_merged.columns))  # Remove duplicates while preserving order
        unmatched_rows = unmatched_rows.reindex(columns=unique_columns, fill_value=pd.NA)
        
        # Concatenate matched and unmatched rows
        df_final_result = pd.concat([df_merged, unmatched_rows], ignore_index=True)
        
        # Sort by timestamp to maintain chronological order
        df_final_result = df_final_result.sort_values('timestamp').reset_index(drop=True)
    else:
        df_final_result = df_merged

    return df_final_result

import requests

# Helper to split list into chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Fetch elevations for a batch of coordinates
def get_elevations_batch(lat_lon_list, dataset="srtm30m"):
    if not lat_lon_list:
        return []
    base_url = "https://api.opentopodata.org/v1"
    url = f"{base_url}/{dataset}"
    locations_str = "|".join([f"{lat},{lon}" for lat, lon in lat_lon_list])
    params = {"locations": locations_str}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        elevations = [res.get("elevation") for res in data.get("results", [])]
        return elevations
    except Exception as e:
        print("Error fetching batch elevations:", e)
        return [None] * len(lat_lon_list)

# Main function to fill elevations in the DataFrame
def get_floor_elevation(df_final, sample_every=10, batch_size=50):
    """
    Adds 'elevation' column to df_final using Open Topo Data API.
    
    Args:
        df_final (pd.DataFrame): Must contain 'latitude_interp' and 'longitude_interp'
        sample_every (int): Query API every Nth row (default 10)
        batch_size (int): Max number of points per API call (default 50)
    """
    import time
    df_final = df_final.copy()
    df_final["elevation"] = np.nan

    if "latitude_interp" not in df_final.columns or "longitude_interp" not in df_final.columns:
        raise ValueError("DataFrame must contain 'latitude_interp' and 'longitude_interp' columns")

    # Select rows to query (every Nth row)
    query_rows = df_final.iloc[::sample_every]
    lat_lon_list = list(zip(query_rows["latitude_interp"], query_rows["longitude_interp"]))

    # Fetch elevations in batches
    elevations = []
    for batch in chunks(lat_lon_list, batch_size):
        batch_elevs = get_elevations_batch(batch)
        
        # flatten and ensure each value is a number or 0 if None or nan
        batch_elevs = [float(e) if e is not None and not np.isnan(e) else 0 for e in batch_elevs]
        elevations.extend(batch_elevs)
        time.sleep(1)

    # Assign fetched elevations back to the DataFrame
    print(query_rows.index)
    df_final.loc[query_rows.index, "elevation"] = elevations
    df_final["elevation"] = df_final["elevation"]

    # Optional: interpolate missing elevations for in-between rows
    # df_final["elevation"] = df_final["elevation"].interpolate(method='linear')

    return df_final


def process_folder(flight_dicts, foldername, cari7=False, m=10,floor= False):

    merged = None
    i=1
    added_cols = []
    fk=""
    flight_path = os.path.join(foldername, flight_dicts["Flight_data"]["file"])
    for key in flight_dicts["Device_data"]:
        if type(flight_dicts["Device_data"][key]) != list:
            filepaths = [flight_dicts["Device_data"][key]]
        else:
            filepaths = flight_dicts["Device_data"][key]
        
        for filepath in filepaths:
            parsed = process_single_flight(f"{foldername}/{filepath}",key,{flight_dicts["Flight_data"]["type"]:flight_path})
            if merged is not None:    
                merged = merge_plots(merged, parsed, f"{key}_{i}")
            else:
                merged = parsed
                fk = key
            added_cols.append(f"{key}_{i}")
            i+=1

    #rename the first row for consitancy
    for col in ["dose_rate","CPS","CPM"]:
        if col in merged.columns:
            merged = merged.rename(columns={col: f"{col}_{fk}_1"})
    
    print("-"*10)
    flight_df = process_flight_file({flight_dicts["Flight_data"]["type"]:flight_path})
    merged = interpolate(merged, flight_df)
    merged = identify_takeoff_landing(merged)

    # Remove rows where altitude_interp is negative
    if "altitude_interp" in merged.columns:
        merged = merged[merged["altitude_interp"] >= 0].reset_index(drop=True)
    print(merged)
    if cari7:
        # convert_to_cari7_format(merged, foldername,particle_type="Neutrons",n = m)
        # convert_to_cari7_format(merged, foldername,particle_type="Total", n = m)
        cari7_df = cari7_to_csv(foldername, merged, m=m)
        merged = merge_plots(merged, cari7_df, "Cari7")
        added_cols.append("Cari7")
    if floor:
        merged = get_floor_elevation(merged)
        print(merged)

    
    return merged, added_cols



if __name__=="__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # print(classify_flight_folder("Busan_2025"))
    
    flight = {
    "SCEL": {
    "Device_data": {
      "Safecastzen": [
        "31241210-new.log"
      ]
    },
    "Flight_data": {"type":"Flight_aware",
        "file":"FlightAware_AFR401_SCEL_LFPG_20211210.kml"}
      
  },
    "Busan_2025 (AF264)": {
    "Device_data": {
      "Safecastzen": [
        "12230322.log",
        "23430322.LOG",
        "32200322.log"
      ]
    },
    "Flight_data": {"type":"Flight_radar",
    "file": "AF264_3993908b.csv"}

  },
    "Cayenne": {
    "Device_data": {
      "Safecastzen": [
        "31221210-new.log"
      ]
    },
    "Flight_data": {"type":"Flight_aware",
        "file":"FlightAware_AFR852_LFPO_SOCA_20211210.kml"}
      
  },
    "Paris Sanfran (AF82 24-06-2025)": {
        "Device_data": {"Safecastzen": ['12250624.log', '2025-06-24_1517.log', '23400624.LOG', '23430624.LOG', '32200624.log']},
        "Flight_data": {"type":"Flight_aware", 
                        "file":"FlightAware_AFR82_LFPG_KSFO_20250624.kml"}
    },
    "Paris Sanfran (AF81 26-06-2025)": {
    "Device_data": {
      "Safecastzen": ["10830627.LOG","12250627.log","2025-06-27_0210.log","23400627.LOG","23430626.LOG","32200627.log"]
    },
    "Flight_data": {
    "type":"Flight_aware",
    "file":"FlightAware_AFR81_KSFO_LFPG_20250627.kml"
    }
    },
    "Hawai_01": {
        "Device_data": {
        "Safecastzen": "31240205-UA9516.log"
        },
        "Flight_data": {"type":"Flight_aware", 
                        "file":"FlightAware_UAL1004_KLAX_PHTO_20220206.kml"}
    
    },
    "Cape Town to Amsterdam (KLM598 - 12_06_2024)": {
        "Device_data": {
            "Safecastzen": ["Raw Data - 32890612.log"]
        },
        "Flight_data": {
            "type": "Flight_aware",
            "file": "FlightAware_KLM598_FACT_EHAM_20240612.kml"
        }
    },
    "Paris to Tokyo (AFR146 - 28_11_2023)": {
        "Device_data": {
            "Safecastzen": ["Raw data - 10831128.LOG"]
        },
        "Flight_data": {
            "type": "Flight_aware",
            "file": "FlightAware_AFR146_LFPG_RJAA_20231128.kml"
        }
    },
    "LH576 FRA CPT 12 July 2025": {
        "Device_data": {
            "Radiacode": ["Radiacode LH576 FRA FRA 12 July 2025.csv"],
            "Safecastzen": ["Safecastzen LH576 FRA CPT 12 July 2025.log"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "LH576 12 July 2025 Flight Radar.csv"
        }
    },
    "LH577 CPT FRA 29 June 2025": {
        "Device_data": {
            "Radiacode": ["Radiacode LH577 CPT FRA 29 June 2025.csv"],
            "Safecastzen": ["Safecastzen LH577 CPT FRA 29 June 2025.log"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "LH577 29 June 2025 Flight Radar.csv"
        }
    },
    "SA222 29 May 2025": {
        "Device_data": {
            "Radiacode": ["Radiacode 29 May 2025 SA222 JBG GRU.csv"],
            "Safecastzen": ["Safecast Zen  29 May 2025 SA222 JNG GRU.log"],
            "log_file_GMC500": ["Cape Town to Joburg 28 May.csv"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "Flight Radar SA222_29 May 2025.csv"
        }
    },
    "SA223 9 June 2025": {
        "Device_data": {
            "Radiacode": ["Radiacode 9 June2025 SA223 GRU JBG.csv"],
            "Safecastzen": ["Safecast Zen  9 June 2025 SA223 GRU JNG.log"],
            "log_file_GMC500": ["GQ-GMC500+_SA223_9June2024.csv"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "Flight Radar SA223_9 June 2025.csv"
        }
    },
    "KL0592 JHB AMS 21 AUG": {
        "Device_data": {
            "Radiacode": ["KL0592 JHB AMS 21 Aug 2025 Radiacode.csv"],
            "Safecastzen": ["KL0592 JHB AMS 21 Aug 2025 Safecast.log"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "KL0592 JHB AMS 21 Aug 2025 Flight Radar.csv"
        }
    },
    "KL0597 AMS CPT 1 SEPT": {
        "Device_data": {
            "Radiacode": ["KL0597 AMS CPT 1 Sept 2025 Radiacode.csv"],
            "Safecastzen": ["KL0597 AMS CPT 1 Sept 2025 Safecast.log"]
        },
        "Flight_data": {
            "type": "Flight_radar",
            "file": "KL0597 AMS CPT 1 Sept 2025 Flight Radar.csv"
        }
    },
    "Frankfort Lax": {
    "Device_data": {
      "Safecastzen": [
        "31240205-new.log"
      ]
    },
    "Flight_data": {
      "Flight_aware": [
        "FlightAware_UAL8845_EDDF_KLAX_20220205.kml"
      ]
    }
  },

    }
    
    merged, added_cols = process_folder(flight["SCEL"], "SCEL",True,floor=True)
    # print(merged)
    merged.to_csv("merged.csv",index=False)
    dose_cols = [f"dose_rate_{name}" for name in added_cols if f"dose_rate_{name}" in merged.columns]
    dose_cols.append("altitude_interp")
    dose_cols.append("elevation")
    # print(dose_cols)
    # cpm_cols = [f"CPM_{name}" for name in added_cols if f"CPM_{name}" in merged.columns]
    # n_devices = len(added_cols)
    # # Scale each row by the average of that row across the selected device columns
    # print(cpm_cols)
    # row_means = merged[cpm_cols].mean(axis=1)
    # rowwise_std = merged[cpm_cols].std(axis=1)

    # # Relative standard deviation (CV)
    # relative_std = rowwise_std / row_means

    # # Now normalize by relative uncertainty
    # normalized_deviations = pd.DataFrame()
    # for col in cpm_cols:
    #     # Avoid division by zero when mean is near zero
    #     mask = np.abs(row_means) > 1e-10
    #     normalized_deviations[col] = np.where(
    #         mask,
    #         (merged[col] - row_means) / row_means,  # Fractional deviation
    #         0
    #     )

    # # Create a histogram DataFrame for dose rates
    # import matplotlib.pyplot as plt

    # # Concatenate all dose rate columns into a single Series
    # all_dose_rates = pd.concat([normalized_deviations[col] for col in cpm_cols], ignore_index=True)
    # # Drop NaN values
    # all_dose_rates = all_dose_rates.dropna()

    # # Create histogram bins and counts
    # counts, bin_edges = np.histogram(all_dose_rates, bins=100)
    # hist_df = pd.DataFrame({
    #     'dose_rate_bin_left': bin_edges[:-1],
    #     'dose_rate_bin_right': bin_edges[1:],
    #     'count': counts
    # })

    # device_precision = pd.DataFrame()
    # fractional_deviations = normalized_deviations
    # for col in cpm_cols:
    #     deviations = fractional_deviations[col].dropna()
    #     device_precision.loc[col, 'mean_fractional_bias'] = deviations.mean()
    #     device_precision.loc[col, 'fractional_precision'] = deviations.std()  # This is now dose-rate independent!
    #     device_precision.loc[col, 'fractional_precision_percent'] = deviations.std() * 100

    # print("Device Precision (dose-rate independent):")
    # print(device_precision)

    #     # Optional: plot the histogram
    # plt.figure(figsize=(8, 4))

    # plt.bar(hist_df['dose_rate_bin_left'], hist_df['count'], 
    #     width=hist_df['dose_rate_bin_right'] - hist_df['dose_rate_bin_left'], align='edge', alpha=0.7)
    # plt.xlabel('CPM (binned)')
    # plt.ylabel('Count')
    # plt.title('Histogram of Counts Per Minute Normalised(all devices)')
    # plt.xlim(-0.4, 0.4)
    # plt.tight_layout()
    # plt.show()

    
    # # dose_cols.append
    # overall_std = rowwise_std.mean()
    # overall_sem = overall_std / np.sqrt(n_devices)
    # anotation = f"Average spread (std): {overall_std:.4f} µSv/h\nUncertainty in mean (SEM): {overall_sem:.4f} µSv/h"
    # print(f"Average spread (std): {overall_std:.4f} µSv/h")
    # print(f"Uncertainty in mean (SEM): {overall_sem:.4f} µSv/h")
    plot_dosage(merged, dose_cols, cari7_col = None ,flightheading="AFR401 Santiago to Paris", Calibrated=False, foldername= "SCEL")
    


if __name__ == "__main_w_":  
    # Define file paths
    # Define a dictionary of flights, each with its own dictionary of file paths
    flights = {
        "Cape Town to Amsterdam (KLM598 - 12_06_2024)": {
            "bnrdd_log": 'Cape Town to Amsterdam (KLM598 - 12_06_2024)/Raw Data - 32890612.log',
            "Flight_aware": 'Cape Town to Amsterdam (KLM598 - 12_06_2024)/FlightAware_KLM598_FACT_EHAM_20240612.kml'
        },
        "Paris to Tokyo (AFR146 - 28_11_2023)": {
            "bnrdd_log": 'Paris to Tokyo (AFR146 - 28_11_2023)/Raw data - 10831128.LOG',
            "Flight_aware": 'Paris to Tokyo (AFR146 - 28_11_2023)/FlightAware_AFR146_LFPG_RJAA_20231128.kml'
        },
        "LH576 FRA CPT 12 July 2025": {
            "Radiacode": "LH576 FRA CPT 12 July 2025/Radiacode LH576 FRA FRA 12 July 2025.csv",
            "Safecastzen": "LH576 FRA CPT 12 July 2025/Safecastzen LH576 FRA CPT 12 July 2025.log",
            "Flight_radar": "LH576 FRA CPT 12 July 2025/LH576 12 July 2025 Flight Radar.csv"
        },
        "LH577 CPT FRA 29 June 2025": {
            "Radiacode": "LH577 CPT FRA 29 June 2025/Radiacode LH577 CPT FRA 29 June 2025.csv",
            "Safecastzen": "LH577 CPT FRA 29 June 2025/Safecastzen LH577 CPT FRA 29 June 2025.log",
            "Flight_radar": "LH577 CPT FRA 29 June 2025/LH577 29 June 2025 Flight Radar.csv"
        },
            "SA222 29 May 2025": {
            "Radiacode": "SA222 29 May 2025/Radiacode 29 May 2025 SA222 JBG GRU.csv",
            "Safecastzen": "SA222 29 May 2025/Safecast Zen  29 May 2025 SA222 JNG GRU.log",
            "Flight_radar": "SA222 29 May 2025/Flight Radar SA222_29 May 2025.csv",
            "log_file_GMC500": "SA222 29 May 2025/Cape Town to Joburg 28 May.csv"
        },
            "SA223 9 June 2025": {
            "Radiacode": "SA223 9 June 2025/Radiacode 9 June2025 SA223 GRU JBG.csv",
            "Safecastzen": "SA223 9 June 2025/Safecast Zen  9 June 2025 SA223 GRU JNG.log",
            "Flight_radar": "SA223 9 June 2025/Flight Radar SA223_9 June 2025.csv",
            "log_file_GMC500": "SA223 9 June 2025/GQ-GMC500+_SA223_9June2024.csv",
        },
            "KL0592 JHB AMS 21 AUG": {
        "Flight_radar": "KL0592 JHB AMS 21 AUG/KL0592 JHB AMS 21 Aug 2025 Flight Radar.csv",
        "Radiacode": "KL0592 JHB AMS 21 AUG/KL0592 JHB AMS 21 Aug 2025 Radiacode.csv",
        "Safecastzen": "KL0592 JHB AMS 21 AUG/KL0592 JHB AMS 21 Aug 2025 Safecast.log"
    },
    "KL0597 AMS CPT 1 SEPT": {
        "Flight_radar": "KL0597 AMS CPT 1 SEPT/KL0597 AMS CPT 1 Sept 2025 Flight Radar.csv",
        "Radiacode": "KL0597 AMS CPT 1 SEPT/KL0597 AMS CPT 1 Sept 2025 Radiacode.csv",
        "Safecastzen": "KL0597 AMS CPT 1 SEPT/KL0597 AMS CPT 1 Sept 2025 Safecast.log"
    },
    }
    flightheadings = [
    "Cape Town to Amsterdam (KLM598 - 12_06_2024)",
    "Paris to Tokyo (AFR146 - 28_11_2023)",
    "Frankfurt to Cape Town (LH576 - 12_07_2025)","Cape Town to Frankfurt (LH577 - 29_06_2025)","Johannesburg to Sao Paulo (SA222 - 29_05_2025)"
    ,"São Paulo to Johannesburg (SA223 - 09_06_2025)","Johannesburg to Amsterdam (KL0592 - 21_08_2025)","Amsterdam to Cape Town (KL0597 - 01_09_2025)"]

    foldernames = ['Cape Town to Amsterdam (KLM598 - 12_06_2024)',"LH576 FRA CPT 12 July 2025","LH577 CPT FRA 29 June 2025","SA223 9 June 2025","SA222 29 May 2025"]
    shifts = []
    foldernames = ["KL0592 JHB AMS 21 AUG"]
    safeshifts = []
    i =-1
    for foldername in flights:
        i+=1
        print("Processing folder:", foldername)
        df_final, extra,cari7_df = process_folder(flights,foldername, cari_convert =False, m=10)
    
        flight_path(df_final,foldername)
        create_kml_flight_tour(df_final,foldername)

            # df_final["Radiacode_dose_rate"] = df_final["Radiacode_dose_rate"]*shift
        
        plot_3d_interactive(df_final, cari7_df,flightheadings[i], ['longitude', 'latitude','altitude', 'dose_rate'])

        input("Press Enter to continue to the next folder...")




    print(shifts)
    print(safeshifts)
    # flight = parse_Flight_radar_csv("LH577 CPT FRA 29 June 2025/LH577 29 June 2025 Flight Radar.csv")
    # plt.figure(figsize=(12, 6))
    # plt.scatter(df_final['timestamp'], df_final['longitude_interp'], marker='o',alpha=0.3,s = 1, color='g', label='alt_iinterp')
    # plt.scatter(cari7_df['timestamp'].sample(n=5000), cari7_df["LON"], marker='o',alpha=0.3,s = 1, color="r", label="alt")

    
    # plt.xlabel('Time')
    # plt.ylabel('Counts Per Minute (CPM)')
    # plt.title('CPM and Additional Data vs Time')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # log_file = flights[foldername]["bnrdd_log"]

    # flight_file = flights[foldername]["Flight_aware"]

    # log_df = processCSV(log_file)


    # flight_df = parse_flight_kml(flight_file)
    # print(flight_df)
    # # flight_df = parse_Flight_radar_csv(flight_file)

    
    # interpolated = interpolate(log_df, flight_df)
    # df_final = identify_takeoff_landing(interpolated)
    # df_final = round_coordinates(df_final, decimal_places=4)
    # df_final.to_csv('final_flight_data_filtered.csv', index=False)

    # step = extract_working_gps(log_df, flight_df)
    # flight_path(df_final,foldername)
    # newfoldername = "(KLM598 - 12_06_2024)".replace(" ","_")
    # #create_kml_flight_tour(df_final,foldername)
    # convert_to_cari7_format(df_final, foldername,neutrons=False)


    #plot_3d_interactive(interpolated)
    # work_dir = r"C:\Users\danie\Documents\project\CARI_7_DVD"
    # cari7_bin = r"C:\Users\danie\Documents\project\CARI_7_DVD\cari-7.exe"
    # make_default_inp(f"CARI7-32890612.LOC", "2024/06/12", work_dir)
    # run_cari7(f"CARI7-32890612.LOC", cari7_bin, work_dir)
