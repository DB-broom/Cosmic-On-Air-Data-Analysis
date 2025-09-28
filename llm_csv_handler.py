import pandas as pd
import json
import re
from google import genai
import os
from functools import lru_cache
from dotenv import load_dotenv
import hashlib

load_dotenv()

def create_prompt(file_path, num_lines=10):
    """
    Read first N lines of file and create LLM prompt.
    
    Args:
        file_path (str): Path to CSV file
        num_lines (int): Number of lines to read
        
    Returns:
        str: Prompt for LLM
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [f.readline().strip() for _ in range(num_lines)]
    
    raw_data = '\n'.join(line for line in lines if line)
    
    prompt = f"""Analyze this CSV data and return a JSON configuration:

    {raw_data}

    Return JSON with this structure:
    {{
    "file_info": {{
        "skip_rows": 0,
        "delimiter": ";",
        "encoding": "utf-8"
    }},
    "columns": {{
        "timestamp": {{"column_name": "Time", "column_index": 0}}, #This is the datetime column. Not the number timeStamp
        "latitude": {{"column_name": "Latitude", "column_index": 11}},
        "longitude": {{"column_name": "Longitude", "column_index": 12}},
        "altitude": {{"column_name": "altitude_col", "column_index": X}},
        "CPM": {{"column_name": "Count Rate, cps", "column_index": 3, "conversion": "multiply_by_60"}},
        "CPS": {{"column_name": "Count Rate, cps", "column_index": 3}}
        "dose_rate": {{"column_name": "Dose Rate", "column_index": 4}}
    }}
    }}

    Requirements:
    - Use exact column names from header
    - Only include fields that exist (skip missing ones)
    - Set conversion: "multiply_by_60" for CPS->CPM
    - Ignore empty columns"""

    
    return prompt

def get_cache_path(file_path):
    """
    Generate a cache file path based on the input file path.
    
    Args:
        file_path (str): Path to the original file
        
    Returns:
        str: Path to the cache file
    """
    # Create cache directory if it doesn't exist
    cache_dir = "gemini_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a hash of the file path for the cache filename
    file_hash = hashlib.md5(file_path.encode()).hexdigest()
    cache_filename = f"{file_hash}.txt"
    return os.path.join(cache_dir, cache_filename)

def gemini_api_call(prompt, file_path=None):
    """
    Call Gemini API with caching support.
    
    Args:
        prompt (str): The prompt to send to Gemini
        file_path (str, optional): Path to the original file for caching
        
    Returns:
        str: Gemini response
    """
    # If file_path is provided, check cache first
    if file_path:
        cache_path = get_cache_path(file_path)
        if os.path.exists(cache_path):
            print(f"Loading cached response from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
    print("Making Gemini API call to find formatting of CSV")
    # Make API call if not cached
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    gemini_client = genai.Client(api_key=gemini_api_key)
    client = gemini_client
    
    # Retry logic for transient errors
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            if response.text is None:
                raise ValueError("Gemini response is None.")
            
            # Save response to cache if file_path is provided
            if file_path:
                cache_path = get_cache_path(file_path)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                print(f"Cached response saved to {cache_path}")
            
            return response.text
        except Exception as e:
            if attempt == 2:
                raise


def process_json_response(llm_response):
    """
    Extract and validate JSON from LLM response.
    
    Args:
        llm_response (str): Raw LLM response
        
    Returns:
        dict: Parsed JSON config
    """
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON found in response")
    
    return json.loads(json_match.group())

def read_csv_with_config(file_path, config):
    """
    Read CSV file using the JSON configuration.
    
    Args:
        file_path (str): Path to CSV file
        config (dict): JSON configuration
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Read CSV
    
    df = pd.read_csv(
        file_path,
        skiprows=(config['file_info'].get('skip_rows', 0)),
        delimiter=config['file_info'].get('delimiter', ','),
        encoding=config['file_info'].get('encoding', 'utf-8')
    )
    # Extract required columns
    result = pd.DataFrame()
    
    for field, col_config in config['columns'].items():
        col_name = col_config['column_name']
        index = col_config["column_index"]

        data = df.iloc[:,index].copy()

        # Apply conversions
        if col_config.get('conversion') == 'multiply_by_60':
            data = data * 60
        elif col_config.get('conversion') == 'feet_to_meters':
            data = data * 0.3048
        
        result[field] = data

    return result

# Complete workflow function
def process_csv_file(file_path):
    """
    Complete workflow: prompt -> LLM -> JSON -> CSV processing
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        str: Prompt to send to LLM
    """
    prompt = create_prompt(file_path)

    response = gemini_api_call(prompt, file_path)

    configured = process_json_response(response)
    csv = read_csv_with_config(file_path,configured)
    return csv
    

# Example usage
if __name__ == "__main__":

    # Generate prompt
    csv = process_csv_file('Cape Town to Amsterdam (KLM598 - 12_06_2024)\Raw Data - 32890612.log')
    print(csv)
    
