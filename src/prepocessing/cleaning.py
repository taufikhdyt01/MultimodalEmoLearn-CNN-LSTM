import pandas as pd
import os
from datetime import datetime
import shutil

# Function to extract time from timestamp (handles both string and pandas Timestamp)
def extract_time_from_timestamp(timestamp):
    try:
        # Check if it's already a pandas Timestamp
        if isinstance(timestamp, pd.Timestamp):
            dt = timestamp
        else:
            # Try different formats
            try:
                dt = datetime.strptime(timestamp, "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    # If all parsing attempts fail, try to convert to string and parse
                    dt = pd.to_datetime(timestamp)
        
        # Return formatted time string for matching with frame filenames
        return f"{dt.hour:02d}_{dt.minute:02d}_{dt.second:02d}"
    except Exception as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return None

# Identify which problem/challenge a record belongs to
def identify_problem(row, problem_identifier_column='page'):
    """
    Identify which problem/challenge a record belongs to based on the page URL.
    
    Args:
        row: The DataFrame row
        problem_identifier_column: Column name that identifies the problem (default: 'page')
        
    Returns:
        A problem identifier (e.g., "penjumlahan-dua-angka", "status-http")
    """
    if problem_identifier_column in row and pd.notna(row[problem_identifier_column]):
        page_url = str(row[problem_identifier_column])
        
        # Skip if it's just the root tantangan path
        if page_url == '/tantangan':
            return "Other"
        
        # Check if it's a challenge page
        if '/tantangan/' in page_url:
            # Extract the challenge name from the URL
            parts = page_url.split('/')
            
            # Find the index of 'tantangan' in the URL parts
            try:
                tantangan_index = parts.index('tantangan')
                
                # Get the challenge name (next part after 'tantangan')
                if tantangan_index + 1 < len(parts) and parts[tantangan_index + 1]:
                    challenge_name = parts[tantangan_index + 1]
                    
                    # Return the challenge name
                    return challenge_name
            except ValueError:
                # 'tantangan' not found in parts
                pass
            
        # If we get here, it's some other page
        return "Other"
    else:
        # If page column is empty or doesn't exist
        return "Unknown"

# Function to process a single sample with problem separation
def process_sample(sample_num, user_id, df_all, base_dir, output_base_dir):
    # Filter dataframe for this user_id
    df_sample = df_all[df_all['user_id'] == user_id].copy()
    
    if df_sample.empty:
        print(f"No data found for Sample {sample_num} (user_id: {user_id})")
        return pd.DataFrame(), [], {}
    
    # Define source directory for frames
    frames_dir = os.path.join(base_dir, f"Sample {sample_num}", "frames")
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found for Sample {sample_num}: {frames_dir}")
        return pd.DataFrame(), [], {}
    
    # Get all frame files in the directory
    frame_files = {}
    for filename in os.listdir(frames_dir):
        if filename.startswith('frame_') and filename.endswith('.jpg'):
            # Extract timestamp from filename (format: frame_11_54_14.jpg)
            time_parts = filename.replace('frame_', '').replace('.jpg', '')
            frame_files[time_parts] = filename
    
    # Group records by problem
    problem_groups = {}
    for index, row in df_sample.iterrows():
        problem = identify_problem(row)
        
        # Skip records that are just '/tantangan' or 'Other'/'Unknown'
        if problem in ["Other", "Unknown"]:
            continue
            
        if problem not in problem_groups:
            problem_groups[problem] = []
        problem_groups[problem].append(index)
    
    # If no valid problems identified, print warning and return
    if not problem_groups:
        print(f"Warning: No valid problems identified for Sample {sample_num}")
        return pd.DataFrame(), [], {}
    
    # Process each problem group
    all_matched_indices = []
    all_unmatched_frames = []
    problem_stats = {}
    
    for problem, indices in problem_groups.items():
        print(f"\n  Processing problem: {problem}")
        
        # Create output directory for this problem
        problem_output_dir = os.path.join(output_base_dir, f"Sample {sample_num}", problem, "cleaned_frames")
        os.makedirs(problem_output_dir, exist_ok=True)
        
        matched_indices = []
        matched_times = set()
        
        # Process records for this problem
        for index in indices:
            row = df_sample.loc[index]
            time_key = extract_time_from_timestamp(row['timestamp'])
            if not time_key:
                continue
            
            # Check if we have a matching frame
            if time_key in frame_files:
                # Check confidence
                max_confidence = max(
                    float(row['neutral']), float(row['happy']), float(row['sad']), 
                    float(row['angry']), float(row['fearful']), 
                    float(row['disgusted']), float(row['surprised'])
                )
                
                is_valid = (max_confidence >= 0.5 and 
                           (pd.isna(row['Classification']) or row['Classification'] != 'Low Confidence'))
                
                if is_valid:
                    # Keep this record and copy the corresponding frame
                    matched_indices.append(index)
                    matched_times.add(time_key)
                    
                    src_path = os.path.join(frames_dir, frame_files[time_key])
                    dst_path = os.path.join(problem_output_dir, frame_files[time_key])
                    shutil.copy2(src_path, dst_path)
        
        # Add to overall matched indices
        all_matched_indices.extend(matched_indices)
        problem_matched_df = df_sample.loc[matched_indices].copy() if matched_indices else pd.DataFrame()
        
        # Save problem-specific Excel
        if not problem_matched_df.empty:
            problem_excel_dir = os.path.join(output_base_dir, f"Sample {sample_num}", problem)
            problem_excel_path = os.path.join(problem_excel_dir, f"cleaned_data.xlsx")
            problem_matched_df.to_excel(problem_excel_path, index=False)
        
        problem_stats[problem] = {
            'records': len(indices),
            'matched': len(matched_indices)
        }
        
        print(f"    - Records: {len(indices)}")
        print(f"    - Valid matched records: {len(matched_indices)}")
    
    # Find frames that don't match any record across all problems
    all_matched_times = set()
    for idx in all_matched_indices:
        if idx in df_sample.index:
            time_str = extract_time_from_timestamp(df_sample.loc[idx, 'timestamp'])
            if time_str:
                all_matched_times.add(time_str)
    
    for time_key, filename in frame_files.items():
        if time_key not in all_matched_times:
            all_unmatched_frames.append(filename)
    
    # Create combined DataFrame with all matched records
    matched_df = df_sample.loc[all_matched_indices].copy() if all_matched_indices else pd.DataFrame()
    
    # Print summary for this sample
    print(f"\nSample {sample_num} (user_id: {user_id}) Summary:")
    print(f"  - Total records in Excel: {len(df_sample)}")
    print(f"  - Total valid matched records: {len(matched_df)}")
    print(f"  - Total frames without valid records: {len(all_unmatched_frames)}")
    for problem, stats in problem_stats.items():
        print(f"  - {problem}: {stats['matched']}/{stats['records']} records matched")
    
    return matched_df, all_unmatched_frames, problem_stats

# Main function to process all samples
def process_all_samples(excel_path, base_dir, output_base_dir, sample_mapping):
    # Load the Excel file containing all samples
    print(f"Loading data from {excel_path}...")
    try:
        df_all = pd.read_excel(excel_path)
        print(f"Loaded {len(df_all)} records.")
        
        # Convert timestamp column to datetime if it's not already
        if 'timestamp' in df_all.columns and not pd.api.types.is_datetime64_any_dtype(df_all['timestamp']):
            df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
            
        # Print the column names to help identify problem/challenge identifier
        print("\nAvailable columns:", df_all.columns.tolist())
            
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each sample
    all_matched_dfs = []
    all_stats = {}
    
    for sample_num, user_id in sample_mapping.items():
        print(f"\nProcessing Sample {sample_num} (user_id: {user_id})...")
        matched_df, unmatched_frames, problem_stats = process_sample(
            sample_num, user_id, df_all, base_dir, output_base_dir
        )
        
        if not matched_df.empty:
            all_matched_dfs.append(matched_df)
        
        all_stats[sample_num] = {
            'user_id': user_id,
            'matched_records': len(matched_df),
            'unmatched_frames': len(unmatched_frames),
            'problem_stats': problem_stats
        }
    
    # Combine all matched records into a single DataFrame
    if all_matched_dfs:
        combined_df = pd.concat(all_matched_dfs, ignore_index=True)
        # Save the cleaned data
        output_excel = os.path.join(output_base_dir, "all_cleaned_data.xlsx")
        combined_df.to_excel(output_excel, index=False)
        print(f"\nSaved cleaned data to {output_excel}")
    else:
        print("\nNo valid records found across all samples.")
    
    # Save detailed summary report
    summary_data = []
    for sample_num, stats in all_stats.items():
        base_row = {
            'Sample': sample_num,
            'User ID': stats['user_id'],
            'Total Matched Records': stats['matched_records'],
            'Total Unmatched Frames': stats['unmatched_frames']
        }
        
        # Add problem-specific stats
        for problem, prob_stats in stats['problem_stats'].items():
            base_row[f"{problem} Records"] = prob_stats['records']
            base_row[f"{problem} Matched"] = prob_stats['matched']
        
        summary_data.append(base_row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_base_dir, "processing_summary.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"Saved processing summary to {summary_path}")
    
    # Print summary
    print("\n===== SUMMARY =====")
    total_matched = 0
    total_unmatched = 0
    
    for sample_num, stats in all_stats.items():
        matched = stats['matched_records']
        unmatched = stats['unmatched_frames']
        user_id = stats['user_id']
        total_matched += matched
        total_unmatched += unmatched
        print(f"Sample {sample_num} (user_id: {user_id}): {matched} matched records, {unmatched} unmatched frames")
        for problem, prob_stats in stats['problem_stats'].items():
            print(f"  - {problem}: {prob_stats['matched']}/{prob_stats['records']} records matched")
    
    print(f"\nTOTAL: {total_matched} matched records, {total_unmatched} unmatched frames")

# Example usage
if __name__ == "__main__":
    # Configuration
    excel_path = "D:/Preprocessing/all_samples_data.xlsx"  # Path to your Excel file with all samples
    base_dir = "D:/Preprocessing"                         # Base directory containing all sample folders
    output_base_dir = "D:/Preprocessing/Cleaned"          # Where to save cleaned data
    
    # Define mapping between Sample number and user_id
    # This needs to be filled in manually based on your data
    sample_to_user_id = {
        1: 97,   
        2: 117,    
        3: 99,
        4: 100,
        5: 101,
        6: 103,
        7: 102,
        8: 118,
        9: 104,
        10: 106,
        11: 107,
        12: 108,
        13: 109,
        14: 110,
        15: 111,
        16: 112,
        17: 114,
        18: 113,
        19: 115, 
        20: 116
    }
    
    # Run the processing
    process_all_samples(excel_path, base_dir, output_base_dir, sample_to_user_id)