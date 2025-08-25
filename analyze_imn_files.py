import json
import gzip
from collections import defaultdict

def load_imns_from_gz(filename):
    """
    Load IMNs from a gzipped JSON file.
    Each line contains one IMN as JSON.
    """
    imns = []
    user_ids = set()
    
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        imn = json.loads(line)
                        imns.append(imn)
                        if 'uid' in imn:
                            user_ids.add(imn['uid'])
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num} in {filename}: {e}")
                        continue
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return [], set()
    
    return imns, user_ids

def analyze_imn_files():
    """
    Analyze both IMN files and compare user overlap.
    """
    file1 = "data/milano_2007_imns.json.gz"
    file2 = "data/test_milano_imns.json.gz"
    
    print("Analyzing IMN files...")
    print("=" * 50)
    
    # Load first file
    print(f"Loading {file1}...")
    imns1, users1 = load_imns_from_gz(file1)
    print(f"Loaded {len(imns1)} IMNs with {len(users1)} unique users")
    
    # Load second file
    print(f"\nLoading {file2}...")
    imns2, users2 = load_imns_from_gz(file2)
    print(f"Loaded {len(imns2)} IMNs with {len(users2)} unique users")
    
    # Calculate overlap
    users_intersection = users1.intersection(users2)
    users_only_in_1 = users1 - users2
    users_only_in_2 = users2 - users1
    
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS:")
    print("=" * 50)
    print(f"Total users in {file1}: {len(users1)}")
    print(f"Total users in {file2}: {len(users2)}")
    print(f"Users present in both files: {len(users_intersection)}")
    print(f"Users only in {file1}: {len(users_only_in_1)}")
    print(f"Users only in {file2}: {len(users_only_in_2)}")
    
    # Calculate overlap percentage
    if users1:
        overlap_percentage = (len(users_intersection) / len(users1)) * 100
        print(f"Overlap percentage: {overlap_percentage:.2f}% of users from {file1} are in {file2}")
    
    # Show some sample user IDs from each category
    print("\n" + "=" * 50)
    print("SAMPLE USER IDs:")
    print("=" * 50)
    
    if users_intersection:
        sample_intersection = list(users_intersection)[:10]
        print(f"Sample users in both files (first 10): {sample_intersection}")
    
    if users_only_in_1:
        sample_only_1 = list(users_only_in_1)[:10]
        print(f"Sample users only in {file1} (first 10): {sample_only_1}")
    
    if users_only_in_2:
        sample_only_2 = list(users_only_in_2)[:10]
        print(f"Sample users only in {file2} (first 10): {sample_only_2}")
    
    # Check if test_milano_imns.json.gz contains all users from the original dataset
    print("\n" + "=" * 50)
    print("DATASET COMPLETENESS CHECK:")
    print("=" * 50)
    
    # The original Milano dataset had 17,241 users
    original_total_users = 17241
    print(f"Original Milano2007_ORIGINAL.csv had {original_total_users} unique users")
    print(f"test_milano_imns.json.gz contains {len(users2)} users")
    
    if len(users2) == original_total_users:
        print("✓ test_milano_imns.json.gz contains ALL users from the original dataset")
    elif len(users2) > original_total_users:
        print(f"⚠ test_milano_imns.json.gz contains MORE users ({len(users2)}) than the original dataset")
    else:
        missing_users = original_total_users - len(users2)
        missing_percentage = (missing_users / original_total_users) * 100
        print(f"✗ test_milano_imns.json.gz is missing {missing_users} users ({missing_percentage:.2f}% of original dataset)")
    
    return {
        'file1_users': users1,
        'file2_users': users2,
        'intersection': users_intersection,
        'only_in_1': users_only_in_1,
        'only_in_2': users_only_in_2
    }

if __name__ == "__main__":
    results = analyze_imn_files()


