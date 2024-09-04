import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import time

def load_transactions_from_file(file_path):
    with open(file_path, 'r') as file:
        transactions = [line.strip().split() for line in file]
    return transactions

def find_frequent_itemsets(transactions, min_support=0.5):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def create_mapping(frequent_itemsets):
    mapping = {}
    itemsets = frequent_itemsets.sort_values(by='support', ascending=False)['itemsets']
    for i, itemset in enumerate(itemsets):
        key = f"Z{i}"  # Using "Z" followed by the index
        mapping[key] = set(itemset)
    return mapping

def compress_dataset(transactions, mapping):
    compressed_transactions = []
    for transaction in transactions:
        compressed_transaction = set(transaction)
        for key, itemset in mapping.items():
            if itemset.issubset(compressed_transaction):
                compressed_transaction -= itemset
                compressed_transaction.add(key)
        compressed_transactions.append(list(compressed_transaction))
    return compressed_transactions

def decompress_dataset(compressed_transactions, mapping):
    reverse_mapping = {k: v for k, v in mapping.items()}
    decompressed_transactions = []
    for transaction in compressed_transactions:
        decompressed_transaction = set()
        for item in transaction:
            if item in reverse_mapping:
                decompressed_transaction.update(reverse_mapping[item])
            else:
                decompressed_transaction.add(item)
        decompressed_transactions.append(list(decompressed_transaction))
    return decompressed_transactions

def calculate_compression_ratio(original_transactions, compressed_transactions, mapping):
    original_size = sum(len(t) for t in original_transactions)
    compressed_size = sum(len(t) for t in compressed_transactions)
    mapping_size = sum(len(v) + 1 for v in mapping.values())  # +1 for each key
    total_size = compressed_size + mapping_size
    compression_ratio = (original_size - total_size) / original_size * 100
    return compression_ratio

if __name__ == "__main__":
    file_path = 'D_medium.dat'  # Updated to D_medium.dat
    
    print("Loading data from D_medium.dat...")
    start_time = time.time()
    
    # Load transactions from the .dat file
    transactions = load_transactions_from_file(file_path)
    print(f"Loaded {len(transactions)} transactions.")
    
    # Step 1: Mine Frequent Itemsets
    print("Finding frequent itemsets...")
    frequent_itemsets = find_frequent_itemsets(transactions, min_support=0.5)
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")
    
    # Step 2: Create Mapping
    print("Creating mapping...")
    mapping = create_mapping(frequent_itemsets)
    print(f"Mapping created with {len(mapping)} entries.")
    
    # Step 3: Compress Dataset
    print("Compressing dataset...")
    compressed_transactions = compress_dataset(transactions, mapping)
    print(f"Compressed transactions count: {len(compressed_transactions)}")
    
    # Step 4: Decompress Dataset
    print("Decompressing dataset...")
    decompressed_transactions = decompress_dataset(compressed_transactions, mapping)
    print(f"Decompressed transactions count: {len(decompressed_transactions)}")
    
    # Step 5: Calculate Compression Ratio
    compression_ratio = calculate_compression_ratio(transactions, compressed_transactions, mapping)
    print(f"Compression Ratio: {compression_ratio:.2f}%")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
