import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

def load_transactions_from_file(file_path):
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    with open(file_path, 'r') as file:
        transactions = [line.strip().split() for line in file]
    time_taken = time.time() - start_time
    print(f"Loaded {len(transactions)} transactions. Time taken: {time_taken:.2f} seconds.")
    return transactions

def find_frequent_itemsets(transactions, min_support=0.05, max_len=2):
    print("Finding frequent itemsets...")
    start_time = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=max_len)
    time_taken = time.time() - start_time
    print(f"Frequent itemsets found: {len(frequent_itemsets)}. Time taken: {time_taken:.2f} seconds.")
    return frequent_itemsets

def generate_key(index):
    key = ''
    while index >= 0:
        key = chr(index % 36 + (48 if index % 36 < 10 else 55)) + key
        index = index // 36 - 1
    return key

def create_mapping(frequent_itemsets, max_mapped_itemsets=100):
    print("Creating mapping...")
    start_time = time.time()
    mapping = {}
    itemsets = frequent_itemsets.sort_values(by=['support', 'itemsets'], ascending=[False, True])['itemsets']
    for i, itemset in enumerate(itemsets[:max_mapped_itemsets]):
        key = generate_key(i)
        if len(itemset) > 1:
            mapping[frozenset(itemset)] = key
    time_taken = time.time() - start_time
    print(f"Mapping created with {len(mapping)} entries. Time taken: {time_taken:.2f} seconds.")
    return mapping

def compress_dataset(transactions, mapping):
    print("Compressing dataset...")
    start_time = time.time()
    compressed_transactions = []
    for transaction in transactions:
        compressed_transaction = set(transaction)
        for itemset, key in mapping.items():
            if itemset.issubset(compressed_transaction):
                compressed_transaction -= itemset
                compressed_transaction.add(key)
        compressed_transactions.append(list(compressed_transaction))
    time_taken = time.time() - start_time
    print(f"Compressed transactions count: {len(compressed_transactions)}. Time taken: {time_taken:.2f} seconds.")
    return compressed_transactions

def save_compressed_transactions(file_path, compressed_transactions):
    print(f"Saving compressed transactions to {file_path}...")
    start_time = time.time()
    with open(file_path, 'w') as file:
        for transaction in compressed_transactions:
            file.write(' '.join(transaction) + '\n')
    time_taken = time.time() - start_time
    print(f"Compressed transactions saved. Time taken: {time_taken:.2f} seconds.")

def load_compressed_transactions(file_path):
    print(f"Loading compressed transactions from {file_path}...")
    start_time = time.time()
    with open(file_path, 'r') as file:
        compressed_transactions = [line.strip().split() for line in file]
    time_taken = time.time() - start_time
    print(f"Loaded {len(compressed_transactions)} compressed transactions. Time taken: {time_taken:.2f} seconds.")
    return compressed_transactions

def decompress_dataset(compressed_transactions, mapping):
    print("Decompressing dataset...")
    start_time = time.time()
    reverse_mapping = {v: k for k, v in mapping.items()}
    decompressed_transactions = []
    for transaction in compressed_transactions:
        decompressed_transaction = set()
        for item in transaction:
            if item in reverse_mapping:
                decompressed_transaction.update(reverse_mapping[item])
            else:
                decompressed_transaction.add(item)
        decompressed_transactions.append(list(decompressed_transaction))
    time_taken = time.time() - start_time
    print(f"Decompressed transactions count: {len(decompressed_transactions)}. Time taken: {time_taken:.2f} seconds.")
    return decompressed_transactions

def calculate_compression_ratio(original_transactions, compressed_transactions, mapping):
    print("Calculating compression ratio...")
    start_time = time.time()
    original_size = sum(len(t) for t in original_transactions)
    compressed_size = sum(len(t) for t in compressed_transactions)
    mapping_size = sum(len(k) + sum(len(item) for item in v) for k, v in mapping.items())
    total_compressed_size = compressed_size + mapping_size
    compression_ratio = ((original_size - total_compressed_size) / original_size) * 100
    time_taken = time.time() - start_time
    print(f"Original Size: {original_size}, Compressed Size: {compressed_size}, Mapping Size: {mapping_size}, Total Compressed Size: {total_compressed_size}")
    print(f"Time taken: {time_taken:.2f} seconds.")
    return compression_ratio

if __name__ == "__main__":
    overall_start_time = time.time()

    file_path = 'D_small.dat'
    compressed_file_path = 'D_small_compressed.dat'
    
    transactions = load_transactions_from_file(file_path)
    frequent_itemsets = find_frequent_itemsets(transactions, min_support=0.05, max_len=2)
    mapping = create_mapping(frequent_itemsets, max_mapped_itemsets=100)
    
    compressed_transactions = compress_dataset(transactions, mapping)
    save_compressed_transactions(compressed_file_path, compressed_transactions)
    
    loaded_compressed_transactions = load_compressed_transactions(compressed_file_path)
    
    decompressed_transactions = decompress_dataset(loaded_compressed_transactions, mapping)
    compression_ratio = calculate_compression_ratio(transactions, loaded_compressed_transactions, mapping)
    
    overall_time_taken = time.time() - overall_start_time
    print(f"Compression Ratio: {compression_ratio:.2f}%")
    print(f"Total Execution Time: {overall_time_taken:.2f} seconds.")
