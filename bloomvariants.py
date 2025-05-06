import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time
import math
import random
from collections import defaultdict

# Set page config
st.set_page_config(
    page_title="Bloom Filter Visualization",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# App title and description
st.title("Bloom Filter Comparison Tool")
st.markdown("""
This app demonstrates different types of Bloom Filters and their performance characteristics.
Explore standard, counting, and ML-based Bloom filters with visualizations of their metrics.
""")

# Define Bloom Filter classes
class StandardBloomFilter:
    def __init__(self, n, fp_rate):
        self.size = self._get_size(n, fp_rate)
        self.hash_count = self._get_hash_count(self.size, n)
        self.bit_array = [0] * self.size
        self.name = "Standard Bloom Filter"

    def _get_size(self, n, p):
        m = -(n * math.log(p)) / (math.log(2)**2)
        return int(m)

    def _get_hash_count(self, m, n):
        return int((m / n) * math.log(2))

    def add(self, item):
        for i in range(self.hash_count):
            idx = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            self.bit_array[idx] = 1

    def query(self, item):
        for i in range(self.hash_count):
            idx = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            if self.bit_array[idx] == 0:
                return False
        return True

class CountingBloomFilter:
    def __init__(self, n, fp_rate):
        self.size = self._get_size(n, fp_rate)
        self.hash_count = self._get_hash_count(self.size, n)
        self.count_array = [0] * self.size
        self.name = "Counting Bloom Filter"

    def _get_size(self, n, p):
        return int(-(n * math.log(p)) / (math.log(2)**2))

    def _get_hash_count(self, m, n):
        return int((m / n) * math.log(2))

    def add(self, item):
        for i in range(self.hash_count):
            idx = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            self.count_array[idx] += 1

    def remove(self, item):
        for i in range(self.hash_count):
            idx = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            self.count_array[idx] = max(0, self.count_array[idx] - 1)

    def query(self, item):
        for i in range(self.hash_count):
            idx = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16) % self.size
            if self.count_array[idx] == 0:
                return False
        return True

class SimpleMLBloomFilter:
    def __init__(self, n, fp_rate):
        # This is a simplified ML filter for demo purposes
        self.positives = set()  # We'll just use a set for the demo
        self.size = int(-(n * math.log(fp_rate)) / (math.log(2)**2))
        self.name = "ML-based Bloom Filter (Simulated)"
    
    def add(self, item):
        self.positives.add(item)
    
    def query(self, item):
        # For simulation, we'll return true for exact matches
        # plus a small chance of false positives based on hash
        if item in self.positives:
            return True
        # Simple hash-based false positive simulation
        hash_val = int(hashlib.md5(item.encode()).hexdigest(), 16) % 1000
        return hash_val < 100  # ~10% false positive rate
        
class SandwichBloomFilter:
    def __init__(self, n, fp_rate):
        self.ml = SimpleMLBloomFilter(n, fp_rate*2)
        self.small = StandardBloomFilter(n, fp_rate/2)
        self.name = "Sandwich Bloom Filter (ML + Standard)"
        
    def add(self, item):
        self.ml.add(item)
        self.small.add(item)
        
    def query(self, item):
        if self.ml.query(item):
            return True
        else:
            return self.small.query(item)

# Function to get memory usage (simplified for the demo)
def get_memory_usage(bf):
    if isinstance(bf, StandardBloomFilter):
        return len(bf.bit_array) / 8  # bits to bytes
    elif isinstance(bf, CountingBloomFilter):
        return len(bf.count_array)  # Each count is 1 byte for simplicity
    elif isinstance(bf, SimpleMLBloomFilter):
        return len(bf.positives) * 20  # Rough approximation
    elif isinstance(bf, SandwichBloomFilter):
        return get_memory_usage(bf.ml) + get_memory_usage(bf.small)
    return 0

# Function to evaluate Bloom Filter performance
def evaluate_bloom_filter(bf, positives, negatives):
    # Add positive items
    start_add = time.time()
    for item in positives:
        bf.add(item)
    add_time = time.time() - start_add
    
    # Test positives (true positives)
    start_tp = time.time()
    true_positives = sum(1 for item in positives if bf.query(item))
    tp_time = time.time() - start_tp
    
    # Test negatives (false positives)
    start_fp = time.time()
    false_positives = sum(1 for item in negatives if bf.query(item))
    fp_time = time.time() - start_fp
    
    # Calculate metrics
    true_positive_rate = true_positives / len(positives) if positives else 0
    false_positive_rate = false_positives / len(negatives) if negatives else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positive_rate
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Query performance
    avg_query_time = (tp_time + fp_time) / (len(positives) + len(negatives))
    throughput = (len(positives) + len(negatives)) / (tp_time + fp_time) if (tp_time + fp_time) > 0 else 0
    
    # Memory usage
    memory_bytes = get_memory_usage(bf)
    
    return {
        "Filter Name": bf.name,
        "True Positive Rate": true_positive_rate * 100,
        "False Positive Rate": false_positive_rate * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1 Score": f1_score * 100,
        "Add Time (s)": add_time,
        "Avg Query Time (s)": avg_query_time,
        "Throughput (q/s)": throughput,
        "Memory (bytes)": memory_bytes
    }

# Generate sample data
def generate_sample_data(size, length=10, charset="abcdefghijklmnopqrstuvwxyz0123456789"):
    data = []
    for _ in range(size):
        item = ''.join(random.choice(charset) for _ in range(length))
        data.append(item)
    return data

# Sidebar - Configuration parameters
st.sidebar.header("Configuration")

# Dataset options
dataset_type = st.sidebar.selectbox(
    "Select Dataset Type",
    ["Random Strings", "Email Addresses", "IP Addresses", "URLs", "Phone Numbers"]
)

if dataset_type == "Random Strings":
    charset = "abcdefghijklmnopqrstuvwxyz0123456789"
    length = st.sidebar.slider("String Length", 5, 20, 10)
elif dataset_type == "Email Addresses":
    charset = "abcdefghijklmnopqrstuvwxyz0123456789@."
    length = st.sidebar.slider("Email Length", 10, 30, 15)
elif dataset_type == "IP Addresses":
    charset = "0123456789."
    length = st.sidebar.slider("IP Length", 7, 15, 12)  # IPv4 format
elif dataset_type == "URLs":
    charset = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;="
    length = st.sidebar.slider("URL Length", 15, 50, 25)
elif dataset_type == "Phone Numbers":
    charset = "0123456789+-() "
    length = st.sidebar.slider("Phone Number Length", 10, 20, 12)

# Dataset size parameters
positive_size = st.sidebar.slider("Number of Positive Items", 10, 1000, 100)
negative_size = st.sidebar.slider("Number of Negative Items", 10, 1000, 100)

# Bloom filter parameters
fp_rate = st.sidebar.slider("Target False Positive Rate", 0.001, 0.5, 0.1, 0.001, format="%.3f")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bloom Filter Selection")
    filter_types = st.multiselect(
        "Select filters to compare",
        ["Standard Bloom Filter", "Counting Bloom Filter", "ML-based Bloom Filter (Simulated)", "Sandwich Bloom Filter"],
        ["Standard Bloom Filter", "Counting Bloom Filter"]
    )
    
    if st.button("Generate Data & Evaluate"):
        # Generate sample data
        positives = generate_sample_data(positive_size, length, charset)
        negatives = generate_sample_data(negative_size, length, charset)
        
        # Make sure positives and negatives don't overlap
        negatives = [item for item in negatives if item not in set(positives)]
        
        # Create and evaluate selected bloom filters
        results = []
        
        if "Standard Bloom Filter" in filter_types:
            bf = StandardBloomFilter(positive_size, fp_rate)
            results.append(evaluate_bloom_filter(bf, positives, negatives))
            
        if "Counting Bloom Filter" in filter_types:
            bf = CountingBloomFilter(positive_size, fp_rate)
            results.append(evaluate_bloom_filter(bf, positives, negatives))
            
        if "ML-based Bloom Filter (Simulated)" in filter_types:
            bf = SimpleMLBloomFilter(positive_size, fp_rate)
            results.append(evaluate_bloom_filter(bf, positives, negatives))
            
        if "Sandwich Bloom Filter" in filter_types:
            bf = SandwichBloomFilter(positive_size, fp_rate)
            results.append(evaluate_bloom_filter(bf, positives, negatives))
        
        # Convert results to DataFrame
        df_results = pd.DataFrame(results)
        
        # Display results in a pretty table
        st.subheader("Performance Results")
        st.dataframe(df_results.style.format({
            "True Positive Rate": "{:.2f}%",
            "False Positive Rate": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "Add Time (s)": "{:.6f}",
            "Avg Query Time (s)": "{:.6f}",
            "Throughput (q/s)": "{:.1f}",
            "Memory (bytes)": "{:,.0f}"
        }))
        
        # Store results in session state for visualization
        st.session_state.results = df_results
        st.session_state.positives = positives
        st.session_state.negatives = negatives

with col2:
    st.subheader("Visualizations")
    
    # Check if we have results to visualize
    if 'results' in st.session_state:
        # Performance metrics to visualize
        metrics = st.multiselect(
            "Select metrics to visualize",
            ["False Positive Rate", "Memory (bytes)", "Avg Query Time (s)", "Throughput (q/s)", "F1 Score"],
            ["False Positive Rate", "Memory (bytes)"]
        )
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Use appropriate colors for each metric
            if metric == "False Positive Rate":
                palette = "Reds_d"
            elif metric == "Memory (bytes)":
                palette = "Blues_d"
            elif metric == "Avg Query Time (s)":
                palette = "Greens_d"
            elif metric == "Throughput (q/s)":
                palette = "Purples_d"
            else:
                palette = "Oranges_d"
                
            # Create bar plot
            sns.barplot(
                x=metric, 
                y="Filter Name", 
                data=st.session_state.results,
                palette=palette,
                ax=ax
            )
            
            plt.title(f"{metric} by Filter Type")
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            
        # Add interactive element - test with custom string
        st.subheader("Test a Custom String")
        custom_string = st.text_input("Enter a string to test against the filters")
        
        if custom_string and st.button("Test"):
            # Create filters again (not ideal but simple for this demo)
            results = []
            is_positive = custom_string in st.session_state.positives
            is_negative = custom_string in st.session_state.negatives
            
            filter_results = {}
            
            if "Standard Bloom Filter" in filter_types:
                bf = StandardBloomFilter(positive_size, fp_rate)
                for item in st.session_state.positives:
                    bf.add(item)
                filter_results["Standard Bloom Filter"] = bf.query(custom_string)
                
            if "Counting Bloom Filter" in filter_types:
                bf = CountingBloomFilter(positive_size, fp_rate)
                for item in st.session_state.positives:
                    bf.add(item)
                filter_results["Counting Bloom Filter"] = bf.query(custom_string)
                
            if "ML-based Bloom Filter (Simulated)" in filter_types:
                bf = SimpleMLBloomFilter(positive_size, fp_rate)
                for item in st.session_state.positives:
                    bf.add(item)
                filter_results["ML-based Bloom Filter (Simulated)"] = bf.query(custom_string)
                
            if "Sandwich Bloom Filter" in filter_types:
                bf = SandwichBloomFilter(positive_size, fp_rate)
                for item in st.session_state.positives:
                    bf.add(item)
                filter_results["Sandwich Bloom Filter"] = bf.query(custom_string)
            
            # Display test results
            st.write("String in positive set:" + (" ✅ Yes" if is_positive else " ❌ No"))
            st.write("String in negative set:" + (" ✅ Yes" if is_negative else " ❌ No"))
            st.write("### Filter Results")
            
            for filter_name, result in filter_results.items():
                result_icon = "✅ Present" if result else "❌ Not present"
                correct = (result and is_positive) or (not result and not is_positive and not is_negative)
                accuracy = "✅ Correct" if correct else "❌ False positive/negative"
                st.write(f"{filter_name}: {result_icon} - {accuracy}")
    else:
        st.info("Select filter types and click 'Generate Data & Evaluate' to see visualizations")

# Educational section
st.markdown("""
## About Bloom Filters

### What is a Bloom Filter?
A Bloom filter is a space-efficient probabilistic data structure designed to test whether an element is a member of a set. False positive matches are possible, but false negatives are not – in other words, a query returns either "possibly in set" or "definitely not in set".

### Types of Bloom Filters

1. **Standard Bloom Filter**: Uses a bit array and multiple hash functions to represent set membership.
   - Pros: Memory efficient, constant time operations
   - Cons: Cannot remove elements, only supports membership queries

2. **Counting Bloom Filter**: Uses counters instead of bits to allow element removal.
   - Pros: Supports element removal, constant time operations
   - Cons: Uses more memory than standard filters

3. **ML-based Bloom Filter**: Uses machine learning to predict set membership.
   - Pros: Can be more memory efficient for certain data types
   - Cons: Training overhead, potential for false negatives

4. **Sandwich Bloom Filter**: Combines ML and traditional approaches for better accuracy.
   - Pros: Offers best of both worlds with better accuracy
   - Cons: More complex implementation, higher query time

### Applications
- Web cache filtering
- Network routing
- Database optimization
- Spell checking
- Cryptocurrency and blockchain
- Cybersecurity (password checks, malware detection)
""")

# Footer
st.markdown("---")
st.markdown("Bloom Filter Comparison Tool | Created with Streamlit")