# Evaluating Bloom Filter Variants

Deployed at https://bloomvariants.streamlit.app/

Team members: Gowri Namratha Meedinti and Miles Bramwit
Associated with : Cornell University 


This repository presents a comprehensive empirical study comparing classical, learned, and hybrid Bloom filter implementations across five real-world datasets. Our work evaluates trade-offs in accuracy, memory efficiency, and speed—providing practitioners with actionable guidance and an interactive selection tool.

## 🔍 Overview

Bloom filters are space-efficient data structures for set membership testing. While traditional filters guarantee no false negatives, learned variants use machine learning models to reduce false positives—at the cost of increased complexity and possible misclassifications.

This project:
- Benchmarks **7 Bloom filter variants**: Standard, Counting, RandomForest, LightGBM, NeuralNet, SVM, and Sandwich.
- Evaluates filters on **5 diverse, real-world datasets** spanning security, web, and infrastructure domains.
- Provides an **interactive Streamlit tool** to recommend the optimal filter based on use case constraints.

---

## 📚 Real-World Dataset Collection

We curated and cleaned five high-impact datasets to reflect practical use cases where Bloom filters are typically deployed:

| Dataset Type     | Description                                                                                     | Source |
|------------------|-------------------------------------------------------------------------------------------------|--------|
| **URLs**         | 40,000 phishing URLs vs. 40,000 highly ranked legitimate URLs (Majestic Million)               | [PhishTank](https://data.mendeley.com/datasets/vfszbj9b36/1) |
| **Passwords**    | 10,000 commonly leaked passwords vs. 10,000 synthetically generated secure passwords            | [SecLists Project](https://github.com/danielmiessler/SecLists) |
| **IP Addresses** | 40,000 malicious IPs from threat intel feeds vs. 40,000 cloud provider IPs                      | Curated from open threat intelligence sources |
| **Phone Numbers**| 580+ spam/blocked numbers vs. 580+ randomly generated clean numbers (US format)                 | Synthetic & spam reports |
| **Emails**       | 580+ spam emails from honeypots and blocklists vs. 580+ clean, randomly generated emails        | Spam traps & synthetic generation |

These datasets were chosen for their diversity in structure:
- **Structured data** (e.g., passwords, IPs) improves ML filter accuracy.
- **Unstructured data** (e.g., URLs) challenges ML filters, making traditional filters more suitable.

---

## 📊 Key Contributions

- **Comprehensive Evaluation**: Benchmarked filters across 5 datasets using 6 metrics—FPR, FNR, F1 score, memory, query time, and throughput.
- **Domain-Specific Insights**: Identified which filters work best for each data type and application goal.
- **Interactive Filter Selector**: Built a Streamlit app that guides users in selecting the most effective Bloom filter variant.

---

## 📈 Evaluation Metrics

We assess each Bloom filter variant using the following key metrics:

- **False Positive Rate (FPR)**: Likelihood of mistakenly identifying a non-member as a member.
- **False Negative Rate (FNR)** *(ML filters only)*: Likelihood of missing an actual member (undesirable in many applications).
- **F1 Score** *(ML filters only)*: Harmonic mean of precision and recall, particularly important in imbalanced datasets.
- **Memory Usage**: Bits used per element, including internal arrays or model weights.
- **Query Time**: Time taken to respond to a single query.
- **Throughput**: Number of queries processed per second.

---

## Summary of Findings

The optimal Bloom filter type varies based on dataset structure and application priorities:

| **Dataset**   | **Best for Accuracy** | **Best for Speed** | **Best for Memory** | **Best Balanced**         |
|---------------|------------------------|---------------------|----------------------|----------------------------|
| URLs          | ML Filters             | Standard BF         | ML Filters           | Standard BF                |
| Passwords     | LightGBM / RF          | Standard BF         | LightGBM             | LightGBM                   |
| IPs           | LightGBM               | Standard BF         | LightGBM             | Sandwich-LightGBM          |
| Phone Numbers | LightGBM               | Standard BF         | LightGBM             | Sandwich-LightGBM          |
| Emails        | LightGBM               | Standard BF         | LightGBM             | Sandwich BF                |

> 💡 **Note**: ML-based filters outperform on structured data (e.g., passwords, IPs), while Standard Bloom Filters remain ideal for high-throughput scenarios or unstructured data like URLs.

---

