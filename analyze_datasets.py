#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon vs KuaiRand Dataset Comparative Analysis
"""

import os
import numpy as np
from collections import defaultdict
import json

def analyze_dataset(dataset_dir, domain_name):
    """Analyze a single domain dataset"""
    train_file = os.path.join(dataset_dir, "train_data.txt")
    
    if not os.path.exists(train_file):
        return None
    
    sequence_lengths = []
    num_users = 0
    num_items = set()
    
    with open(train_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) < 1:
                continue
            
            user_id = parts[0]
            sequence = parts[1:]
            
            num_users += 1
            sequence_lengths.append(len(sequence))
            num_items.update(sequence)
    
    # Sequence length statistics
    seq_stats = {
        'domain': domain_name,
        'num_users': num_users,
        'num_items': len(num_items),
        'seq_length_mean': np.mean(sequence_lengths),
        'seq_length_median': np.median(sequence_lengths),
        'seq_length_std': np.std(sequence_lengths),
        'seq_length_min': min(sequence_lengths),
        'seq_length_max': max(sequence_lengths),
        'seq_length_q25': np.percentile(sequence_lengths, 25),
        'seq_length_q75': np.percentile(sequence_lengths, 75),
        'total_interactions': sum(sequence_lengths),
        'sparsity': 1 - (sum(sequence_lengths) / (num_users * len(num_items)) if num_users > 0 and len(num_items) > 0 else 0)
    }
    
    return seq_stats, sequence_lengths

def main():
    base_data_dir = "/home/dxlab/jupyter/seungjin/FedDCSR/data"
    
    # Amazon dataset (by category)
    amazon_domains = ['Beauty', 'Books', 'Clothing', 'Food', 'Games', 'Garden', 'Home', 'Kitchen', 'Movies', 'Sports']
    amazon_results = {}
    
    print("=" * 100)
    print("AMAZON DATASET ANALYSIS")
    print("=" * 100)
    
    for domain in amazon_domains:
        domain_dir = os.path.join(base_data_dir, domain)
        result = analyze_dataset(domain_dir, domain)
        if result:
            stats, _ = result
            amazon_results[domain] = stats
            print(f"\n{domain:15} | Users: {stats['num_users']:5} | Items: {stats['num_items']:5} | "
                  f"Avg Seq Len: {stats['seq_length_mean']:6.2f} | Std Dev: {stats['seq_length_std']:6.2f} | "
                  f"Min-Max: {stats['seq_length_min']:3}-{stats['seq_length_max']:3} | "
                  f"Sparsity: {stats['sparsity']:.4f}")
    
    # KuaiRand dataset (by tab)
    kuairand_bases = [
        'kuairand/Tab_filter0', 'kuairand/Tab_filter1', 'kuairand/Tab_filter2', 'kuairand/Tab_filter4',
        'kuairand/Tab0', 'kuairand/Tab1', 'kuairand/Tab2', 'kuairand/Tab4',
        'kuairand/Tablong0', 'kuairand/Tablong1', 'kuairand/Tablong2', 'kuairand/Tablong4'
    ]
    kuairand_results = {}
    
    print("\n" + "=" * 100)
    print("KUAIRAND DATASET ANALYSIS")
    print("=" * 100)
    
    for domain_path in kuairand_bases:
        domain_dir = os.path.join(base_data_dir, domain_path)
        domain_name = domain_path.split('/')[-1]
        result = analyze_dataset(domain_dir, domain_name)
        if result:
            stats, _ = result
            kuairand_results[domain_name] = stats
            print(f"\n{domain_name:20} | Users: {stats['num_users']:5} | Items: {stats['num_items']:5} | "
                  f"Avg Seq Len: {stats['seq_length_mean']:6.2f} | Std Dev: {stats['seq_length_std']:6.2f} | "
                  f"Min-Max: {stats['seq_length_min']:3}-{stats['seq_length_max']:3} | "
                  f"Sparsity: {stats['sparsity']:.4f}")
    
    # Comparison summary
    print("\n" + "=" * 100)
    print("AMAZON vs KUAIRAND COMPARISON SUMMARY")
    print("=" * 100)
    
    if amazon_results:
        amazon_stats = list(amazon_results.values())
        amazon_avg_seq = np.mean([s['seq_length_mean'] for s in amazon_stats])
        amazon_avg_users = np.mean([s['num_users'] for s in amazon_stats])
        amazon_avg_items = np.mean([s['num_items'] for s in amazon_stats])
        amazon_avg_sparsity = np.mean([s['sparsity'] for s in amazon_stats])
        
        print(f"\n🔴 Amazon Dataset (Average):")
        print(f"   - Average Users per Domain: {amazon_avg_users:.1f}")
        print(f"   - Average Items per Domain: {amazon_avg_items:.1f}")
        print(f"   - Average Sequence Length: {amazon_avg_seq:.2f}")
        print(f"   - Average Sparsity: {amazon_avg_sparsity:.4f}")
        print(f"   - Sequence Length Range: {min([s['seq_length_min'] for s in amazon_stats])} - {max([s['seq_length_max'] for s in amazon_stats])}")
        print(f"   - Sequence Length Std Dev Range: {min([s['seq_length_std'] for s in amazon_stats]):.2f} - {max([s['seq_length_std'] for s in amazon_stats]):.2f}")
    
    if kuairand_results:
        kuairand_stats = list(kuairand_results.values())
        kuairand_avg_seq = np.mean([s['seq_length_mean'] for s in kuairand_stats])
        kuairand_avg_users = np.mean([s['num_users'] for s in kuairand_stats])
        kuairand_avg_items = np.mean([s['num_items'] for s in kuairand_stats])
        kuairand_avg_sparsity = np.mean([s['sparsity'] for s in kuairand_stats])
        
        print(f"\n🔵 KuaiRand Dataset (Average):")
        print(f"   - Average Users per Domain: {kuairand_avg_users:.1f}")
        print(f"   - Average Items per Domain: {kuairand_avg_items:.1f}")
        print(f"   - Average Sequence Length: {kuairand_avg_seq:.2f}")
        print(f"   - Average Sparsity: {kuairand_avg_sparsity:.4f}")
        print(f"   - Sequence Length Range: {min([s['seq_length_min'] for s in kuairand_stats])} - {max([s['seq_length_max'] for s in kuairand_stats])}")
        print(f"   - Sequence Length Std Dev Range: {min([s['seq_length_std'] for s in kuairand_stats]):.2f} - {max([s['seq_length_std'] for s in kuairand_stats]):.2f}")
    
    # Cross-domain sequence length diversity
    print("\n" + "=" * 100)
    print("CROSS-DOMAIN SEQUENCE LENGTH DIFFERENCE ANALYSIS")
    print("=" * 100)
    
    if amazon_results:
        amazon_seq_lens = [s['seq_length_mean'] for s in amazon_results.values()]
        print(f"\n🔴 Amazon Cross-Domain Sequence Length Difference:")
        print(f"   - Max: {max(amazon_seq_lens):.2f} ({[k for k, v in amazon_results.items() if v['seq_length_mean'] == max(amazon_seq_lens)][0]})")
        print(f"   - Min: {min(amazon_seq_lens):.2f} ({[k for k, v in amazon_results.items() if v['seq_length_mean'] == min(amazon_seq_lens)][0]})")
        print(f"   - Difference: {max(amazon_seq_lens) - min(amazon_seq_lens):.2f}")
        print(f"   - Std Dev: {np.std(amazon_seq_lens):.2f}")
        print(f"   - Coefficient of Variation (CV): {np.std(amazon_seq_lens) / np.mean(amazon_seq_lens):.4f}")
    
    if kuairand_results:
        kuairand_seq_lens = [s['seq_length_mean'] for s in kuairand_results.values()]
        print(f"\n🔵 KuaiRand Cross-Domain Sequence Length Difference:")
        print(f"   - Max: {max(kuairand_seq_lens):.2f} ({[k for k, v in kuairand_results.items() if v['seq_length_mean'] == max(kuairand_seq_lens)][0]})")
        print(f"   - Min: {min(kuairand_seq_lens):.2f} ({[k for k, v in kuairand_results.items() if v['seq_length_mean'] == min(kuairand_seq_lens)][0]})")
        print(f"   - Difference: {max(kuairand_seq_lens) - min(kuairand_seq_lens):.2f}")
        print(f"   - Std Dev: {np.std(kuairand_seq_lens):.2f}")
        print(f"   - Coefficient of Variation (CV): {np.std(kuairand_seq_lens) / np.mean(kuairand_seq_lens):.4f}")
    
    # Intra-domain sequence length diversity
    print("\n" + "=" * 100)
    print("INTRA-DOMAIN SEQUENCE LENGTH DIVERSITY (ACROSS USERS WITHIN SAME DOMAIN)")
    print("=" * 100)
    
    print(f"\n🔴 Amazon Intra-Domain Sequence Length Std Dev (across users):")
    amazon_within_std = [s['seq_length_std'] for s in amazon_results.values()]
    for domain, std in zip(amazon_results.keys(), amazon_within_std):
        print(f"   - {domain:15}: Std Dev = {std:6.2f}")
    print(f"   - Average Std Dev: {np.mean(amazon_within_std):.2f}")
    
    print(f"\n🔵 KuaiRand Intra-Domain Sequence Length Std Dev (across users):")
    kuairand_within_std = [s['seq_length_std'] for s in kuairand_results.values()]
    for domain, std in zip(kuairand_results.keys(), kuairand_within_std):
        print(f"   - {domain:20}: Std Dev = {std:6.2f}")
    print(f"   - Average Std Dev: {np.mean(kuairand_within_std):.2f}")
    
    # Other characteristics
    print("\n" + "=" * 100)
    print("OTHER DATA CHARACTERISTICS COMPARISON")
    print("=" * 100)
    
    if amazon_results and kuairand_results:
        amazon_total_users = sum([s['num_users'] for s in amazon_results.values()])
        amazon_total_items = sum([s['num_items'] for s in amazon_results.values()])
        amazon_total_interactions = sum([s['total_interactions'] for s in amazon_results.values()])
        
        kuairand_total_users = sum([s['num_users'] for s in kuairand_results.values()])
        kuairand_total_items = sum([s['num_items'] for s in kuairand_results.values()])
        kuairand_total_interactions = sum([s['total_interactions'] for s in kuairand_results.values()])
        
        print(f"\nOverall Data Scale:")
        print(f"   Amazon:   {amazon_total_users:,} users | {amazon_total_items:,} items | {amazon_total_interactions:,} interactions")
        print(f"   KuaiRand: {kuairand_total_users:,} users | {kuairand_total_items:,} items | {kuairand_total_interactions:,} interactions")
        
        print(f"\nSparsity Comparison:")
        amazon_sparsity_overall = 1 - (amazon_total_interactions / (amazon_total_users * amazon_total_items))
        kuairand_sparsity_overall = 1 - (kuairand_total_interactions / (kuairand_total_users * kuairand_total_items))
        print(f"   Amazon:   {amazon_sparsity_overall:.6f}")
        print(f"   KuaiRand: {kuairand_sparsity_overall:.6f}")

if __name__ == "__main__":
    main()
