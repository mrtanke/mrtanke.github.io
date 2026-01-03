---
title: 'Linformer: Self-Attention with Linear Complexity'
date: 2025-12-04 15:10:45+00:00
draft: false
description: 'Paper-reading notes: Linformer'
ShowWordCount: true
ShowReadingTime: false
tags:
- attention
- efficient-llms
---


# 1. Method

## **1.1. Key Observation**

The self-attention matrix is **low-rank**. Both empirical spectra and theory show that most information is concentrated in a few singular values.

$$
P = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)
$$

## **1.2. Main Idea**

If $P$ is low-rank, then instead of computing an $n\times n$ matrix, we can approximate it using a factorization of size $n \times k$) where $k \ll n$.

## **1.3. Linear Projections**

Linformer introduces two projection matrices $E, F \in R^{n \times k}$ to reduce the sequence length of **keys** and **values**:

$$
K' = EK,\qquad V' = FV.
$$

## **1.4. Linear Self-Attention**

Attention is computed as

$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{Q {K'}^\top}{\sqrt{d}}\right)V'.
$$

This reduces complexity from $O(n^2)$ to **$O(nk)$**.

## **1.5. Practical Choices**

- k is small and independent of sequence length.
- Projections can be **shared** across heads and layers.

## **1.6. Empirical Results**

Linformer achieves similar or better accuracy than standard Transformers while being much faster and using much less memory.

# **2. Novelty**

1. **First to show self-attention is inherently low-rank.**
    
    This gives a theoretical reason why quadratic attention is unnecessary.
    
2. **Introduces a simple and general low-rank factorization of attention.**
    
    Instead of sparsity or hashing, Linformer directly compresses the sequence dimension.
    
3. **Achieves true linear complexity (O(n))** in both time and memory.
4. **Projection dimension does not grow with sequence length.**
    
    This enables extremely long-sequence training on standard hardware.
    
5. **Compatible with standard Transformer design.**
    
    No special sparsity patterns, no complex algorithms. Easy to drop in.