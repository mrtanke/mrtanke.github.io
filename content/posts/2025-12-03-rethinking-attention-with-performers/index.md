---
title: Rethinking Attention with Performers
date: 2025-12-03 15:09:23+00:00
draft: false
description: 'Paper-reading notes: Performers'
ShowWordCount: true
ShowReadingTime: false
tags:
- attention
- efficient-llms
---


# 1. Introduction

Transformers are widely used in many areas, but their **softmax attention** requires **quadratic time and memory**, making them expensive for long sequences. Because of this limitation, many prior works propose “efficient attention” by adding **structural assumptions**.

Common ideas include:

- limiting attention to local neighbors,
- enforcing sparsity,
- using pooling or clustering (e.g., k-means),
- hashing similar tokens together,
- sliding windows,
- or using low-rank approximations of the attention matrix.

These methods reduce cost but depend heavily on **hand-designed priors** such as sparsity patterns or low-rank structure. They often lack theoretical guarantees, and some approaches still fail to capture long-range interactions or introduce approximation bias.

**Performers** provide a new solution: they approximate **full softmax attention** accurately while using only **linear** time and memory. The key technique, **FAVOR+** (Fast Attention Via positive Orthogonal Random features), approximates softmax and other kernels using random features with strong theoretical guarantees (unbiased or nearly-unbiased estimation and low variance).

Because FAVOR+ works for many kernelizable attention mechanisms, Performers make it feasible to compare different attention kernels on large-scale tasks, something that was previously too expensive with traditional quadratic attention.

Finally, experiments show that Performers achieve competitive performance on tasks from pixel prediction to text and protein modeling, while remaining fully compatible with standard Transformer architectures.

# 2. Method

Performers reformulate softmax attention as a **kernel function** and then approximating this kernel using **random feature maps** so that attention can be computed without constructing the quadratic matrix.

The key observation is that the softmax kernel $\exp(q^\top k)$ can be written as an expectation over random features. Performer introduces **Positive Random Features (PRF)** that approximate $\exp(q^\top k)$ using mappings of the form

$$
\phi(x) = h(x) (\exp(\omega_1^\top x),\dots, \exp(\omega_m^\top x)),
$$

where $h(x)$ is a stabilizing factor and $\omega_i$ are sampled from an isotropic Gaussian distribution. Unlike classical Fourier features (cos), PRFs always produce **positive**, **non-oscillatory** values, yielding an **unbiased** approximation to the softmax kernel that remains stable even when dot products are small or negative. This allows the softmax kernel to be approximated by

$$
\exp(q^\top k) \approx \phi(q)^\top \phi(k),
$$

so the attention can be computed as

$$
\widetilde{\text{Att}}(Q,K,V) = D^{-1}\big(Q'((K')^\top V)\big),
$$

which requires only **O(Lrd)** operations and never forms an $L \times L$ matrix.

To further reduce variance, Performer augments PRFs with **Orthogonal Random Features (ORF)**. Here, the random vectors $\omega_1,\dots,\omega_m$ are orthogonalized (e.g., by Gram–Schmidt), significantly tightening concentration bounds and producing exponentially smaller variance compared to independently sampled features. 

The combined **PRF + ORF** mechanism forms FAVOR+ (Fast Attention Via Orthogonal Random features), achieving high accuracy with relatively few random features $r \ll L$. This enables Performers to match or surpass the accuracy of softmax attention while using **linear time and memory**.

# 3. Novelty

Performers introduce **FAVOR+**, a new linear-time attention mechanism that accurately approximates softmax using **positive orthogonal random features**. 

This replaces the quadratic $(L^2)$ attention matrix with a linear $O(Lrd)$ computation while preserving accuracy. 

The method is **unbiased, low-variance, and fully compatible** with standard Transformers, enabling fast and memory-efficient training on long sequences.