---
title: 'Thinking Like Transformers'
date: 2025-12-07T15:14:48+00:00
draft: false
description: 'Paper-reading notes: RASP'
tag: 'Notes'
ShowWordCount: true
ShowReadingTime: false
---


## **1. What is RASP?**

A small, symbolic language that models how Transformers compute.

- Describes **sequence operations** using simple functions instead of neural weights.

## **2. Core Components**

### **2.1 s-ops (sequence operators)**

Functions that take a sequence and return another sequence of the same length.

Examples:

- `tokens` → original input
- `indices` → [0,1,2,...]
- `length` → broadcast length
- elementwise ops: +, ==, %, conditionals
    
    → These mimic **MLP/FFN** behavior.
    

### **2.2 select (symbolic attention)**

<aside>

select(q, k, predicate)

</aside>

- Produces an n×n boolean matrix → attention pattern.
- Tells which positions “attend” to which.

### **2.3 aggregate (value combination)**

<aside>

aggregate(selector, values)

</aside>

- For each position: gather values from selected positions and average them.
- Symbolic version of **attention-value combination**.

### **2.4 selector_width**

- Counts how many positions were selected for each token.
- Used for **counting / histogram** tasks.

## **3. What RASP Can Express**

RASP programs can represent many Transformer-computable tasks:

- Histogram
- Double histogram
- Sequence reversal
- Matching parentheses
- Dyck languages
- Filtering, counting, boolean logic over sequences

Shows that Transformers can perform structured, compositional operations.

## **4. How RASP Models Transformers**

- **Elementwise s-ops → Transformer MLP**
- **select + aggregate → Self-attention**
- **No loops → fixed-depth computation**, like real Transformers
- Models information flow limits: only attention can move information across tokens.

## **5. Examples**

### **Reverse sequence**

<aside>

flip = select(indices, length - indices - 1, ==)
reverse = aggregate(flip, tokens)

</aside>

### **Histogram**

<aside>

same = select(tokens, tokens, ==)
hist = selector_width(same)

</aside>

## **6. Compilation**

RASP programs can be compiled into real Transformer architectures:

- number of layers
- number of heads
- attention patterns

Enables empirical testing of symbolic algorithms.

## **7. Key Insights**

RASP provides a **clean, understandable model** of Transformer computation. It highlights what Transformers can and cannot compute. It acts like **pseudocode for attention-based algorithms**.