---
title: 'What Formal Languages Can Transformers Express? A Survey'
date: 2025-11-30T08:47:55+00:00
draft: false
description: 'Paper-reading notes: What Formal Languages Can Transformers Express? A Survey'
tag: 'Notes'
ShowWordCount: true
ShowReadingTime: false
---


# **1. Core Question**

Transformers work extremely well, but **what can they theoretically express or compute?** (Not about training, but about the architecture’s raw capability.)

Expressivity is studied from two angles:

- **Approximation theory:** Can transformers approximate certain functions?
- **Formal language theory (the focus):** Can transformers recognize/generate certain **formal languages**?

# **2. Formal Background**

### **2.1. Chomsky hierarchy**

- **Regular** → finite automaton
- **Context-free** → pushdown automaton
- **Context-sensitive** → bounded-tape Turing machine
- **Recursively enumerable** → full Turing machine

### **2.2. Circuit hierarchy**

Transformers behave like constant-depth parallel circuits, so we compare with:

- **AC⁰** (very weak — no PARITY)
- **TC⁰** (MAJORITY allowed)
- **NC¹** (log-depth circuits, stronger)

### **2.3. DLOGTIME-uniform**

Means: the circuit for length n must be generable by a very simple, O(log n)-time algorithm.

Equivalent to: **no cheating, one systematic architecture**, just like a real transformer.

# **3. Key Positive Results**

Depending on assumptions:

- With **hard attention**, special positional encodings, or **extra intermediate steps**
    
    → transformers can simulate **arbitrary algorithms** (Turing-complete).
    
- With **scratchpads / chain-of-thought**
    
    → decoder-only LMs can match **probabilistic Turing machines** in expressivity.
    
- With sinusoidal or arbitrary positional encodings
    
    → they can handle some **context-free** patterns (e.g., limited Dyck languages).
    

# **4. Key Negative Results**

Under **realistic constraints**:

- constant depth
- finite precision
- softmax attention
- no scratchpad / CoT

transformers collapse to **very weak circuit classes** (AC⁰ or TC⁰).

Consequences:

- They **cannot express PARITY** on unbounded input length.
- They have trouble with even simple forms of **counting** or **nesting** (e.g., Dyck-k).
- They cannot recognize certain languages beyond TC⁰.

**Interpretation:**

A shallow, finite-precision transformer cannot compute global, unbounded structure.

# **5. Ramifications (Why this matters)**

- Theory explains why LLMs struggle with long-range algorithmic tasks.
- Theory explains why **chain-of-thought** reliably boosts reasoning: **extra generated steps dramatically increase theoretical power.**
- It clarifies how sensitive expressivity is to assumptions like precision, depth, attention type, and positional encoding.

# **6. Takeaway**

A finite-precision transformer with fixed depth is as weak as a small parallel circuit (TC⁰), but with hard attention or chain-of-thought steps, its expressivity can rise all the way to Turing-complete.