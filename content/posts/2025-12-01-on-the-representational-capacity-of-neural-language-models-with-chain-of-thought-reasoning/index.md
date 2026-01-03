---
title: On the Representational Capacity of Neural Language Models with Chain-of-Thought
  Reasoning
date: 2025-12-01 08:49:03+00:00
draft: false
description: 'Paper-reading notes: On the Representational Capacity of Neural Language
  Models with Chain-of-Thought Reasoning'
ShowWordCount: true
ShowReadingTime: false
tags:
- reasoning
- language-model
---


## **1. Motivation**

Traditional theoretical results on transformer “Turing-completeness” are not actually about **language models**. They:

- study **language recognition**, not **probabilistic generation**,
- and often add **extra symbols** to simulate a Turing machine.

But a real LM = a probability distribution over strings of a fixed alphabet.

This paper provides a theory of expressivity **for actual language models**, especially when augmented with **chain-of-thought (CoT)** steps.

## **2. Core Idea: Treat LMs as Probabilistic Models of Computation**

Instead of saying “this LM recognizes this language,” the paper asks: What **distributions** over strings can an LM represent? (probabilistic Turing-machine viewpoint)

This shift allows a proper comparison between:

- LMs without CoT
- LMs that generate extra intermediate steps (CoT)

## **3. Key Concept Introduced: Regular Reducibility**

CoT inserts extra reasoning tokens. To compare LMs with and without CoT, we need a way to ignore these internal steps in the output.

<aside>

### **Regular reducibility:**

LM (A) is reducible to LM (B) if:

- You can convert samples from (A) into samples from (B)
- using only a **finite-state transducer** (very simple machine).

Example: deleting or rewriting CoT reasoning steps.

</aside>

## **4. Main Results**

### **(1) CoT increases the power of RNN language models**

- **Constant-precision RNN LM (no CoT)**
    
    → equivalent to **deterministic probabilistic finite-state automata (PFSAs)**
    
    → very weak
    
- Constant-precision RNN LM **with CoT**
    
    → equivalent to **non-deterministic PFSAs**
    
    → strictly more expressive
    

So **CoT makes even weak RNN LMs more powerful**.

### **(2) Turing-complete RNNs can be viewed as doing implicit CoT**

Previous “Turing-complete RNN” constructions use complicated hidden-state motions.

The authors reinterpret these as **implicit chain-of-thought reasoning** inside the hidden state.

### **(3) The Big Result: CoT makes LMs probabilistic-Turing-complete**

With enough numeric precision:

- **Linear-precision RNN LMs + CoT**, and
- **Logarithmic-precision transformer decoder LMs + CoT**,

can simulate **any probabilistic Turing machine**.

Thus CoT raises LM generative capacity to the full power of probabilistic computation. This is the probabilistic analogue of Turing-completeness — but now for **true LMs**, not for recognizers.

## **5. Implications**

1. **CoT adds computational steps**, increasing representational power.
    
    This gives a theoretical explanation for the empirical success of CoT prompting.
    
2. **LM expressivity depends heavily on precision and intermediate computation.**
3. **Transformers without CoT (limited depth, finite precision)** behave like small probabilistic automata (very weak).
4. **Transformers with CoT** behave like full probabilistic Turing machines (maximally expressive).

## **6. Takeaway**

**Chain-of-thought** turns a limited LM into a model as expressive as a probabilistic Turing machine, by giving it extra internal computation steps that fundamentally increase its generative power.

## 7. Turing machine

<aside>

A simple abstract computer with an infinite tape, a read-write head, and a finite set of states. It follows a table of rules that tell it how to read/write symbols and move on the tape.

It can express **any algorithm**, which makes it the foundation of computability theory and the “maximal” model of expressivity.

Video source: [https://www.bilibili.com/video/BV1br4y1N762/?spm_id_from=333.337.search-card.all.click&vd_source=650390d4a2decee4b694a632313a3cca](https://www.bilibili.com/video/BV1br4y1N762/?spm_id_from=333.337.search-card.all.click&vd_source=650390d4a2decee4b694a632313a3cca)

</aside>