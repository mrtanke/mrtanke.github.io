---
title: 'Tracr: Compiled Transformers as a Laboratory for Interpretability'
date: 2025-12-08 08:32:26+00:00
draft: false
description: 'Paper-reading notes: Tracr'
ShowWordCount: true
ShowReadingTime: false
tags:
- compilers
- attention
- reasoning
---

Laboratory for Interpretability

**Tracr** translates RASP programs to transformer weights in six steps:

1. **Split** the RASP program into small steps (a).
2. **Figure out** what each step can output (a).
3. **Label** each step as MLP or Attention (b).
4. **Arrange** them into Transformer layers (c).
5. **Insert** no-op blocks to fill empty spots (c).
6. **Generate** real Transformer weights that implement each step.

![image.png](2c340e9a-8266-4e2c-9683-00b701af7b7b.png)