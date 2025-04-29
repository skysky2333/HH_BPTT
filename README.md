# Backpropagation-through-time training of an unrolled Hodgkin--Huxley model for automatic conductance estimation


Precise estimation of biophysical parameters such as ion channel conductances in single neurons is essential for understanding neuronal excitability and for building accurate computational models. However, inferring these parameters from experimental data is challenging, often requiring extensive voltage-clamp measurements or laborious manual tuning. Here we present a novel approach that leverages backpropagation through time (BPTT) to automatically fit a Hodgkin-Huxley (HH) conductance-based model to observed voltage traces. We unroll the HH model dynamics in time and treat the unknown maximum conductances as learnable parameters in a differentiable simulation. By optimizing the model to minimize the mean squared error between simulated and observed membrane voltage, our method directly recovers the underlying conductances (for sodium g_Na, potassium g_K, and leak g_L) from a single voltage response. In simulations, the BPTT-trained model accurately identified conductance values across different neuron types and remained robust to typical levels of measurement noise. Even with a single current-clamp recording as training data, the approach achieved precise fits, highlighting its efficiency. This work demonstrates a powerful automated strategy for biophysical system identification, opening the door to rapid, high-fidelity neuron model customization from electrophysiological recordings.

---

This repository contains the code accompanying the:

[preprint](https://www.biorxiv.org/content/10.1101/2025.04.27.650871v1)
