# JMLR-DRPMD
This repository contains codes that support the numerical example of the submitted paper "Policy Gradient for Robust Markov Decision Processes".

There are three folders. The folder Garnet Problem includes:

1. C++ codes for generating Garnet problems with different size,
2. C++ codes for soving the robust MDPs with our method DRPMD and the benchmark method robust value iteration,
3. C++ codes for evaluating the time of two different inner tolerance selections and,
4. Python codes for plotting the error decreasing performence.

The folder Inventory Management Problem includes:

1. Python codes for a basic pre-setted function collections,
2. Python codes for solving a generated inventory problem by using DRPMD (REINFORCE on policy update) + MCTMA (inner solver)
   with softmax policy and Gaussian mixture parametric transition kernel and,
3. Python codes for plotting the performence.

The folder Cart-Pole Problem includes:

1. Python codes for the non-robust policy trainning,
2. Python codes for the robust policy trainning within a pre-defined perturbed environment,
3. Python codes for the robust and non-robust policies testing within a new perturbed cart-pole environment (cartpole.py).

