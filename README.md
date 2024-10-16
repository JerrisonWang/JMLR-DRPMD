# JMLR-DRPMD
This repository contains codes that support the numerical example of the submitted paper "Policy Gradient for Robust Markov Decision Processes".

There are three folders. The folder Garnet Problem includes:

1. C++ codes for generating Garnet problems with different size,
2. C++ codes for soving the robust MDPs with our method DRPMD and the benchmark method robust value iteration,
3. C++ codes for evaluating the time of two different inner tolerance selections and,
4. Python codes for plotting the error decreasing performence.

The folder Inventory Management Problem includes:

Python codes for a simple function collections,
Python codes for generating a inventory problem with parameterized transition and applying DRPG to solve it and,
Two CSV files which specify the cost and the empirical transition of the inventory management problem.
