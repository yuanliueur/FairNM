# FairNM - A Fairness Enhanced Name Matching System
This repository is a supplement to the thesis "Fuzzy Name Matching with Fairness Constraints". It provides the original code, data and fairness test bench.

## Introduction
FairNM is a token-based name matching algorithm that focuses on enhancing fairness. Extensive research into the characteristics of different names revealed that names of Korean and Chinese descent tend to be considerably shorter and exhibit less variation in their family names. Conventional name matching algorithms often underperform when applied to these languages. In order to address this issue, FairNM introduces two key components: the Siamese Neural Network-based Short Name Module and the Name Weighting module. These components work together to improve the accuracy and fairness of the name matching process for Korean and Chinese names.

## Folder Structure
This repo contains the following:
- data: comprising a name database labeled with their language codes;
- models: the required SNM models;
- results: log of the thesis results
- test_bench_tool: a jupyter notebook where one can test the accuracy and fairness of their own name matching algorithm


