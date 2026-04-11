#!/bin/bash

g++ -O3 -fopenmp monte_carlo.cpp -o monte_carlo.out && ./monte_carlo.out
python3 benchmark.py
