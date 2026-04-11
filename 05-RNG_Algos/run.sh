#!/bin/bash

g++ -O3 -fopenmp main.cpp -o main.out && ./main.out
python3 benchmark.py
