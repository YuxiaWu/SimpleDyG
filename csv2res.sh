#!/bin/bash

dataset="UCI_13"
python_file="csv2resources.py"
for timestamp in {12..12}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="ML_10M_13"
python_file="csv2resources.py"
for timestamp in {12..12}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="dialog"
for timestamp in {15..15}
do
    python "$python_file" "$dataset" "$timestamp"
done

dataset="hepth" 
for timestamp in {11..11}
do
    python "$python_file" "$dataset" "$timestamp"
done
