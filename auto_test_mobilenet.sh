#!/bin/bash

model_name=searchable_mobilenet

widths=(0.25 0.5 1.75 1.0 1.25 1.5 1.75 2.0)
depths=(0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4)

for width in ${widths[@]}
do 
	for depth in ${depths[@]}
	do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth""
		  python3 evaluator.py --attack reference --plot True --model $model_name --width $width --depth-multi $depth
		  echo "-----------------------------------------------------------------------------------------------"
		  
	done
done
