#!/bin/bash

model_name=searchable_resnet

widths=(0.25 0.5 1.75 1.0 1.25 1.5 1.75 2.0)
depths=(9 18 34 50 56)

for width in ${widths[@]}
do 
	for depth in ${depths[@]}
	do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth""
		  python3 evaluator.py --attack reference --plot True --model $model_name --width $width --depth $depth
		  echo "-----------------------------------------------------------------------------------------------"
		  
	done
done

