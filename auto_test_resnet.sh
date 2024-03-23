#!/bin/bash

# Default for ResNet is (depth=9 | width=1.0)

model_name=searchable_resnet

widths=(1.0 1.25 1.5 1.75 2.0 2.25 2.5)
widths=(1.0)
depths=(9 15 20 26 32 38 44)

for width in ${widths[@]}
do 
	for depth in ${depths[@]}
	do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth""
		  python3 evaluator.py --attack reference --plot True --model $model_name --width $width --depth $depth
		  echo "-----------------------------------------------------------------------------------------------"
		  
	done
done

