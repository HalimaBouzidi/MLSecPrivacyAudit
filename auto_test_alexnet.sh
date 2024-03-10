#!/bin/bash

model_name=searchable_alexnet

widths=(0.25 0.5 1.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0)
depths=(2 3 4 5 6 7 8 9 10)

for width in ${widths[@]}
do 
	for depth in ${depths[@]}
	do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth""
		  python3 evaluator.py --attack reference --plot True --model $model_name --width $width --depth $depth
		  echo "-----------------------------------------------------------------------------------------------"
		  
	done
done

exit 0