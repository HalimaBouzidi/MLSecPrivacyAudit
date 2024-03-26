#!/bin/bash

# Default for Transformer is (depth=10 | width=1.0)

model_name=asymetric_cnn

widths=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5)
#depths=(10 12 14 16 18 20)
depth=1

for width in ${widths[@]}
do 
	# for depth in ${depths[@]}
	# do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth"" ;
		  python3 trainer.py --model $model_name --width $width --depth $depth ; 
		  python3 evaluator.py --attack shadow --plot True --model $model_name --width $width --depth $depth ;
		  echo "-----------------------------------------------------------------------------------------------"
		  
	# done
done