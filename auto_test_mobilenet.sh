#!/bin/bash

# Default for AlexNet is (depth=1.2 | width=1.0)

model_name=depth_cnn

widths=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)
#depths=(1.0 1.2 1.4 1.6 1.8 2.0)
depth=1

for width in ${widths[@]}
do 
	# for depth in ${depths[@]}
	# do

		  echo "Test for the model: "$model_name" | width ratio: "$width" | depth: "$depth"" ;
		  python3 trainer.py --model $model_name --width $width --depth $depth ; 
		  python3 evaluator.py --attack shadow --plot True --model $model_name --width $width --depth-multi $depth ;
		  echo "-----------------------------------------------------------------------------------------------"
		  
	# done
done
