#!/bin/bash

python3 run_inference.py --object-weights "detection/object.pt" \
	--char-weights "detection/char.pt" \
	--out-dir "out" \
	--dataset-dir "/ws/cortex/anpr/data/dataset-5/test/clean_images" \
	--object-imgsz 1280 \
	--char-imgsz 256 \
	--device cpu
