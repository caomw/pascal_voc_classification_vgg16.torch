# pascal_voc_classification_vgg16.torch
A repro of Maxime Oquab's FullImages setup from the CVPR'14 paper

# Steps
1. Populate *data* (go through README.txt inside)
2. Preprocess the VOC dataset:
  ```th preprocess_voc.lua```
3. Run repro:
  ```th repro.lua```
4. mAP is printed occasionally on the screen
