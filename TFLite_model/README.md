## TFLite model

1. Compiler the TFLite model with Edge TPU compiler (version 15)

Edge TPU compiler version 15: https://github.com/google-coral/edgetpu/files/5546715/edgetpu-compiler_15.0_amd64.deb.tar.gz

edgetpu_compiler model_0_"model_name"_quant_"stage".tflite

2. Install the PyCoral API


Linux Dibian package:

sudo apt-get update

sudo apt-get install python3-pycoral

Mac:

https://coral.ai/software/#pycoral-api


3. Run inference with pycoral

Tutorial: https://coral.ai/docs/edgetpu/pipeline/#run-a-pipeline-with-python

Label:imagenet_labels.txt

Input image: parrot.jpg
