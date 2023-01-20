.ONESHELL:
SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


all: env install install_model test

env: 
	@conda create -n cde python=3.7
	conda init bash

install:
	snap install protobuf --classic
	$(CONDA_ACTIVATE) cde
	@pip install -r requirements.txt
	@git clone https://github.com/tensorflow/models.git
	cd models/research
	@protoc object_detection/protos/*.proto --python_out=.
	cp object_detection/packages/tf2/setup.py .
	@python -m pip install .

install_model:
	gdown 1e0UTKwhgJN9DuD2qYsLcWcKd6WomvRkl
	unzip folder.zip
	rm -rf folder.zip
	cp -r folder/inference_graph models/research/object_detection
	mkdir models/research/object_detection/training
	cp folder/labelmap.pbtxt models/research/object_detection/training

test:
	@conda activate cde
	@python models/research/object_detection/builders/model_builder_tf2_test.py

clean:
	rm -rf __pycache__
	rm -f *.pyc
	rm -f *.log