#!/bin/bash
mkdir /root/.jupyter
cp /workspace/jupyter/jupyter_notebook_config.py /root/.jupyter/
cp /workspace/jupyter/jupyter_notebook_config.json /root/.jupyter/
jupyter notebook --allow-root

