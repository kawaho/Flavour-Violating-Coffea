#!/usr/bin/env bash
NAME=coffeaenv_local
python -m venv --copies $NAME
source $NAME/bin/activate
pip install --upgrade pip
pip install coffea


