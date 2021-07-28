#!/usr/bin/env bash
# following https://aarongorka.com/blog/portable-virtualenv/
NAME=LFVcoffea
python -m venv --copies $NAME
source $NAME/bin/activate
pip install --upgrade pip
pip install coffea
sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*

tar -zcf ${NAME}.tar.gz $NAME
