#!/usr/bin/env sh
HOME=`pwd`

# Chamfer Distance
# cd $HOME/extensions/chamfer_dist
# python setup.py install --user

# Compactness Constraint
cd $HOME/extensions/expansion_penalty
python setup.py install --user

