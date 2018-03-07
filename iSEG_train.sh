#! /bin/bash
################################ MNI Header ##################################-
## @NAME       :  iSEG_segment.sh
## @DESCRIPTION:  Apply pre-trained model to all iSEG files
## @COPYRIGHT  :
##               Copyright 2017 Vladimir Fonov, McConnell Brain Imaging Centre, 
##               Montreal Neurological Institute, McGill University.
##
##     This program is free software: you can redistribute it and/or modify
##     it under the terms of the GNU General Public License as published by
##     the Free Software Foundation, either version 3 of the License, or
##     (at your option) any later version.
## 
##     This program is distributed in the hope that it will be useful,
##     but WITHOUT ANY WARRANTY; without even the implied warranty of
##     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##     GNU General Public License for more details.
## 
##     You should have received a copy of the GNU General Public License
##     along with this program.  If not, see <http://www.gnu.org/licenses/>.
##             
########################################################################


# pre-train using ACE-IBIS dataset first
th src/cnn_random_fcn_ace_iseg.lua

# run cross-validation experiment using iSEG training data
th src/cnn_random_fcn_ace_iseg_s2.lua

# train the final model, using all available iSEG training data
th src/cnn_random_fcn_ace_iseg_s2_final.lua
