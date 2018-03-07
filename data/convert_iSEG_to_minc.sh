#! /bin/bash
################################ MNI Header ##################################-
## @NAME       :  convert_iSEG_to_minc.sh
## @DESCRIPTION:  convert iSEG files to minc
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

#1. convert all files to minc

for j in {testing,training}/*-T{1,2}.hdr;do
 nii2mnc -float $j ${j/.hdr/.mnc}
done

for j in training/*-label.hdr;do
 nii2mnc -byte $j ${j/.hdr/.mnc}
done
