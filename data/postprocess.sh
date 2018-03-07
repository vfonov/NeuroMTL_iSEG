#! /bin/bash
################################ MNI Header ##################################-
## @NAME       :  preprocess.sh
## @DESCRIPTION:  prepare results of iSEG testing for submission
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

for i in testing/mnc/subject-*-T1.mnc;do

s=$(basename $i|cut -d - -f 1,2)

if [ ! -e testing/${s}-cls.mnc  ];then
itk_resample --byte --labels  iSEG_testing/${s}-cls.mnc --like $i --lut-string  "3 250;2 150;1 10"  testing/${s}-cls.mnc 
fi

itk_convert --inv-x  --byte testing/${s}-cls.mnc testing/${s}-label.hdr --clob --verbose

done
