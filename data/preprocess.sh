#! /bin/bash
################################ MNI Header ##################################-
## @NAME       :  preprocess.sh
## @DESCRIPTION:  convert iSEG labels and crop all files
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

mkdir -p iSEG_training iSEG_testing



# convert lables
for i in training/subject-*-label.mnc;do

s=$(basename $i|cut -d - -f 1,2)

minccalc -byte -express \
     "A[0]>200?3:A[0]>100?2:A[0]>9?1:0" training/${s}-label.mnc training/${s}-cls.mnc



done

# generate majority label and crop it ( based on visual inspection)
multiple_volume_similarity training/subject-*-cls.mnc --maj maj.mnc

# crop
mincreshape -dimrange zspace=85,127 maj.mnc maj_z.mnc


# crop all training files and create flipped version

param2xfm -scale -1 1 1 flip.xfm

rm -f iSEG_sym_train_list.lst

for i in training/subject-*-label.mnc;do

s=$(basename $i|cut -d - -f 1,2)

itk_resample --byte --label --like maj_z.mnc \
    training/${s}-cls.mnc iSEG_training/${s}-cls.mnc

mincresample -nearest -like maj_z.mnc \
     training/${s}-T1.mnc iSEG_training/${s}-T1.mnc

mincresample -nearest -like maj_z.mnc \
     training/${s}-T2.mnc iSEG_training/${s}-T2.mnc

echo iSEG_training/${s}-T1.mnc,iSEG_training/${s}-T2.mnc,iSEG_training/${s}-cls.mnc >>iSEG_sym_train_list.lst

# flip around y-z plane
itk_resample --byte --label --like maj_z.mnc \
    training/${s}-cls.mnc iSEG_training/${s}-cls_f.mnc \
    --transform flip.xfm

mincresample -nearest -like maj_z.mnc \
     training/${s}-T1.mnc iSEG_training/${s}-T1_f.mnc \
     -transform flip.xfm

mincresample -nearest -like maj_z.mnc \
     training/${s}-T2.mnc iSEG_training/${s}-T2_f.mnc \
     -transform flip.xfm 

echo iSEG_training/${s}-T1_f.mnc,iSEG_training/${s}-T2_f.mnc,iSEG_training/${s}-cls_f.mnc >>iSEG_sym_train_list.lst 
done


# crop all testing files too
for i in testing/subject-*-T1.mnc;do

s=$(basename $i|cut -d - -f 1,2)

mincresample -nearest -like maj_z.mnc \
          testing/${s}-T1.mnc iSEG_testing/${s}-T1.mnc          
          
mincresample -nearest -like maj_z.mnc \
          testing/${s}-T2.mnc iSEG_testing/${s}-T2.mnc                
done


