#! /bin/sh
################################ MNI Header ##################################-
## @NAME       :  iSEG_sanity_check.sh
## @DESCRIPTION:  Apply pre-trained model to files that were used for training (for sanity check)
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


echo "SUBJ,CSF,GM,WM" > iSEG/check.csv

for i in iSEG/iSEG_training/subject-*-T1.mnc;do
  ref=${i/-T1/-cls}
  out=iSEG/check/$(basename $i |sed -e 's/-T1/-cls/')

  if [ ! -e $out ];then
  th manual_apply_fcn_iSEG.lua -t1 $i -t2 ${i/T1/T2} -gpu -model iseg_train/v2_final_1_fold_training.t7 -out $out
  fi

  echo $(basename $i -T1.mnc),$(volume_similarity  --kappa $ref $out |cut -d , -f 2|tr '\n' ','|cut -d , -f 1,2,3) >> iSEG/check.csv
done
