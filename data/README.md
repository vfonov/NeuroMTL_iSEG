# Training and testing data preparation

Currently ACE-IBIS is used for pre-training, followed by iSEG data for refinement.


## iSEG dataset

* Download iSEG training datasets from http://iseg2017.web.unc.edu/download/ 
( `iSeg-2017-Testing.zip` and `iSeg-2017-Training.zip` ) and unpack them into `testing` and `training`

* Convert to minc using `convert_iSEG_to_minc.sh`

* Generate cropped and augmented (flipped) datasets using `preprocess.sh`

* Train deep-net, generate output

* Convert minc output files to nifti using `postprocess.sh`


## ACE-IBIS dataset

Currently the data is not available for public release. 
The training data should be referenced in `ace_i_list.lst` for pre-training, in the format `T1w,T2w,CLS` 
