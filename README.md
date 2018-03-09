# NeuroMTL iSEG
NeuroMTL iSEG method implimentation in torch, short description of the method is here:  https://doi.org/10.1101/278465

## Dependencies:
`minc2-simple display xlua cudnn optim paths`

## iSEG challenge results:
See at http://iseg2017.web.unc.edu/rules/results/


## Running:

* in `data` directory prepare testing and training data
* run `iSEG_train.sh` to train deep-net
* run `iSEG_sanity_check.sh` to validate results
* run `iSEG_segment.sh` to prepare final results
* in `data` directory run `postprocess.sh` to convert from minc to submission data format (Analyze)


## Copyright
GNU-v3, see LICENSE file

