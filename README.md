# SurgPilot

1. Data Sources

Training:
-CholecSeg8k: dataset with 13 classes --> we filtered it to obtain only the "blood" class
-HemoSet: dataset based on pig surgery
-Rabbani: dataset 

Testing: 
-SurgBlood-test

2. Data preprocessing and Harmonization

-Images and masks have been converted to .png format in order to have a uniform dataset.
-Masks binarization: different kind of masks we normalized them to obtain binarized ones (bg: 0, blood: 255) 
-Merging: we merged the 3 datasets in order to obtain a more complete and complex dataset for training models


