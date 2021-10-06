## Predicting Foveal Hypoplasia scores ([and generating Foveal Hypoplasia images](https://github.com/datduong/stylegan2-ada-FovealHypoplasia))

**Paper coming soon ...**

**Data will be uploaded soon ...** [how your dataset folder should look](https://asciinema.org/a/435740).

**Pre-trained models will be uploaded soon ...**

**[What are Foveal Hypoplasia scores?](https://en.wikipedia.org/wiki/Macular_hypoplasia)**


## Instruction 

### Pre-train on free large dataset

Because our Foveal Hypoplasia dataset is small, we need to borrow the power of a larger auxilary dataset. You must download [this dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) which contains about 100k images ([their paper](https://pubmed.ncbi.nlm.nih.gov/29474911/)).

We trained our own EfficientNetB4 on these 100k images ([script](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/Experiment/Scripts/train_100k_oct.sh), [already trained model]()). Please change the directory path according your own machine. 


### Train on our foveal hypoplasia images

There are three kinds of predictions made from the images and metadata: 
1. [Foveal Hypoplasia scores](https://www.researchgate.net/figure/Schematic-demonstrating-the-Leicester-Grading-System-for-Foveal-Hypoplasia-showing_fig1_337025851) (continuous but can be treated as discrete from 1 to 4). [Train/test csv](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/TrainTestInputs/FH_OCTs_label_train_input.csv).
2. [LogMAR](https://en.wikipedia.org/wiki/LogMAR_chart) (continuous value). [Train/test csv](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/TrainTestInputs/FH_OCTs_label_train_input.csv) are same as above, we just use the column "logMAR" as our label.
3. Driving score (derived from LogMAR). [Train/test csv](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/TrainTestInputs/FH_OCTs_label_train_input_driving.csv).

These csv inputs contain the labels (e.g. LogMAR) and metadata (e.g. age, nystagmus status, ...). In the column names, known diagnoses begin with the letter "d", and known genes begin with the letter "g".

### Train classifier with real images

Scripts to train models are [here](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts); for example, you can train [images to predict foveal hypoplasia scores](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts/Img_FH_score). We used 5-fold cross-validation, so there are 5 models which will later be combined to make a final ensemble classifier. Run [ensemble with this code](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/ensemble.sh). 

### Train classifier with real and generated images

Please use [this GAN model](https://github.com/datduong/stylegan2-ada-FovealHypoplasia) to make fake images. Scripts to train a classifier using both real and fake images are [here](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts/Img_withfake_FH_score). Again, please change folder path according to your own machine. 

We trained the classifier using both real and fake images. Fake images were generated at the mix ratio of 90-to-10. For example, a fake image would have 90% characteristic of Foveal Hypoplasia score 1 and 10% characteristic of Foveal Hypoplasia score 2. We also tested the mix ratio 75-to-25, but did not see significant differences. Because of the mix ratio, we train using soft-labels (instead of a 1-hot encoding). 

### Example of training

Because of our small data size, training finishes rather quickly and needs low GPU memory. 

[![asciicast](https://asciinema.org/a/435777.svg)](https://asciinema.org/a/435777)


### Compute accuracy

Because Foveal Hypoplasia (FH) scores can be viewed as continuous values, we use these two metrics: correlation and linear regression R<sup>2</sup>. Predictions aligning well with true FH scores will have high correlation and high linear regression R<sup>2</sup>. High correlation implies that a more severe case will have a higher FH score than a mild case. High linear regression R<sup>2</sup> indicates that predictions are close to the true values. 

This [R script](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/GetFinalCorrR2.R) computes correlation and linear regression R<sup>2</sup>. Please accordingly [change your model names](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/GetFinalCorrR2.R#L8) and [folder paths](https://github.com/datduong/ClassifyFovealHypoplasia/blob/master/GetFinalCorrR2.R#L15). 

