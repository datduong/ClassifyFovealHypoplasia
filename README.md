## Foveal Hypoplasia

**Paper coming soon ...**

**Data will be uploaded soon ...** [structure of dataset folder](https://asciinema.org/a/435740).

**Pre-trained models will be uploaded soon ...**

**[What is Foveal Hypoplasia?](https://en.wikipedia.org/wiki/Macular_hypoplasia)**


## Instruction 

### Pre-train on free large dataset

Because our Foveal Hypoplasia dataset is small, we need to borrow the power of a larger auxilary dataset. You must download [this dataset](https://data.mendeley.com/datasets/rscbjbr9sj/3) which contains about 100k images ([their paper](https://pubmed.ncbi.nlm.nih.gov/29474911/)).

We trained our own EfficientNetB4 on these 100k images ([script](https://github.com/datduong/ClassifyFovealHypoplasia/blob/5cffa3ea8694d25b89bde3dab07b04895cf0da65/Experiment/Scripts/train_100k_oct.sh), [already trained model]()). Please change the directory path according your own machine. 


### Train on our foveal hypoplasia images

There are three kinds of predictions made from the images and metadata: 
1. Foveal Hypoplasia scores (continuous but can be treated as discrete from 1 to 4). [Train/test csv]().
2. LogMAR (continuous value). [Train/test csv]().
3. Driving score (derived from LogMAR). [Train/test csv]().

These csv inputs contain the labels (e.g. LogMAR) and metadata (e.g. age, nystagmus status, ...). In the column names, known diagnoses begin with the letter "d", and known genes begin with the letter "g".

### Scripts

Scripts to train models are [here](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts); for example, you can train [images to predict foveal hypoplasia scores](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts/Img_FH_score). We used 5-fold cross-validation, so there are 5 models which will later be combined to make a final ensemble classifier. 

### Train classifier with generated images. 

Please use [this GAN model]() to make fake images. Because fake images can only be trained from the real images, we will not be considering metadata. Scripts to train a classifier using both real and fake images are [here](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts/Img_withfake_FH_score). Again, please change folder path according to your own machine. 

We trained the classifier using both real images and fake images. The fake images were generated at the mix ratio of 90-to-10. For example, a fake image would have 90% characteristic of Foveal Hypoplasia score 1 and 10% characteristic of Foveal Hypoplasia score 2. 

### Example of training. 

Because of our small data size, training finishes rather quickly and needs low GPU memory. 




