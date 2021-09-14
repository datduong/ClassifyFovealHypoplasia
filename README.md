## Foveal Hypoplasia

**Paper coming soon ...**

**Data will be uploaded soon ...** [Structure of dataset folder](https://asciinema.org/a/435740)

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

### Scripts

Scripts to train models are [here](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts); for example, you can train [images to predict foveal hypoplasia scores](https://github.com/datduong/ClassifyFovealHypoplasia/tree/master/Experiment/Scripts/Img_FH_score).
### Example of training. 

Because of our small data size, training finishes rather quickly and needs low GPU memory. 




