# Imagematching_D2Net

## Overview

To address the problem that matching low-resolution images is affected by noise, which makes location estimation difficult,a method using deep learning feature matching is proposed.

The main idea and code of feature extracting in this repository are based on [D2-Net](https://dusmanu.com/publications/d2-net.html).



## Getting start:
Python 3.9+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to install the required packages:
### Dependencies

- PyTorch 
- OpenCV
- Numpy
- Matplotlib
- skimage
- pandas
- tqdm
- SciPy

## Training 
 After setting up the folder for the dataset, the training can be started right away:

 ```bash
python train.py
```

## Testing
After saving the trained model,the testing can be started right away:

```bash
python test.py
```
## Visualizing matching

```bash
python visualize.py
```


