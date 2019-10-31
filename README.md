# Variable selection with false discovery rate control in deep neural networks

The code in this repository implements the variable selection procedure (SurvNet) proposed in [https://arxiv.org/abs/1909.07561](https://arxiv.org/abs/1909.07561).

## Getting started

In order to run SurvNet, an operative version of Python and TensorFlow is needed. The code has been tested on Python 3.6.5 and TensorFlow 1.8. For installation of TensorFlow, please see [https://www.tensorflow.org/install](https://www.tensorflow.org/install). Other Python dependencies include *numpy* and *scikit-learn*.

## Running SurvNet

As a demo, *main.py* performs variable selection on dataset 1 in the paper (in one simulation). The output results include (1) the initial test loss, the initial test accuracy, the final test loss, and the final test accuracy, which are saved in *initnfinal.txt* (2) the statistics, such as the number of remaining variables and the estimated FDR, at each step of variable selection, which are saved in *step.txt* (3) the column indexes of the selected variables and the measurements of their importance, which are saves in *result.txt*. Note that *main.py* imports *initial.py* and *SurvNet.py*, and *SurvNet.py* imports *variable_selection.py*.

To run SurvNet on other datasets, simply substitute dataset 1 with your own data in *main.py*. The folder named *Data* contain the code used to generate/obtain all of the datasets in the paper.
