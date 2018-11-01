# CSI 5138 Homework Exercise 3

Use Vanilla RNN and LSTM to for text classification and sentiment analysis on a standard dataset of movie reviews. The dataset and its description can be found at: https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset . 

Specifically for several state dimensions [20, 50, 100, 200, 500], we look at tuning hyperparameters for the best test classification. Overall, given that the input dimension is 500 (we set the max length of sequences to 500 for the reviews based on histogram analysis), we find state dimensions 50 and 100 perform best. For Vanilla RNN we find these states perform best with learning rate 0.01-0.001, larger batch sizes 200-500, and larger dropout 0.5. For LSTM we find these states perform best with learning rate 0.01, batch size 200, and smaller dropout 0.1.

## Jupyter Notebook and Python

This homework makes use of [jupyter notebook](http://jupyter.org/) written in python.

## Files

`Homework3.ipynb`: Python code.

`Report.ipynb`: Reported findings of hyperparameter tuning.

`experiments`: Runs for each model (Vanilla RNN, LST) and for each state, giving the final testscore of accuracy and loss, as well as the history of the accuracy and loss after each epoch.

## GitHub

This homework can also be found on the following [github account](https://github.com/sofa13/csi5138_a3).
