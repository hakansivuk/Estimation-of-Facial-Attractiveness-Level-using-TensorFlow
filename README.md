# Estimation-of-Facial-Attractiveness-Level-using-TensorFlow

In this repository, a CNN model is implemented with TensorFlow and trained on face images to predict facial attractiveness score. Different deep learning techniques are used and compared such as batch normalization, regularization. Different initialization methods are also tried.

[Report](https://github.com/hakansivuk/Estimation-of-Facial-Attractiveness-Level-using-Tensorflow/blob/master/Report.ipynb) is intended to serve as a discussion of techniques we tried and motivations behind them. First of all, as it is suggested in the homework file, effects of different methods are examined through several experiments. In Experiments section, comparison of different methods are explained with their motivations, theoretical background and results. After that, evaluation score of the best model according to validation score is stated (on test set). Finally, success and failure cases are shown and possible further improvements are discussed.

[Homework File](https://github.com/hakansivuk/Estimation-of-Facial-Attractiveness-Level-using-Tensorflow/blob/master/HW.ipynb)

Download the dataset into the 'datasets' directory. The file structure should look like this:

├── `datasets`  
│   ├── test  
│   ├── training  
│   └── validation  
├── `training`  
│   ├── data.py  
│   ├── \__ init__.py  
│   ├── loss.py  
│   └── model.py  
├── train.py  
├── .gitignore    
├── requirements.txt  
└── README.md  

Install the missing libraries from the `requirements.txt`


Run the code with:

`python train.py`

