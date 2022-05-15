# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Victor Nascimento created the model. It is a Random Forest Classifier using sklearn with the following hyperparameters:
* n_estimators=500
* criterion='gini'
* max_depth=100
* max_features='auto'
* All other hyperparameters are the default sklearn hyperparameters.  

## Intended Use
This model should be used to predict if the person income if <=50K or >50K based on a set of variables like
age, workclass, education, marital-status, occupation and etc.

## Training Data
The Census Income Dataset was used to build the model. 
More information about the data can be found in : https://archive.ics.uci.edu/ml/datasets/census+income

The Dataset is available in : https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/data/census.csv

80% of the dataset was used to train the model and 20% was used for evaluation.

## Evaluation Data
20% of the Census Income Dataset was used for evaluation.

## Metrics
Precision, Recall and F1 Score were the metrics in various slices of the dataset.
All metrics can be found in model_performance.csv folder.

In the validation set the model achieved:
* 0.73 in Precision 
* 0.62 in Recall
* 0.67 in F1-Score

## Ethical Considerations
The model performance may vary in slices of the data with sensitive features such as sex, native country and race.
All predictions needs to be handled with caution.
Please, see the model metrics in each slice of dataset in model_performance.csv in the 'model' folder. 
## Caveats and Recommendations
It is possible to achieve a better performance model doing some feature Engineering. 
Also, some hyperparameter tuning can be done to find the best hyperparameters.
Advanced models such as ANN can also be tested.