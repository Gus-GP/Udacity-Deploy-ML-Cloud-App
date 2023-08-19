# Model Card

## Model Details

Gustavo Grinsteins created the model. It uses a logistic regression algorithm using the default hyperparameters in scikit-learn 1.3.0. 

## Intended Use

The model task is to classify whether income exceeds $50K/yr based on 1996 census data. The development and use of this model was intended as a learning activity.

## Training Data

The data comes from the [Unoversity of California Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/20/census+income). For data cleaning, white spaces and tuples with null attributes were removed. For data pre-processing, a one hot encoder was used for features and a label binarizer was used for labels.

## Evaluation Data

Twenty percent of the 48,842 dataset tuples were used for model evaluation. 

## Metrics

[Precision](https://en.wikipedia.org/wiki/Precision_and_recall), [Recall](https://en.wikipedia.org/wiki/Precision_and_recall), and [Fbeta](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#:~:text=The%20F%2Dbeta%20score%20is,while%20beta%20%3C%201%20favors%20precision.) were used as performance metrics.

Global Output Metrics:
* Precision: 0.75
* Recall: 0.28
* Fbeta: 0.41

Metrics by categorical feature slices: 
* Please check out the file "slice_output.txt"

## Ethical Considerations

This dataset deals with sensitive information about race and ethnicity. Since 1996, The United States Census Bureau has implemented improvements within the census surveys to better capture race and ethnicity data. Given that this is an older dataset, it is plausible that this data contains bias introduced by older surveying rules, questions, and procedures. 

This prediction tool should NOT be used to make predictions solely based on someones race or ethnicity.

## Caveats and Recommendations

The main goal for this project is not to produce the best prediction model but gain practice using MLOps tools. Therefore, the data cleaning, data processing, and model optimization procedures are minimal. The data is messy and requires a more in depth cleaning approach in order to produce better results. Different models should be tested for prediction (e.g. Decision Trees, Random Forests) and compared to select a model that performs best with this data set.