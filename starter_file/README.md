# Predicting Credit Card Churn

This is the Capstone project for the Machine Learning Engineer for Azure Nanodegree by Udacity. This project will use both AutoML and HyperDrive to train models in order to try to predict credit card churn at a consumer credit card bank.

## Project Set Up and Installation
To get the dataset for this project from [Kaggle](https://www.kaggle.com/), you must register for a free account. The dataset can then be downloaded either directly from the website or with through their API, as I have done for this project. Downloading the dataset through the API requires you to first install the opendatasets package, which can be done with:

    pip install opendatasets

Once installed and imported, the dataset can be downloaded with:
    
    import opendatasets
    opendatasets.download('https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn')

## Dataset

### Overview
In this project, I will be examining what factors contribute most to customer attrition at a consumer credit card bank based on a [dataset submitted to Kaggle](https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn). This dataset mainly contains demographic information  such as customer age, gender, number of dependents, education level, and marital status as well as information on their income category, which credit card they have, and how long the account has been open for. The dataset also contains two Naive Bayes Classifier fields which the uploader suggested to be deleted and not be considered for analysis.

### Task
The goal for these models is to be able to predict credit card churn and identify which factors contribute most to customer attrition.

### Access
As previously noted, the data was downloaded from Kaggle into my Azure ML Studio Workspace. I was then able to read the data to a pandas dataframe, drop the unneeded `CLIENTNUM`, `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1`, and `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2` fields, and register the dataframe as a TabularDataset.

## Automated ML
For the AutoML configurations, I chose to set the `experiment_timeout_minutes` to 30 in order to ensure that the model would finish training in a timely manner. I used 4 for the `max_concurrent_iterations` as this was the same number of nodes that I provisioned for my Compute Cluster. The `primary_metric` I chose to use was AUC_weighted in order to account for the imbalanced classes in this dataset as only 16% of the records in the dataset are `Attrited Customers` compared to 84% being `Existing Customers`. Lastly, I enabled `enable_early_stopping` in order to save compute resources if the models are no longer improving.

### Results
The various types of models trained with AutoML included Logistic Regressions, XGBoostClassifiers, and RandomForests. While the scores were lower than the best model, it can be noted that all of them had performed reasonably well, only going as low as 0.88905727 for the AUC_weighted score, as shown in the screenshot of the RunDetails and the plot of the AUC_weighted scores below:
![automl-run-details.png](/.github/images/automl-run-details.png)
![automl-auc-weighted-plot.png](/.github/images/automl-auc-weighted-plot.png)

The best model from this run was the VotingEnsemble (PreFittedSoftVotingClassifier) which had an AUC_weighted score of 0.99301462. The RunID and properties of the model can be found in the image below:
![automl-best-model.png](/.github/images/automl-best-model.png)

The VotingEnsemble model is essentially a combination of several other models that were trained in other iterations of this run, such as those that I listed previously. The specific models that were used and their hyperparameters can be found in the above image.

To answer the question of what features contribute most to customer attribution, the VotingEnsemble model found that `Total_Trans_Ct` was the most important feature, followed by `Total_Trans_Amt`, then `Total_Revolving_Bal`, as seen in the chart below:
![automl-feature-importance.png](/.github/images/automl-feature-importance.png)

The graphs below show the individual feature importance for the top three features previously noted. Two fairly clear clusters are clear in each of the plots, where the left cluster is for the customers that are unlikely to be attrited while the right cluster is customers that are likely to be attrited. We can also see that in the plots for `Total_Trans_Ct` and `Total_Trans_Amt`, customers that have higher values for both are less likely to be attrited than those who have not had more than ~100 transactions or spent more than ~$10k total. There is not as clear of a pattern for the `Total_Revolving_Bal` feature, which explains why this feature was not ranked as highly as the other two.
![feature-importance-1.png](/.github/images/feature-importance-1.png)
![feature-importance-2.png](/.github/images/feature-importance-2.png)
![feature-importance-.png](/.github/images/feature-importance-3.png)

## Hyperparameter Tuning
I opted to use a simple LogisticRegression model for this experiment as it fit the problem of binary classification well. The hyperparameters that were tuned for the model were `C` and `max_iter`, which are the regularization strength and the maximum number of iterations, respectively.

For the parameter sampling method, RandomParameterSampling was used in order to reduce compute resource usage that would have been used up by a Grid Search to only give a slightly improved model. The range of values that I set for `C` was between 0.001 and 100.0, while the values I selected for `max_iter` were specifically 10, 50, 100, 250, 500, and 1000. I decided to use uniform sampling for `C` and choice sampling for `max_iter` in order to ensure that I was getting a decent combination of hyperparameter pairs to lead to better optimization.

For the termination policy, I used the BanditPolicy in order to terminate any run that does not perform as well as the best performing run based on the slack factor and evaluation interval. This saves both time and resources as poorly performing models that will not perform better than the current best performing model will be terminated accordingly.

### Results
The best model I got from hyperparameter tuning with HyperDrive had a value of 17.02299055359339 for `C` and 250 for `max_iter`, which resulted in a `AUC_weighted` score of 0.9157469751872698, as seen in the images below:
![hyperdrive-run-details.png](/.github/images/hyperdrive-run-details.png)
![hyperdrive-best-model.png](/.github/images/hyperdrive-best-model.png)

While this was not as good as the performance that was seen with AutoML, the results were fairly acceptable. Over the 36 models trained, the lowest score was ~0.70161, which is significantly less performant than the worst AutoML model that was trained. It can also be noted that there is a greater range of values for `AUC_weighted` for this experiment than we previously saw with AutoML, as seen in the plot below (note that not all of the runs were shown here, so the lower AUC scores are not visible in this plot):
![hyperdrive-auc-weighted-plot.png](/.github/images/hyperdrive-auc-weighted-plot.png)

It is also not entirely clear how the regularization strength and number of iterations factor into the performance of the model. Going with a higher or lower value for either will not definitively give a better model, so it ultimately comes down to trial and error to optimize the hyperparameters as best as possible without using grid search. The parallel coordinates plot below shows the relationship between the hyperparameters and the primary metric:
![hyperdrive-auc-weighted-parallel-plot.png](/.github/images/hyperdrive-auc-weighted-parallel-plot.png)

In the future, I would improve the performance of this by running more models, expanding on the range of for each of the hyperparameter samples, and also consider trying other models as well. While 0.9157 is not a bad score, it can certainly be improved, and these adjustments could help achieve that improvement in performance.

## Model Deployment
I deployed the best model from the AutoML run, the VotingEnsemble model, as seen in the image below:
![automl-model-deployment.png](/.github/images/automl-model-deployment.png)

I was also able to test the endpoint successfully by sending a POST request to the `/score` endpoint for the model deployment with a sample input json within a list in a field with the key `data`. The `data` field takes a list because multiple documents can be scored with a single request. The model indicated that the customer represented by the sample request I submitted would have been classified as an `Existing Customer`. The request and response can be seen in the image below:
![hyperdrive-sample-request.png](/.github/images/hyperdrive-sample-request.png)

## Screen Recording
The screencast for this project can be found here: https://streamable.com/rkqbbi
