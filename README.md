# AWS-Machine-Learning-Engineer-Capstone
AWS Machine Learning Enginner Nanodegree

Customer churn, also referred to as customer attrition, poses a significant challenge for businesses. It arises when customers discontinue the use of a company's products or services, and high churn rates can have adverse effects on a company's revenue and profitability [1] .
To tackle this issue, machine learning algorithms can be leveraged to identify the factors that contribute to churn. Churn models are designed to identify early warning signs and recognize customers who are more likely to voluntarily leave [2] and [3]. As part of this project, I will be delving into three different algorithms: logistic regression, decision tree, and random forest. Through the application of these three powerful tools, I aim to develop a highly accurate classifier that can predict which customers are likely to churn and which are not.


## Package Version

```
polars                          0.17.0
folium                          0.14.0
imblearn                        0.0
sagemaker                       2.144.0
kaggle                          1.5.13
```

## Notebook enviroment

<img src="https://user-images.githubusercontent.com/94936606/230729952-7bd1afa7-f09a-4fce-bb36-521a10c327f8.png" width=50% height=50%>

## Loading kaggle dataset

### Setup kaggle API

1 - Create a new account in kaggle.com if you do not have one 

![kaggle1](https://user-images.githubusercontent.com/94936606/230730795-348c16ca-2298-4f9c-add7-52bceb578b39.PNG)

2 - Access account settings

![kaggle2](https://user-images.githubusercontent.com/94936606/230730837-2b691a9a-a232-4609-af5a-640ed1902b31.PNG)

3 - Click on 'Create New API Token'

![kaggle3](https://user-images.githubusercontent.com/94936606/230730871-298fb652-44c6-4f7c-aea4-f3dedb50e64d.PNG)

4- Upload the `kaggle.json` to SageMaker

![sagemaker1](https://user-images.githubusercontent.com/94936606/230730938-dcba94d6-7f1e-47bb-b495-dec20037a832.PNG)

4- Run the `setup_kaggle_api.sh` script on terminal or run in a notebook cell `!bash setup_kaggle_api.sh`

5- Run the `load_dataset.sh` script on terminal or run in a notebook cell `!bash load_dataset.sh`


## References

[SageMaker](https://aws.amazon.com/sagemaker/)

[SageMaker Hyperparameter Tuning](https://sagemaker.readthedocs.io/en/v1.44.4/tuner.html)

[SagaMaker Training Jobs](https://sagemaker.readthedocs.io/en/v2.145.0/overview.html#prepare-a-training-script)

[SageMaker Batch Transfom](https://sagemaker.readthedocs.io/en/v2.145.0/overview.html#sagemaker-batch-transform)

[SageMaker Inference Pipelines](https://sagemaker.readthedocs.io/en/v2.145.0/overview.html#inference-pipelines)

[AWS S3](https://aws.amazon.com/s3/)

# Definition

## Project Overview


### Datasets description:




## Problem Statement


## Metrics


# Analysis




## Exploratory Visualization




## Algorithms and Techniques




## Benchmark 




# Methodology

## Data Preprocessing


## Implementation



## Results

Model Evaluation and Validation


# Conclusion

Reflection

Improvement
