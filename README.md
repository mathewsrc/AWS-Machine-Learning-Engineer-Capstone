# AWS-Machine-Learning-Engineer-Capstone
AWS Machine Learning Enginner Nanodegree

```
sudo yum install -y zip

mkdir packages

cd packages 

python3 -m venv venv

source venv/bin/activate

mkdir python

cd python

pip install --no-deps scikit-learn -t .

pip install --no-deps imblearn -t .

pip install --no-deps numpy -t .

pip install --no-deps polars -t .

pip install --no-deps scipy -t .

rm -rf *dist-info

rm -rf scipy.libs/

du -sh

cd ..

zip -r train_test_split_lambda.zip python

aws s3 cp train_test_split_layer.zip s3://sagemaker-us-east-1-484401254725/scikit-churn-prediction/lambdalayers/

```


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
