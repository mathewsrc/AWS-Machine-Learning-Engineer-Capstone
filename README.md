# AWS-Machine-Learning-Engineer-Capstone

Project developed for AWS Machine Learning Engineer Scholarship offered by Udacity (2023)


## Churn Prediction

Customer churn, also referred to as customer attrition, poses a significant challenge for businesses. It arises when customers discontinue the use of a company's products or services, and high churn rates can have adverse effects on a company's revenue and profitability. To tackle this issue, machine learning algorithms can be leveraged to identify the factors that contribute to churn. Churn models are designed to identify early warning signs and recognize customers who are more likely to voluntarily leave. As part of this project, I will be delving into three different algorithms: logistic regression, decision tree, and random forest. Through the application of these three powerful tools, I aim to develop a highly accurate classifier that can predict which customers are likely to churn and which are not.

[!Read more]https://github.com/mathewsrc/AWS-Machine-Learning-Engineer-Capstone/blob/master/Churn%20Prediction%20-%20AWS%20Machine%20Learning%20Engineer%20Nanodegree.pdf

## Project overview

![capstone drawio (13)](https://user-images.githubusercontent.com/94936606/231004594-a071aca6-845f-41c4-8159-45b4a5127ab1.png)


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

N. Forhad, M. S. Hussain, and R. M. Rahman, "Churn analysis: Predicting churners," in Proceedings of the Ninth International Conference on Digital Information Management (ICDIM 2014), Phitsanulok, Thailand, 2014, pp. 237-241, doi: 10.1109/ICDIM.2014.6991433.

Qureshi, Saad, Ammar Rehman, Ali Qamar, Aatif Kamal, and Ahsan Rehman. "Telecommunication Subscribers' Churn Prediction Model Using Machine Learning." In Proceedings of the 8th International Conference on Digital Information Management (ICDIM 2013), 2013, pp. 133-137, doi: 10.1109/ICDIM.2013.6693977.

Ullah, Irfan, Basit Raza, Ahmad Malik, Muhammad Imran, Saif Islam, and Sung Won Kim. "A Churn Prediction Model Using Random Forest: Analysis of Machine Learning Techniques for Churn Prediction and Factor Identification in Telecom Sector." IEEE Access, vol. 7, pp. 104634-104647, 2019, doi: 10.1109/ACCESS.2019.2914999.

Khan, Muhammad, Johua Manoj, Anikate Singh, and Joshua Blumenstock. "Behavioral Modeling for Churn Prediction: Early Indicators and Accurate Predictors of Custom Defection and Loyalty." In Proceedings of the IEEE International Congress on Big Data (BigData Congress), 2015, pp. 7-14, doi: 10.1109/BigDataCongress.2015.107.

G. Menardi and N. Torelli, "Training and assessing classification rules with imbalanced data," Data Mining and Knowledge Discovery, vol. 28, no. 1, pp. 92-122, 2014, https://doi.org/10.1007/s10618-012-0295-5.

V. S. Spelmen and R. Porkodi, "A Review on Handling Imbalanced Data," in Proceedings of the 2018 International Conference on Current Trends towards Converging Technologies (ICCTCT), Coimbatore, India, 2018, pp. 1-11, doi: 10.1109/ICCTCT.2018.8551020.

N. V. Chawla, K. W. Bowyer, and W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.

R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model selection," in Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI'95), San Francisco, CA, USA, 1995, pp. 1137-1143.
