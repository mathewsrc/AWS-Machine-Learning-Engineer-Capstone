#!/bin/bash	
kaggle datasets download -d gauravtopre/bank-customer-churn-dataset
unzip -o -p bank-customer-churn-dataset > bank_customer_churn.csv && mv bank_customer_churn.csv datasets/
	
