# Deploying a predictive model on the cloud

This project applies skills acquired in the Udacity course "Deploying a Machine Learning Model with FastAPI" (MLOps Engineer Nanodegree) to develop a classification model on publicly available Census Bureau data. Unit tests are created to monitor the model performance on various data slices. Application deployment is done using [Render](https://render.com/). Code validation, API tests, and deployments are incorporated into a Continuous Integration/Continuous Deployment (CI/CD) framework using GitHub Actions (CI) and Render's auto-deployments (CD).

Udacity's starter repo is [here](https://github.com/udacity/nd0821-c3-starter-code)

## Dataset

[census.csv](https://archive.ics.uci.edu/dataset/20/census+income)

## Python Environment Libraries

Project is implement in Python (3.8) using the following libraries:

* numpy
* pandas
* scikit-learn
* pytest
* requests
* fastapi
* uvicorn

For detailed package information please see **requirements.txt**

## Model Information

For the Logistic Regression model information please see **model_card.md**








