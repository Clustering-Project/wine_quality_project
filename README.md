# wine_quality_project
First team project!

## Goal

* Find specific features that have similarities and group those features and label them using a clustering model.
* The purpose of the regression model is to predict quality based off features including those that the cluster model generated.

## Initial hypotheses

H0: There is no significant association between the quality of the wine and the Alcohol/Density Cluster.

Ha: There is a significant association between the quality of the wine and the Alcohol/Density Cluster.

## Data dictionary

| Feature               | Description                                                            |
| --------------------- | -----------------------------------------------------------------------|
|fixed_acidity          | total amount of acids                                                  |
|volatile_acidity       | measure of volatile acids                                              |
|citric_acid            | measure of natural acid                                                |
|residual_sugar         | amount of sugar remaining in the wine after fermentation               |
|chlorides              | chloride concentration level                                           |
|total_sulfur_dioxide   | total amount of sulfur dioxide                                         |
|density                | mass of the wine per unit volume                                       |
|pH                     | measures the acidity or alkalinity of the wine on a scale from 0 to 14 |
|sulphates              | measure of compounds added to wine as a preservative.                  |
|alcohol                | alcohol by volume                                                      |
|quality                | rating of the wine's overall quality                                   |
|type                   | red or white wine                                                     |


## Planning:
Questions to ask about the data set based off of what I want my model to predict: 
- Do any features have a correlation with quality?. 
- Do cluster enhance performamce?
- What model is best for predicting quality?

- Final report should be in .ipynb, Modules should be in .py.
- Audience will be data scientist team.
- Determine correlation between features and target variable.
- Develop my null hypothsisis and alternative hypothesis.
- Explore data using visuals and statistical tests.
- Create cluster model.
- Create regression model
  
## Acquisition:
- Registered an account at data.world and downloaded the wine-quality dataset into a csv.
- Used pandas to read in the csv file onto a jupyter notebook.

## Preparation:
- Replaced blank spaces with underscores for columns.

## Exploration & pre-processing:
- Made visuals and used stats to understand which features had a significant correlation, relationship with the target variable.
- Used top 2 features (alcohol, density) to create cluster

## Modeling:
- Created a regression model to predict quality of wine.
- The random forest regressor model on unscaled data without using clustered features has the best performance in predicting wine quality.  

Baseline = .88

Test RMSE = .60

## Delivery:
- Deployed my model and a created a reproducable report
- Made recommendations
- Created canva slides for storytelling

## Key findings, recommendations, and takeaways
- Decrease in density improves quality
- Increase in alcohol percentage impoves quality
- Decrease in density improves quality
- Clusters using top 2 key features do not have significant impact on regression performance.

Recommend: 
- Tune hyperparameters
- Feature engineering
- More clusters

## Instructions or an explanation of how someone else can reproduce project and findings

Enviroment setup: 
- Install Conda, Python, VS Code or Jupyter Notebook
- Clone this repo remotely through your terminal (CLI)
