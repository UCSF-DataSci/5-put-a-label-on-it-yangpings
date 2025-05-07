# Assignment 5: Health Data Classification Results

This file contains your manual interpretations and analysis of the model results from the different parts of the assignment.

## Part 1: Logistic Regression on Imbalanced Data

### Interpretation of Results

In this section, provide your interpretation of the Logistic Regression model's performance on the imbalanced dataset. Consider:

- Which metric performed best and why? 
- Which metric performed worst and why? 
- How much did the class imbalance affect the results? 
- What does the confusion matrix tell you about the model's predictions? 

*Your analysis here...* <br>
Accuracy performed the best because we see that most of the data outcomes are very imbalanced, which means that any guess 
in favor of the majority would most likely be right. The worst performing metric is recall because most of the disease outcomes are negative, which made 
sensitivity hard to predict. It has affected greatly towards the evaluation, as seen from great differences between accuracy, auc and recall, f1. The confusion matrix
confirms our predictions, as we see an overwhelming number of negatives being evaluated and less positives. We can see that most of the error comes from predicted negative but appeared 
be positive, which shows the drawback of an imbalanced dataset.

## Part 2: Tree-Based Models with Time Series Features

### Comparison of Random Forest and XGBoost

In this section, compare the performance of the Random Forest and XGBoost models:

- Which model performed better according to AUC score?
- Why might one model outperform the other on this dataset?
- How did the addition of time-series features (rolling mean and standard deviation) affect model performance?

*Your analysis here...* <br>
We see that XGBoost performed better than random forest according to the auc score. XGBoost performs better because it handles class imbalances
slightly better than the random forest. The addition of time-series features allow reduced noise, sd and mean regularization, 
which prevents a local spike affecting the whole outcome, which means that it stabilizes results.
## Part 3: Logistic Regression with Balanced Data

### Improvement Analysis

In this section, analyze the improvements gained by addressing class imbalance:

- Which metrics showed the most significant improvement?
- Which metrics showed the least improvement?
- Why might some metrics improve more than others?
- What does this tell you about the importance of addressing class imbalance?

*Your analysis here...* <br>
We see that recall showed the most significant improvement where accuracy showed the least improvements. Some metrics improve more than
others because of the nature of imbalanced datasets we are working with in this assignment. Specific metrics that rely on the distribution of outcomes,
such as sensitivity and specificity, will have noticeable improvements compared to those which do not rely so much on outcome distributions. This tells us that 
accuracy and the auc do not determine how good a model is when being applied to one dataset and can be lethal without understanding the dataset before using the correct model

## Overall Conclusions

Summarize your key findings from all three parts of the assignment:

- What were the most important factors affecting model performance?
- Which techniques provided the most significant improvements?
- What would you recommend for future modeling of this dataset?

*Your conclusions here...* <br>
We see that in this case, feature outcomes is the most important driving factor of model performance. The most significant improvements are
done in part 3, which we see major upsides from recall, which are important in this dataset. For future modeling, I would strongly recommend
models that take least effect by imbalanced datasets and check recall before actually determining if a model is truly effective.