## Project Description
In this project description, we will cover:
* The Project Overview
* Business Understanding: 
  - Explaining stakeholders, stakeholder audience and business questions
* Data Analysis:
  - Source of data and deliverables
  - Data Description 
  - Data Understanding
    **Modeling**
    **Model Evaluation** 
* Recommendations 
* Conclusion:
  - Summary and three relevant findings.

## Project Overview

This project aims to predict customer churn in an e-commerce platform using machine learning to design targeted strategies to boost customer satisfaction and to reduce churn for the e-commerce platform. 
The model will analyse customer interactions, purchase history, and behavioural patterns to predict the likelihood of customers discontinuing their engagement with the ecommerce platform. 
The model should further assist in implementation of retention strategies to improve customer satisfaction.

## Business Problem

Customer churn is a major problem for e-commerce platforms as it is important to retain customers in a growing platform. Losing customers not only hurts immediate income, but also damages brand reputation, growth and long-term loyalty. A data-driven approach is required to to identify early signs of churn.

## Business Understanding
* Target stakeholders: E-commerce Platform Management (CEO, marketing director)
- Defines business goals, provides data access, and evaluates the impact of churn prediction on key metrics (revenue, customer lifetime value).

* Stakeholder audience:
- Customers: Benefit from better product recommendations, personalized offers, and improved platform experience based on the churn prediction model.

#### Business questions.
1. Which customer features are most indicative of a potential churn?  
2. Which consumer behaviour are most indicative of a potential churn?
3. How impactful is the products returned to the customer churn or retention?

## Data Analysis
#### The Source of Data

* This project uses the [Ecommerce](ecommerce_customer_data_large.csv) in the data folder in this project's GitHub repository. 
* Access the information on the dataset from https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis 

#### Deliverables
There are three deliverables for this project:

1. A **non-technical presentation** 
2. A **Jupyter Notebook**
3. A **GitHub repository** - Access: [https://github.com/otienobaker1](https://github.com/otienobaker1/ECOMMERCE-CHURN-RATE-PREDICTOR)

#### Data Description 

The data we have used in this dataset with their respective descriptions include:
* `Customer ID` - A unique identifier for each customer.
* `Customer Name` - The name of the customer.
* `Customer Age`- The age of the customer.
* `Gender` - The gender of the customer.
* `Purchase Date` - The date of each purchase made by the customer.
* `Product Category` - The category or type of the purchased product.
* `Product Price` - The price of the purchased product.
* `Quantity` - The quantity of the product purchased.
* `Total Purchase Amount` - The total amount spent by the customer in each transaction.
* `Payment Method` - The method of payment used by the customer (e.g., credit card, PayPal).
* `Returns` - Whether the customer returned any products from the order (binary: 0 for no return, 1 for return).
* `Churn` - A binary column indicating whether the customer has churned or not churned meaning, left the platform or not, respectively (0 for retained, 1 for churned).

#### Data Understanding 

- We begin by importing libraries for: 
> Data analysis and preprocessing.
> Data visualization.
> Applying logistic regression models.
> Model evaluation. 
> Sampling.

- We then load the e-commerce file and perform:
**Data Preprocessing**

- We check for null values by checking the percentage of null values in each column to provide us with valuable insights into data quality and potential issues for each column. We find that only one column, `Returns`, has 18.95% null values.
- We utilize the use of a "SimpleImputer" from scikit-learn to handle `Returns` missing values by setting the strategy parameter to "most_frequent". This indicates that the imputer will replace missing values with the most frequent value occurring in the column.

***Feature Engineering and clean up***

* A class called "CustomerDataPreprocessor" is defined which is responsible for cleaning and transforming customer data related to purchases. 
- Define an attribute of the number of times a customer has shopped: 
> A new feature called 'Timeshopped'.

- handle_purchase_date is defined:
> Transform the 'Purchase Date' column into datetime format.
> Sort the data by customer ID and purchase date in ascending order.
> Creating two new columns, `FirstPurchaseDate` and `LastPurchaseDate`, and calculating the time difference between the first and last purchase by subtracting the two columns.
> Drops temporary `FirstPurchaseDate` and `LastPurchaseDate` columns.

- split_and_categorize_time:
> extracting features from the 'Purchase Date' column by creating separate columns for `Day`, `Month`,`Time`(in 24hour clock) and `Year`.
> Creates `Time_s`(Time Shopped) categorizing purchase time as "Day" or "Night".
> Dropping the `Day` and `Time` columns.

- Define the age threshold for labeling:
> Removing the `Customer Age` column because there is redundancy with age group.
> Creates a new `AgeGroup` column, is converted into a binary "Young" or "Old" category based on a 40 year threshold.
> Removing the original `Age` column.

- Remove duplicate fields of orders:
> Sort data by purchase date in descending order.
> Removes duplicate customer entries based on`Customer ID`.

- Label encode the returns field:
> Encodes the `Returns` column as "returned" as 1 or "not returned"as 0.
> A new feature 'Returned_prod' is created that combines product category and return status.

- convert_days_to_months:
> Convert the `TimeDifference` from days to months by dividing by 30.44.
> Replace any 0 values with 1.

- total_purchase:
> Converts `Total Purchase Amount` to numerical format.
> Creates a new column `Total Amount Spent` for each customer by adding their total purchases using groupby and transform.

- For further analysis, an instance of the CustomerDataPreprocessor is created, various cleaning and transformation steps to your data is applied and lastly, the preprocessed DataFrame is stored back into the original variable. 


**Visualization**
***Churn Rate Analysis***

* ChurnVisualizer Class is created:
- A class is defined that helps analyze and visualize customer churn data. 
- Various methods are used to create different plots:

> A pie chart is used to represent the percentage of female(0) and male(1) that have churned.

![Churnrate for Male and Female](visualizations/Gender-PieChart)      

> visualize_total_sales_and_churn_rate represents the total sales and churn rate per year.
> visualize_total_amount_spent_distribution compares the spending distributions between churned and non-churned customers.
![Spent Amount By Customers](visualizations/Histogram-Distributions.png)

> plot_churn_count_by_category identifies product categories with higher churn rates by isolating churned customers then counts how many churned in each product category, showing which categories have higher churn rates.
![Product Category](visualizations/ProductCategory_Churn.png)

> plot_churn_rate_by_return_status analyzes churn rate differences on whether customers returned products.
![Returned Products](visualizations/ReturnProduct_Churn.png)

> plot_churn_rate_over_time_shopped analyzes how churn rate relates to the number of times customers have shopped.
![Time Shopped](visualizations/TimeShopped_Churn.png)

> plot_churn_rate_over_years tracks overall churn rate changes across all the years.
![Over the Years](visualizations/Years_Churn.png)

> plot_churn_rate_over_months visualizes the monthly churn rate variations in each year
which is calculated by dividing the number of churned customers by the total customers expressing it as a percentage.
- Months in a year with significantly higher churn require targeted retention efforts.

> plot_churn_rate_over_time_spent shows how churn rate relates to the amount of time customers spend on the platform.
![Time Spent](visualizations/TimeSpent_Churn.png)

> plot_churn_rate_by_payment_method compares churn rates through different payment methods used by customers.
![Payment Method](visualizations/PaymentMethod_Churn.png)

Summary of the Churn Rate and Sales through the years:
	Year	Total Sales	Churn Rate
0	2020	4948906	0.205339
1	2021	32903519	0.200106
2	2022	177182499	0.196968
3	2023	466311375	0.201139


***Encoding Categorical Features***

* Label encoding categorical columns:
- A LabelEncoder is used to convert the string values, in those columns chosen, to numerical labels to handle categorical data.
- Dropping irrelevant features:
`'Customer ID', 'Purchase Date','Product Price','Quantity','Total Purchase Amount','Customer Name'`

***Modeling Baseline Model (Decision Tree)***

- Separating the target variable, Churn as y, and the rest of the features, making it X. 
- Splitting fthis data into sets for training and testing the model with 20% for testing and 80% for training.
- Visualizing the churned and non-churned customers in both training and testing sets to ensure balanced representation.
- By running check_churn_distribution(y_train), check_churn_distribution(y_test), two bar plots will be displayed. one for the training set and another for the testing set. 
![Churn Distribution](visualizations/churn-distr.png)
![Churn Distribution](visualizations/churn-distr2.png)

- Perform oversampling, using RandomOverSampling,on the dependent variable to create a more balanced churn distribution. This is used to balance the distribution of churned and non-churned customers in your training data, helps reduce potential bias and aims to improve model's performance. 
![Churn Distribution](visualizations/Distr_Churn.png)


- Evaluation of the Decision Tree model: 
> The model achieves an accuracy of 67.572% on the testing data which is a decent accuracy.
> The training data has a 100% accuracy, suggesting potential overfitting.

- Evaluation of the Confusion Matrix:
> Has true Positives of 6322 which means the model identified 6322 customers who churned.
> False Positives of 1605, the model falsely predicted churn for 1605 customers who didn't churn .
> False Negatives of 1616, where the model missed/did not predict the churn for 1616 customers. 
> True Negatives of 390, showing correct prediction for 390 customers who didn't churn.

- This model performs perfectly on training data but poorly on new data, 0.67 vs. 1 accuracy.
- It misses many actual churns, False Negatives, due to prioritizing the non-churns.

***Complex Model (Random Forest)***

- The model achieved an accuracy of 76.53% on the test set.
- The confusion matrix shows far more non-churn customers than churned customers.
- Based on the performance:
> Non-churn customers (Class 0) has a high precision (80%) and recall (94%). Therefore, the F1-score is 0.87

> Churned customers (Class 1) has a very low precision (21%) and recall (6%) which is missing many actual churns, providing an F1-score of s 0.09

***Model with tuned hyperparameters***

- We perform hyperparameter tuning on the Decision Tree model using GridSearchCV. The grid search identifies a model with no limit on tree depth ('max_depth': None) and minimum number of samples('min_samples_split': 50) required to split an internal node as the best-performing one based on accuracy.
- This model is then used to make predictions on the test data.

- Accuracy score:
> The tuned Decision Tree model has an accuracy of 58.64% on the test set.
> The confusion matrix indicates a higher number of correct predictions for non-churned customers ,5155, compared to churned customers, 670.

- Feature Importance Analysis:
> The blocks of code indicate feature importance scores from the best model.
> A DataFrame to organize features and their importance is created and sorts it in descending order of importance.

![Feature Importance](visualizations/FeatureImportance-RandomForest.png)

> The model indicates that "total amount spent" is the most important feature, followed by "time difference", "month", "timeshopped", "payment method" etc.
> This suggests that these features play a significant role in predicting customer churn.


***Creating an XGBoost Model***

- An XGBoost model is created for churn prediction: 
> Random Search explores various hyperparameter combinations using RandomizedSearchCV and performs hyperparameter tuning while considering the F1-score.
> Addressing class imbalance by using oversampled training data.
> Evaluating the model performance with confusion matrix, classification report and accuracy.

- Outcome:
> Has a moderate accuracy of 62.7%.
> Non-churned customers have a precision of 80% and recall of 71% 
> Churned customers has a precision of 20% and recall 29%.

***Comparing the Models***

> Finally we compare the performance of the three models, Decision Tree, Random Forest and XGBoost: 
- We find that Random Forest outperforms other models with the highest accuracy, 95%, precision of 0.925 and F1-score of 0.954.
- Tuned Decision Tree performs well having a  moderate accuracy, 87.3% and  a high recall,0.988, showing good analysis of churned customers.
- XGBoost underperforms with the Lowest accuracy, 67.9% and all other metrics

## Recommendations

1. Collecting feedback and personal preferences from males and older customers to reduce churn rates.

2. Provide incentives for new and big spenders  

3. Provide a feedback survey for the products that have been returned.


## Conclusion 

Random Forest excelled overall while Tuned Decision Tree prioritized identifying churned customers. Feature engineering tailors the approach to business needs with continuous monitoring and updates providing an informative churn prediction system. 

#### Key Findings
1. "Total amount spent" emerged as the most important feature for churn prediction showing its relevance for future analysis.
2. The model with the best accuracy is the Random Forest, 95.2869%, indicating great performance.
3. Cross-validation using multiple metrics (accuracy, precision, recall, F1-score) provides a more comprehensive picture than just the accuracy.

#### Summary 

Data-driven insights were used for implementation strategies and the way forward to keep customers satisfied. Several models were evaluated, with Random Forest showing the best overall performance and other metrics.However, we are not provided with complete data for the year 2023 therefore, we have limited resources.
