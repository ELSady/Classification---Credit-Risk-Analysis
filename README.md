## Data Science Project: Credit Risk Analysis Overview
* Dataset consists of 32581 customer's credit history from The Bank on top of their current credit loan status.
* Visualize each customers's Loan Amount, Grade, Intent, Interest Rate and Income Perentage.
* Visualize and analyze those above features and how they fare based on customer's Age, Income yearly, Home Ownership and Loan Status.
* Visualize and analyze which customers segment who are likely to default of their loans.
* Build Supervised Machine learning Classification model, checking its feature importance and Shap value to determine which features contributes when it comes to predicting customer's default of their payment obligation.

![Are-you-eligible-for-the-reduced-home-loan-interest-rates-FB-1200x700-compressed](https://user-images.githubusercontent.com/96014656/179934775-73e29d3f-bd60-436c-afbf-44c797912c92.jpg) <br>

Financial institutions used credit risk analysis models to determine the probability of default of a potential borrower. The models provide information on the level of a borrower’s credit risk at any particular time. If the lender fails to detect the credit risk in advance, it exposes them to the risk of default and loss of funds. Lenders rely on the validation provided by credit risk analysis models to make key lending decisions on whether or not to extend credit to the borrower and the credit to be charged. With the continuous evolution of technology, banks are continually researching and developing effective ways of modeling credit risk. A growing number of financial institutions are investing in new technologies and human resources to make it possible to create credit risk models using machine learning languages.

### Business Questions:
* How much the customers Loan amount and its Interest Rate's distribution and how much is the customer's average?
* How is the distribution The Loan Grade and Intent of customers? Which type and grade dominant the most and the least?
* How is the Loan Status for each of the customers? How does it affect each of the customers loan amount type and grade?
* How are both of the Home ownership and Income yearly of customer affect loan amount type and grade?
* Which factor contributes the most in terms of credit loan default?

### Project Step by step:
* Dataset Profiing
* Dataset inspection, chceking for missing and anomaly values in dataset
* Dataset Cleaning, should there be any existing missing / anomlay values, otherwise prooced straight to next step
* Descriptive Statistic Analysis
* Exploratory Data Analysis
* Supervised Machine learning Classification model building and implementation

### Pakcages Used:
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit Learn
* Pycaret

### Dataset Profiling 
* Dataset consists of 2240 observations and 29 features with a total size of 64960. 


* 3116 missing values detected on `Loan Interest Rate` feature, and 895 were found on `Employment Length` one. Because of both features missing values are in still within the range threshold of values 5% FIlling factor, we have the option to drop them. However, instead we will impute the values, because at the end of the day, data is data and it is too valuable to just be dropped. 


### Data Cleaning 
* Imputing the missing values occured on `Loan Interest Rate` and `Employment Length` using scikit learn.

* Checking for anomaly values. There were non detected, 

* Cross checking for any missing values within dataset.

* Dataset is clean.

### Descriptive Statistics

Following insight we can get from the aboive statistics:
* Average yearly income of customers are in the range of 67000 USD.
* While average Loan amount is in the range of 9600 USD. 
* Lastly, the average emplyment of customers are 4 to 5 years.

### Exploratry Data Analysis
`Age` Feature <br>

* Majority of customers are in their late 20's and early 30's. 
* Loan amount for all ranges of cutomers are quite diverse. Interestingly, age does not solely determine the amount of loan of a given customer. Because, say we usually assume the mature ones (around their late 30's and 40's) having much more money than the rest of the customer age range, hence we commonly assume there will be a gap disparity in terms of the amount money loaned by these group of customer compared to the rest. However, this is no the case here.
* The Loan intent are also diverse across all customer's age ranges. Age also does not solely determine the loan type / intent of a customer. Sewms like every age group have their own objective of pursuing a loan which falls to one of the categories. 
* While for Loan grade is quite diverse aswell across all customer's age range, albeit many of younger ones preter to go wtih the lower grade ones (F and G class).

`Income` Features <br>

* IF we take a look to amount of loan, we can see that a high income of a customer does not straight correlate with a higher amount of loan. Similar to that `Age` previously, it is quite diverse. 
* For The Loan Intent is similar across all age, the high earner customers and the lower ones do have their own objective of pursuing fonje of these loan intent categories. 
* Loan grade pretty much in line with the how much customers earning, the hihger earning ones surely choosing the 1st and 2nd tier grade whilst the lower ones choosing the 4th and 5th tier grade. 

`Home Ownership` Feature <br>

* Distribution for home ownerships of all customers we can see majority of customers are either renting or mortgaging their houses, to be specific 50% are rent, 41% are mortage, while the rest of 9% of customers owned their house, 1 % fo other categories.
* Average amount of loan based on their home wonership, it seems to be relativel;y no different amongst all of these categories with an average of around 10000 USD. Specificly, Other-home ownership customers borrowed the most around 12000 USD, followed by mortgage of around 11000 USD then own for around 9000 lanstly rent similar ot own, around 8000 USD>
* Loan interest rate based on customer's home ownership heavily depends on the amount of loan they obtained. Because the amount of loan are similar for 4 categories with a small margin of diffference, then loan interst should of the same amoount. Here, we can see that it is indeed the case. OF all categories, the loan interest are around 10 and 11 USD.


* Customers Loan Intent baed on their home ownership, we can see majority of customers taking a loan are either rent or mortgage their house, whilst noly a handful of them own the house and very few in other's category. Rent-home ownership customer they mostly taking a lona for medical and education purposes, whiloe mortgage-homeownership customers mostly for education and debt consolidation purpose, for customers owning their house majority take loans for venture. 
* For loan grade based on customers home ownership, rent-homeownership mostly take the B and A grade, mortgage-homeownership mostly take the A grade, customers owning their house are also mostly A grade aswell.

`Loan Status` Feature <br>

Loan status refers to a default condition of a given customers. Default is the failure to make required interest or principal repayments on a debt, whether that debt is a loan or a security.  


* We can already see that 22% percent of customers are defaulting from their payment obligation. May want to take deep dive look of this, and determine which segment of customers will likely default from their payment obligation.
* It is clearly seen, customers who are defaulting are ones who have higher loan amount, loan interest and loan percentage in respect to their yearly income cmopared to non defautl ones. Especially the latter one with an oustanding 20% of incomme cmopared to 10% of non-default counterpart. Now, in normal circumstances this wuold not be a problem, however, should this be a major factor of customer defaulting then it needs to be something of concern.
* Bank should look out more closely on customers who take would loans either for medical and debtconsolidation purposes as these categories are the ones defaulting the most. At the same time, should also look out of customers with a lower tier grade loans, as the plot suggests, defaulting customers are ones who took the lower grades loan grades (5th, 4th and 3rd tier).

### Supervised Classification Model
Using Pycaret as our tool for data processing and model building / implementation. The figure follows is the setting for the grid processing.

### Model Building and Implementation
* Comparing and evaluating models performaces. We want to look at the `AUC score` as means to compare and evaluate models above. Due to the imbalanced nature of our dataset, `AUC score` is more precise and accurate than your regular Accuracy metric, though F1 harmony metric can also be tool to evaluate our models.
* One note aside from looking at a higher `AUC score`, we may want to look at and also consider other factor, which is the time. Time refers to how long the model took to implement to dataset. As we can see
* SO based on that consideration we will use Random Forest, LGBM alongside Graident Boosting Classifier as a base models.

