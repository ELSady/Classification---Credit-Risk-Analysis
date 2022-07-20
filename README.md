## Data Science Project: Credit Risk Analysis Overview
* Dataset consists of 32581 customer's credit history from The Bank on top of their current credit loan status.
* Visualize each customers's Loan Amount, Grade, Intent, Interest Rate and Income Perentage.
* Visualize and analyze those above features and how they fare based on customer's Age, Income yearly, Home Ownership and Loan Status.
* Build Supervised Machine learning Classification model, checking its feature importance and Shap value to determine which features contributes when it comes to predicting customer's response for the current campaign.

![Are-you-eligible-for-the-reduced-home-loan-interest-rates-FB-1200x700-compressed](https://user-images.githubusercontent.com/96014656/179934775-73e29d3f-bd60-436c-afbf-44c797912c92.jpg) <br>

Financial institutions used credit risk analysis models to determine the probability of default of a potential borrower. The models provide information on the level of a borrower’s credit risk at any particular time. If the lender fails to detect the credit risk in advance, it exposes them to the risk of default and loss of funds. Lenders rely on the validation provided by credit risk analysis models to make key lending decisions on whether or not to extend credit to the borrower and the credit to be charged. With the continuous evolution of technology, banks are continually researching and developing effective ways of modeling credit risk. A growing number of financial institutions are investing in new technologies and human resources to make it possible to create credit risk models using machine learning languages.

### Business Questions:
* How much the customers Loan amount and its Interest Rate's distribution and how much is the customer's average?
* How is the distribution The Loan Grade and Intent of customers? Which type and grade dominant the most and the least?
* How is the Loan Status for each of the customers? How does it affect each of the customers loan amount type and grade?
* How are both of the Home ownership and Income yearly of customer affect loan amount type and grade?

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
* Loan amount for all ranges of cutomers are quite diverse. Interestingly, age does not solely determine the amount of loan of a given customer. Because, say we usually assume the mature ones (around their late 30's and 40's) having much more money than the rest of the customer age range, hence we commonly assume there will be a gap disparity in terms of the amount money loaned by these group of customer compared to the rest. However, this is no thte case here.
* All acorss age range have  
