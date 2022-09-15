# Credit-card-fraud-detection

The project presented here is aimed at developing a data-driven model as part of automatic tools for detecting credit card fraud transactions. This is a supervised learning task, i.e. we have a labeled dataset containig information related to customer accounts, transactions made by customers, and personal customer information, as well as the binary target variable where 0 stands for authorized transactions while fraud transactions are marked with 1. 

The problem of fraud detection is a challenging one as we're usually dealing with highly imbalanced data. The percentage of fraud transactions is often negligible, comprising less than 1% of the dataset, so we have to come up with ways to handle that imbalance as well as suitable metrics to measure the performance of our model in a meaningful way. The accuracy metric often applied in binary classification problems would not be particularly useful in this scenario because it would attain a decent score even if we failed to detect any fraud samples at all. 

Thus we want to apply metrics like precision and recall. The former says how precise (how sure) our model is in detecting fraud transactions while recall is the amount of fraud cases our model is able to detect. Mathematically speaking, precision is the ratio of true positives to true positives + false positives while recall is the ratio of true positives to true positives + false negatives. 

Business-wise, there is a very fine tradeoff between the two metrics in our case. We need to protect user’s finances by trying to flag as many fraud transactions as possible (i.e. maximize recall) while at the same time try not to mislabel too many transactions (i.e. maximize precision) so users can reliably use their credit cards without the inconvenience of having transactions declined. The question which metric should be given more importance in this case is subject to debate although I believe more weight needs to be put on precision. This is because false positives (i.e. mistakenly labeling an authorized transaction as fraud) can be very harmful because they prevent customers from using their credit cards and cause a lot of inconvenience, which might result in customers refusing to use our service in the future, potentially leading to a greater dicrease in profit than the amount we would lose if we authorized some fraud transactions. Moreover, a great number of false alerts puts a great strain on human investigators.

Speaking of target metrics, it would be reasonable to expect a precision of 0.9 or higher for class 1, i.e. 90% of all the transactions labeled by our model as fraud are indeed fraud, and a recall of 0.6 or higher for class 1, meaning it would be possible to catch 60% of all fraud transactions.

The project is divided into several stages:

1. Business understanding, during which we determine project goals, discuss customer expectations, set ML metrics, etc.
2. Data infrastructure, during which we acquire the necessary data, provide data storage architecture and compute assets.
3. EDA (exploratory data analysis), during which we vizualize data, detect outliers, find relations between variables, etc.
4. Data preparation, during which we produce clean data for further engineering, extract new features, perform feature selection and create data pipeline.
5. Modeling, during which we develop a range of ML models, evaluate their quality and performance on nested cross-validation, determine the model that is best suitable for the task, create prediction pipeline and provide clean code with a clear project structure.
6. Deployment, during which we deploy the model with a data pipeline to a production-like environment and run tests to ensure the solution performs well, then introduce changes and modifications if necessary.
7. Acceptance, during which we hand the project over to the customer.

Each project-related task falls under a specific stage, satisfies the SMART criteria and has a corresponding stage label. 
