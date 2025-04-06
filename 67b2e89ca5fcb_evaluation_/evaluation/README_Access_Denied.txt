Real-Time Bidding System for Digital Advertising

------------------------------------------------------------
Team Name: Access Denied
------------------------------------------------------------
Team Members:
- Samay Jain (23/MC/128)
- Ibrahim Haneef (23/IT/72)
- Mohd. Hassan (23/IT/100)
- Hemant Singh (23/IT/69)

------------------------------------------------------------
Table of Contents
------------------------------------------------------------
1. Instructions to Run the Code
2. Description of the Approach
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Selection
6. Hyperparameter Tuning
7. Evaluation Strategy
8. Validation Results
9. Additional Information

------------------------------------------------------------
1. Instructions to Run the Code
------------------------------------------------------------
Environment Setup:
- Python Version: 3.9  
  Ensure you are using Python 3.9 to maintain compatibility with our code and libraries.

Install Dependencies:
Run the following command to install all required libraries:
This will install packages such as pandas, numpy, scikit-learn, LightGBM, joblib, and tqdm.

Training the Models:
Execute the training script to build and evaluate our models:
Upon completion, this script will generate the following files:
- `ctr_model.pkl`
- `cvr_model.pkl`
- `market_price_model.pkl`
- `label_encoders.pkl`
- `scaler.pkl`

These files contain the trained models and preprocessing objects, which are essential for the bidding engine.

Testing the Bidding Engine:
To simulate bid requests and verify the engine’s functionality, run:
This test script will output the bid price, allowing you to check if the system is making correct bid decisions.

------------------------------------------------------------
2. Description of the Approach
------------------------------------------------------------
Our approach to building the RTB system is structured around four key pillars:

- Data Processing: 
  We efficiently handle large datasets by processing data in chunks. This strategy minimizes memory usage and speeds up the workflow.

- Feature Engineering:  
  We transform raw bid data into meaningful features. This involves extracting temporal features, user-agent details, and ad slot characteristics, among others. These engineered features are crucial in improving model performance.

- Model Training:  
  We develop three distinct models:
  - A LightGBM classifier for predicting the Click-Through Rate (CTR)
  - A LightGBM classifier for predicting the Conversion Rate (CVR)
  - A LightGBM regressor for estimating the market price  
  Each model is optimized to handle the inherent class imbalances in the dataset.

- Bidding Strategy: 
  The system computes an Expected Value (EV) for each bid using:
  
      EV = CTR + N * CVR

  (For example, if N = 10 for a particular advertiser, then Score = Clicks + 10 * Conversions.)  
  If the EV (or the EV-to-market-price ratio) exceeds a predefined threshold, the system places a bid slightly above the predicted market price. Budget constraints are enforced to ensure cost-effective spending.

This comprehensive approach ensures that the RTB system can make real-time decisions while balancing performance and budget.

------------------------------------------------------------
3. Exploratory Data Analysis (EDA)
------------------------------------------------------------
Our EDA process was critical in understanding the underlying data distributions and relationships. Key insights include:

Data Volume:
- Approximately 53 million bids were analyzed.
- 12 million impressions represent 18.6% of the total bids.
- 10,011 clicks (≈0.015% of impressions).
- 487 conversions (≈0.0007% of impressions).

Class Imbalance:
- The dataset exhibits severe imbalance in both click and conversion events, a challenge that influenced our choice of model and evaluation metrics.

Temporal Patterns:
- We examined bid volumes, CTR, and CVR across different hours and days to capture time-based trends and user behavior patterns.

Advertiser Behavior:
- Distinct bidding patterns and performance metrics were observed across various advertisers, providing insight into market dynamics.

Ad Slot Characteristics:
- Analysis of ad slot dimensions and types revealed correlations with performance metrics.

Pricing Dynamics:
- Detailed analysis of floor prices, bid prices, and winning prices helped in understanding market behavior.

This thorough EDA guided the feature engineering process and ensured that the models focus on the most impactful variables.

------------------------------------------------------------
4. Feature Engineering
------------------------------------------------------------
Feature engineering transforms raw bid data into structured inputs for our models. Key features include:

Time Features:
- **Hour of Day & Day of Week:** Capture variations in user activity.
- **is_weekend Flag:** Distinguish between weekday and weekend behaviors.

User-Agent Features:
- Flags such as is_mobile, is_chrome, is_firefox, and is_safari help infer the device and browser characteristics of users.

Ad Slot Features:
- ad_area: Calculated as width × height, it represents the physical space of the ad.
- is_premium_ad Flag: Indicates whether the ad slot is considered premium.

Categorical Features:
- We use label encoding for features like region, city, ad exchange, domain, URL, and ad slot ID, which allows the models to handle non-numeric data.

Numerical Features:
- Scaling is applied to ad slot dimensions and floor price to normalize these values across different ranges.

Each feature was selected based on domain knowledge and validated through EDA, ensuring they contribute positively to model performance.

------------------------------------------------------------
5. Model Selection
------------------------------------------------------------
We selected LightGBM for its speed, efficiency, and accuracy, especially on large and imbalanced datasets. Our models include:

- CTR Prediction Model:
  A LightGBM Classifier that predicts the probability of an ad impression leading to a click.

- CVR Prediction Model: 
  A LightGBM Classifier that predicts the probability of a click resulting in a conversion.

- Market Price Prediction Model:
  A LightGBM Regressor that estimates the expected market price for an ad impression.

LightGBM’s ability to handle large datasets and its flexibility in dealing with imbalanced data make it ideal for our RTB system.

------------------------------------------------------------
6. Hyperparameter Tuning
------------------------------------------------------------
We optimized our models using the following hyperparameters:

- `n_estimators`: 100  
  Controls the number of boosting rounds.
  
- `learning_rate`: 0.05  
  Determines the step size at each iteration while moving toward a minimum of the loss function.

- `max_depth`: 7  
  Limits the depth of each tree to prevent overfitting.

- `num_leaves`: 31  
  Controls the complexity of the tree model.

- `min_child_samples`: 20  
  Specifies the minimum number of samples required to be at a leaf node.

- `class_weight`: 'balanced' (used for classifiers)  
  Adjusts weights inversely proportional to class frequencies.

- `random_state`: 42  
  Ensures reproducibility of results.

These settings were derived from initial experiments and can be further refined using grid search or Bayesian optimization techniques.

------------------------------------------------------------
7. Evaluation Strategy
------------------------------------------------------------
To ensure our models generalize well, we employed a rigorous evaluation strategy:

Data Splitting:
- Train Set: ~7,669,559 rows
- Validation Set: ~958,695 rows
- Test Set: ~958,695 rows

Metrics:
- CTR & CVR Models: Evaluated using ROC AUC and PR-AUC, which are appropriate for imbalanced classification.
- Market Price Model: Evaluated using RMSE (Root Mean Squared Error).

Early Stopping:
- We implemented early stopping with a patience of 10 iterations to prevent overfitting during training.

This strategy helps ensure that the models are robust and perform well on unseen data.

------------------------------------------------------------
8. Validation Results
------------------------------------------------------------
Our validation results indicate strong predictive performance:

CTR Model:
- **ROC AUC: 0.8803
- **PR-AUC: 0.0049

CVR Model:
- ROC AUC: 0.8516
- PR-AUC: 0.5911

Market Price Model:
- RMSE: 47.92

These metrics confirm that the models can effectively distinguish between positive and negative cases, despite the data imbalance.

------------------------------------------------------------
9. Additional Information
------------------------------------------------------------
Budget Pacing:  
The system incorporates a basic budget pacing mechanism, ensuring that bids are placed only when sufficient budget is available. This is crucial for maintaining cost efficiency in a live bidding environment.

Bidding Logic: 
Bids are made only if the Expected Value (EV) exceeds a defined threshold. When this condition is met, the bid is set slightly above the predicted market price to increase the chance of winning the auction without overspending.

Future Improvements:
- Hyperparameter Tuning & Model Ensembling: Further tuning and combining multiple models could enhance prediction accuracy.
- Training on Larger Datasets: Expanding the training data to include more days would likely improve reliability.
- Advanced Budget Pacing: Implementing a more sophisticated mechanism could optimize spend over time.
- Additional Features: Integrating real-time user behavior signals or other contextual data could further boost performance.
- Stress Testing: Simulating high-load conditions will help ensure the system's scalability and robustness.

This comprehensive README provides an overview of our real-time bidding system. For more detailed insights, please refer to the full documentation and inline code comments.

