import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing distributions
from lifelines import WeibullFitter,\
                      ExponentialFitter,\
                      LogNormalFitter,\
                      LogLogisticFitter,\
                      WeibullAFTFitter,\
                      LogNormalAFTFitter,\
                      LogLogisticAFTFitter


# Reading data
df = pd.read_csv('telco.csv')
#print(df.head())

# Data Preprocessing

# Indexing by ID
df.set_index('ID', inplace=True)

# Categorizing churn, if yes:1,if no:0
df['churn'] = pd.Series(np.where(df.churn.values == "Yes", 1, 0), df.index)

# Encoding the categorical variables into dummies
encode_cols = ['region', 'marital', 'ed', 'retire', 'gender', 'voice', 'internet', 'forward', 'custcat']
survival_df = pd.get_dummies(df, columns=encode_cols, prefix=encode_cols, drop_first=True)

# We have zeros in the 'tenure' column.
# This model does not allow for non-positive durations. Suggestion: add a small positive value to zero elements.
survival_df["tenure"] = np.where(survival_df["tenure"] == 0, 0.000001, survival_df["tenure"])
#print(survival_df.head())

# Instantiate distributions
wb = WeibullFitter()  # Weibull distribution
ex = ExponentialFitter()  # Exponential distribution
log = LogNormalFitter()  # Log-Normal distribution
loglogis = LogLogisticFitter()  # Log-Logistic distribution

# Fitting and plotting to compare different distributions
fig, ax = plt.subplots(figsize=(16, 8))

for model, color, label in zip([wb, ex, log, loglogis], ['blue', 'green', 'orange', 'red'],
                               ['Weibull', 'Exponential', 'Log-Normal', 'Log-Logistic']):
    model.fit(durations=survival_df["tenure"], event_observed=survival_df["churn"])
    model.plot_survival_function(ax=ax, color=color, label=label)

    print("The AIC value for", model.__class__.__name__, "is", model.AIC_)
    print("The BIC value for", model.__class__.__name__, "is", model.BIC_)
    model.print_summary()

# Add legend and title
ax.legend()
ax.set_title('Survival Curves')

plt.show()

# Building models using AFT fitters
wb_aft = WeibullAFTFitter()
log_aft = LogNormalAFTFitter()
loglogis_aft = LogLogisticAFTFitter()

fig, ax = plt.subplots(figsize=(16, 8))

# Fitting and plotting different distributions
models = [wb_aft, log_aft, loglogis_aft]
labels = ["WeibullAFTFitter", "LogNormalAFTFitter", "LogLogisticAFTFitter"]
colors = ['blue', 'orange', 'red']  # Adjusted color for LogLogisticAFTFitter

for model, label, color in zip(models, labels, colors):
    model.fit(survival_df, duration_col="tenure", event_col="churn")

    model.print_summary()

    # Assuming you want to plot the survival function for the first row of the dataframe
    plt.plot(
        model.predict_survival_function(survival_df.loc[1]),
        label=label,
        color=color
    )

# Add legend and labels
plt.legend()
plt.title('Survival Curves - AFT Models')
plt.xlabel('Time')
plt.ylabel('Survival Probability')

plt.show()


# keeping best model and significant features

logn_aft = LogNormalAFTFitter()
logn_aft.fit(survival_df, duration_col='tenure', event_col='churn')
#logn_aft.print_summary()
survival_df = survival_df[["tenure", "churn", "address", "age", "custcat_E-service", "custcat_Plus service", "custcat_Total service", "internet_Yes", "marital_Unmarried", "voice_Yes"]]

# doing same above steps but with only significant features
# and we will see better performance and decreased AIC value
logn_aft = LogNormalAFTFitter()
logn_aft.fit(survival_df, duration_col='tenure', event_col='churn')

# Calculate CLV with best model
pred_clv = logn_aft.predict_survival_function(survival_df)

# takeing one year interval
pred = pred_clv.loc[1:12, :]
MM = 1500
r = 0.08
for col in range(1, len(pred.columns)+1):
    for row in range(1, 13):
        pred[col][row] = pred[col][row] / (1 + r / 12)**(row - 1)
df['CLV'] = MM * pred.sum(axis = 0)

# CLV for customers that left the company within a year
average_clv_left = df.loc[(df["tenure"] <= 12) & (df["churn"] == 1), "CLV"].mean()
#print(f"Average CLV for customers who left within a year: {average_clv_left}")

# Yearly CLV of customers that left
clv_yearly = df.loc[(df["tenure"] <= 12) & (df["churn"] == 1), "CLV"].sum()
#print(f"Yearly CLV of customers who left within a year: {clv_yearly}")
