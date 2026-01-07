# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# #### The following data contains information about every wildfire in Oregon between the years 2000 and 2022. 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import cross_val_score

# %%
fire = pd.read_csv("/Users/sydney/Desktop/Projects/pred-analysis/data/ODF_Fire_Occurrence_Data_2000-2022.csv")
fire.columns

# %%
fire.head()

# %%
# Extract year and month from ReportDateTime in order to count the monthly and yearly fire counts.
fire["ReportDateTime"] = pd.to_datetime(fire["ReportDateTime"])
fire["Year"] = fire.ReportDateTime.dt.year
fire["Year_Month"] = fire.ReportDateTime.dt.strftime("%Y-%m")

# %%
# Display the total number of fires every 6 months in the data in chronological order
month_count = fire.groupby("Year_Month").size().reset_index(name="fire_count")

plt.figure(figsize=(15, 6))
ax = sns.lineplot(data=month_count, x="Year_Month", y="fire_count")
ax.set_xticks(ax.get_xticks()[::6]) 
plt.xticks(rotation=45)
plt.xlabel("Date (Year - Month)")
plt.ylabel("Number of Fires")
plt.tight_layout()

# %% [markdown]
# The plot shows a strong seasonal pattern in fire frequency, with pronounced spikes during the summer months and consistently low fire counts during winter. At peak periods, monthly fire counts exceed 450, while winter months often experience minimal activity.
# An additional trend is that peak summer fire activity appears to decrease over time. This decline may be influenced by several factors, such as improved fire prevention policies, stricter seasonal fire bans, increased public awareness, and expanded firefighting resources. However, further analysis would be required to confirm the underlying causes of this trend.

# %%
# Visualize the association of the total acreage burned each year
burned_acres = fire.groupby("Year")["EstTotalAcres"].sum().reset_index()

sns.regplot(data=burned_acres, x="Year", y="EstTotalAcres", ci=95, scatter_kws={"s": 50})
plt.title("Total Acreage Burned per Year")
plt.xlabel("Year")
plt.ylabel("Total Acres Burned");

# %%
# Fit a linear model and view summary stats to clarify generally positive association in previous graph.
X = burned_acres["Year"]
y = burned_acres["EstTotalAcres"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# %% [markdown]
# The regression results indicate an average increase of approximately 11,650 acres burned per year; however, the p-value (0.293) is not statistically significant. The 95% confidence interval ranges from −10,800 to 34,100 acres, which includes zero, indicating low confidence in the estimated slope despite the apparent upward trend in the visualization.

# %% [markdown]
# Returning to the scatterplot, there does appear to be a pretty considerable outlier early in the time series. Filter that point out of the data, then repeat q1.4.

# %%
# Filter out considerable outlier - remove that year
outlier = burned_acres.loc[burned_acres["EstTotalAcres"].idxmax(), "Year"]
filtered_years = burned_acres[burned_acres["Year"] != outlier]

# %%
# Repeat regression after outlier removal
X = filtered_years["Year"]
y = filtered_years["EstTotalAcres"]

X = sm.add_constant(X)
filtered_model = sm.OLS(y, X).fit()
print(filtered_model.summary())

# %% [markdown]
# After removing the outlier, the regression slope increases to approximately 22,680 acres per year with a significant p-value (0.009) and a fully positive 95% confidence interval (6,444 to 38,900). This increases confidence in a positive trend, though the excluded outlier represents a real event and should be considered when interpreting the results.

# %%
# Compare the burn times of fires started by humans and fires started by lightning.
fire["Control_DateTime"] = pd.to_datetime(fire["Control_DateTime"])
fire["ReportDateTime"] = pd.to_datetime(fire["ReportDateTime"])

fire["Burn_time_days"] = (fire["Control_DateTime"] - fire["ReportDateTime"]).dt.days

fire = fire[fire["Burn_time_days"] > 0]

fire["Log_Burn_time"] = np.log(fire["Burn_time_days"])

# %%
sns.histplot(data=fire[fire["HumanOrLightning"]=="Human"], x="Log_Burn_time", color="blue", label="Human Started", kde=True)
sns.histplot(data=fire[fire["HumanOrLightning"]=="Lightning"], x="Log_Burn_time", color="yellow", label="Lightening Started", kde=True)

plt.title("Log Transformed Burn Times (Human vs Lightening Fires)")
plt.xlabel("Burn Time in Days (Log Transformed)")
plt.ylabel("Density")
plt.legend();

# %% [markdown]
# Human caused fires tend to have shorter burn times, with most lasting 1 to 2 days, while lightning caused fires show greater variability and a longer tail toward extended durations. These differences may reflect faster detection of human caused fires and potential reporting biases, specifically given the sharp peak observed in the human fire distribution.

# %% [markdown]
# ### Conclusion
#
# This analysis identified strong seasonal patterns in wildfire activity, with peak fire frequency occurring during summer months and substantial year to year variability in total acreage burned. While some visualizations suggested temporal trends, statistical results showed that conclusions about long term increases depend on modeling choices and sensitivity to outliers. Differences in burn duration between human and lightning caused fires highlight the role of detection and response, though potential reporting bias should be considered. Overall, these findings emphasize the complexity of wildfire behavior and the importance of cautious interpretation.
