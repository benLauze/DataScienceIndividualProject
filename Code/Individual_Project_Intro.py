#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[ ]:


NAME = "Benjamin Lauze"
COLLABORATORS = ""


# ---

# # Data Science Individual Project
# ## Objective
# 
# Provide a data-driven solution to a problem that excites you using the tools discussed (or related to) in this course.
# 
# ## Datasets
# The dataset must be different with your group project final topic. 
# 
# ## Tasks
# For your project, you should:
# - Pick an issue/problem that excites you
# - Create at least three questions in the topic/issue/problem to help your solve the problem
# - Select or create datasets
# - Familiarize yourself with that data, if necessary:
#     - data munging
#     - feature engineering
# - Choose proper model/method
#     - Train/Fit the model by the datasets
#     - Potential Methods: Classification, Regression, ...
#     - Potential Tools: SciKit, TensorFlow (tf), ...
# - Analyze results
# - Future work
# - Reference
# 
# ## Deliverables
# Deliverables for your project:
# 
# - Draft of report: 
#     - Required: draft of introduction, datasets, methodology. 
#     - Optional: draft of result, discussion
# 
# Due 11/24 (Sunday), at 11:59 pm
# 
# More details about submission will be released before the due
# 
# - Final report: introduction, datasets, methodology, results, discussion/suggestion, reference + codes
# 
# Due on Sunday of the last lecture week (12/8) at 11:59 pm
# 
# More details about the requirements of report will be release before the due
# 
# - Method 
#     - Published code, pictures and report to a repository with readme [reference](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
#     - if use private Github repo, must add 'pangwit' by the following steps in [link](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-access-to-your-personal-repositories/inviting-collaborators-to-a-personal-repository)
# 
# 
# 
# ## Rubric
# |Category | Explanation| %|
# |--|--|-|
# |Introduction|Why was the project undertaken? What was the research question, the tested hypothesis or the purpose of the research?|10|
# |Selection of Data|What is the source of the dataset? Characteristics of data? Any munging, imputation, or feature engineering?|20|
# |Methods|What materials/tools were used in answering the research question?|20|
# |Results|What answer was found to the research question; what did the study find? Any visualizations?|20|
# |Discussion|What might the answer imply and why does it matter? How does it fit in with what other researchers have found? What are the perspectives for future research? |20|
# |Coding & Reference|Clear citation at end of the report. ipynb file with clear comments and datafile.|10|
# 
# Rubric based on the IMRAD:https://en.wikipedia.org/wiki/IMRAD
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from scipy import stats


# In[ ]:


#Question 1: How do different sectors contribute to carbon emissions?
dfSector = pd.read_csv('co-emissions-by-sector.csv')
dfSector


# In[ ]:


#Cleaning Question 1
#Some forestry sectors have negative emissions, due to plant life absorbing it
dfSector = dfSector.drop_duplicates()
dfSector = dfSector.drop(columns = "Code")
dfSector = dfSector[dfSector['Year'] >= 2010]
dfSector


# In[ ]:


def filter_countries(dfSector, selected_countries, column_name='Entity'):
    # Validate inputs
    if not isinstance(selected_countries, list):
        raise ValueError("selected_countries must be a list")
        
    selected_countries = [str(country).strip() for country in selected_countries]
    
    filtered_dfSector = dfSector[
        dfSector[column_name].str.strip().isin(selected_countries) |  # Exact match
        dfSector[column_name].str.contains('|'.join(selected_countries), case=False)  # Partial match
    ]
        
    filtered_dfSector = filtered_dfSector.reset_index(drop=True)
    
    return filtered_dfSector


#Usage
selected_countries = ['United States', 'China', 'India', 'Russia', 'Japan']
filtered_dfSector = filter_countries(dfSector, selected_countries)
filtered_dfSector.to_csv('filtered_dataset.csv', index=False)
filtered_dfSector


# In[ ]:


plt.figure(figsize=(20, 15))

#List of countries and sectors to visualize
countries = ['China', 'India', 'Japan', 'Russia', 'United States']
sectors = [
    'Transport', 
    'Electricity and Heat', 
    'Industry', 
    'Buildings', 
    'Carbon dioxide Manufacturing and Construction'
]

#Color palette for consistent sector coloring
colors = ['blue', 'green', 'red', 'purple', 'orange']

#Create subplots for each country
for i, country in enumerate(countries, 1):
    #Select subplot
    plt.subplot(5, 1, i)
    
    #Filter data for the specific country
    country_data = filtered_dfSector[filtered_dfSector['Entity'] == country]
    
    #Plot each sector
    for sector, color in zip(sectors, colors):
        plt.plot(country_data['Year'], country_data[sector], 
                 marker='o', label=sector, color=color)
    
    #Formatting
    plt.title(f'{country} Emissions Sectors (2010-2020)')
    plt.xlabel('Year')
    plt.ylabel('Emissions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

#Adjust layout
plt.tight_layout()

#Save the figure
plt.savefig('multi_country_emissions_time_series.png', bbox_inches='tight')


# In[ ]:


#Question 2 (Region): Which world regions produce the most emissions?
dfRegion = pd.read_csv('annual-co-emissions-by-region.csv')
dfRegion 


# In[ ]:


#Cleaning Question 2
dfRegion = dfRegion.drop_duplicates()
#dfRegion = dfRegion.drop(columns = "Code")
dfRegion = dfRegion[dfRegion['Year'] >= 2010]
def filter_regions(dfRegion, selected_regions, column_name='Entity'):
    # Validate inputs
    if not isinstance(selected_regions, list):
        raise ValueError("selected_regions must be a list")
        
    selected_regions = [str(country).strip() for country in selected_regions]
    
    filtered_dfRegion = dfRegion[
        dfRegion[column_name].str.strip().isin(selected_regions) |  # Exact match
        dfRegion[column_name].str.contains('|'.join(selected_regions), case=False)  # Partial match
    ]
        
    filtered_dfRegion = filtered_dfRegion.reset_index(drop=True)
    
    return filtered_dfRegion


#Usage
selected_regions = ['United States', 'China', 'India', 'Russia', 'Japan']
filtered_dfRegion = filter_regions(dfRegion, selected_regions)
filtered_dfRegion.to_csv('filtered_dataset2.csv', index=False)
filtered_dfRegion


# In[ ]:


data_encoded = pd.get_dummies(filtered_dfRegion, columns=['Entity'], prefix='Country')

#Prepare features (X) and target variable (y)
features = ['Year'] + [col for col in data_encoded.columns if col.startswith('Country_')]
X = data_encoded[features]
y = data_encoded['Annual CO₂ emissions']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create and train the multiple regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#Make predictions
y_pred = model.predict(X_test_scaled)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#Print model evaluation metrics
print("Model Evaluation:")
print(f"Mean Squared Error: {mse:,.0f}")
print(f"R-squared Score: {r2:.4f}")

#Create a dataframe of coefficients
feature_names = features
coefficients = model.coef_
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
print("\nModel Coefficients:")
print(coef_df)

#Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("Actual vs Predicted CO2 Emissions")
plt.tight_layout()
plt.savefig("actual_vs_predicted_co2_emissions.png")
plt.show()

#Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel("Predicted CO2 Emissions")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.savefig("residual_plot.png")
plt.show()


# In[ ]:


#Question 3: What are some ways we could change our energy consumption to limit carbon emissions?
dfSolar = pd.read_csv('Solar Power Dataset.xlsx - Data.csv')
dfSolar


# In[ ]:


#dfSolar['Solar power net generation in the United States from 2000 to 2023 (in gigawatt hours)'] = dfSolar['Solar power net generation in the United States from 2000 to 2023 (in gigawatt hours)'].str.replace(',', '').astype(float)

#Create the time series plot
plt.figure(figsize=(12, 6))
plt.plot(dfSolar['Year'], dfSolar['Solar power net generation in the United States from 2000 to 2023 (in gigawatt hours)'], 
         marker='o', linestyle='-', linewidth=2, markersize=8)

#Customize the plot
plt.title('Solar Power Net Generation in the United States (2010-2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Net Generation (Gigawatt Hours)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

#Rotate x-axis labels for better readability
plt.xticks(rotation=45)

#Add data labels
for x, y in zip(dfSolar['Year'], dfSolar['Solar power net generation in the United States from 2000 to 2023 (in gigawatt hours)']):
    plt.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=9)

#Tight layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("solar_plot.png")

#Show the plot
plt.show()


# In[ ]:


#Question 3: What are some ways we could change our energy consumption to limit carbon emissions?
dfWind = pd.read_csv('Wind Power Dataset.xlsx - Data.csv')
dfWind


# In[ ]:


#dfWind['Net electricity generation from wind in the United States from 2000 to 2023 (in terawatt hours)'] = dfSolar['Net electricity generation from wind in the United States from 2000 to 2023 (in terawatt hours)'].str.replace(',', '').astype(float)

#Create the time series plot
plt.figure(figsize=(12, 6))
plt.plot(dfSolar['Year'], dfWind['Net electricity generation from wind in the United States from 2000 to 2023 (in terawatt hours)'], 
         marker='o', linestyle='-', linewidth=2, markersize=8)

#Customize the plot
plt.title('Net electricity generation from wind in the United States from (2010 to 2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Net Generation (Terawatt Hours)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

#Rotate x-axis labels for better readability
plt.xticks(rotation=45)

#Add data labels
for x, y in zip(dfWind['Year'], dfWind['Net electricity generation from wind in the United States from 2000 to 2023 (in terawatt hours)']):
    plt.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=9)

#Tight layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("wind_plot.png")

#Show the plot
plt.show()


# In[ ]:


#Question 4: What government policies (across the world, or in the United States specifically) have correlated to changes in emissions?
#Fossil fuel demand worldwide from 1965 to 2020, with a forecast until 2050 by scenario (in exajoules)
dfForecast = pd.read_csv('Forecast Dataset.xlsx - Data.csv')
dfForecast


# In[ ]:


#dfForecast['Year'] = pd.to_datetime(dfForecast['Year'], format='%Y')
#dfForecast.set_index('Year', inplace=True)

#Focus on actual supply and available scenarios
supply_columns = ['Actual supply', 'Stated policies scenario', 
                  'Announced pledges scenario', 
                  'Remaining under 1.5° C global heating scenario']

#Remove rows with all NaN values
df_cleaned = dfForecast.dropna(how='all', subset=supply_columns)

#Plotting
plt.figure(figsize=(12, 8))
for col in supply_columns:
    if col in dfForecast.columns:
        plt.plot(dfForecast.index, dfForecast[col], label=col, marker='o')

plt.title('Fossil Fuel Energy Supply Scenarios Over Time', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Energy Supply', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Time Series Decomposition (for Actual Supply)
actual_supply = dfForecast['Actual supply'].dropna()
decomposition = seasonal_decompose(actual_supply, period=1)


def adf_test(timeseries):
    print('Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#Trend Analysis
def trend_analysis(data):
    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
    
    print("\nTrend Analysis:")
    print(f"Slope: {slope:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    return slope, r_value**2, p_value

#Comparative Scenario Analysis
def scenario_comparison(df):
    scenarios = ['Actual supply', 
                 'Stated policies scenario', 
                 'Announced pledges scenario', 
                 'Remaining under 1.5° C global heating scenario']
    
    print("\nScenario Comparison:")
    for scenario in scenarios:
        if scenario in df.columns:
            data = df[scenario].dropna()
            print(f"\n{scenario}:")
            print(f"Mean: {data.mean():.2f}")
            print(f"Standard Deviation: {data.std():.2f}")
            trend_analysis(data)

#Perform analyses
print("Time Series Analysis of Energy Supply Scenarios")
adf_test(actual_supply)
trend_analysis(actual_supply)
scenario_comparison(dfForecast)

#Visualization of Trend and Decomposition
plt.figure(figsize=(12, 10))

plt.subplot(411)
plt.plot(actual_supply.index, actual_supply.values)
plt.title('Actual Supply Time Series')

plt.subplot(412)
plt.plot(actual_supply.index, decomposition.trend)
plt.title('Trend')

plt.subplot(413)
plt.plot(actual_supply.index, decomposition.seasonal)
plt.title('Seasonality')

plt.subplot(414)
plt.plot(actual_supply.index, decomposition.resid)
plt.title('Residuals')

plt.tight_layout()
plt.savefig("fossil_fuel_plot.png")
plt.show()


# In[ ]:


#Question 5: How is the world currently being affected by climate change (temperature increases, an excess of natural disasters, etc)? 
dfTemp = pd.read_csv('Global Temp Dataset.xlsx - Data.csv')
dfTemp


# In[ ]:


#Create the time series plot
plt.figure(figsize=(12, 6))
plt.plot(dfTemp['Year'], dfTemp['Annual anomalies in global land and ocean surface temperature from 1950 to 2023, based on temperature departure (in degrees Celsius)'], 
         marker='o', linestyle='-', linewidth=2, markersize=8)

#Customize the plot
plt.title('Global Tempurature Deviations, Overtime (1950 - 2023)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Deviation (Celsius)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

#Rotate x-axis labels for better readability
plt.xticks(rotation=45)

#Add data labels
for x, y in zip(dfTemp['Year'], dfTemp['Annual anomalies in global land and ocean surface temperature from 1950 to 2023, based on temperature departure (in degrees Celsius)']):
    plt.text(x, y, f'{y:,.0f}', ha='center', va='bottom', fontsize=9)

#Tight layout to prevent cutting off labels
plt.tight_layout()

plt.savefig("temp_plot.png")
plt.show()


# In[ ]:




