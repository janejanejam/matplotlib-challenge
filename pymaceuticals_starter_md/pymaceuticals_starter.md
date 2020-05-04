## Observations and Insights 


```python
# 1. Capomulin and Ramicane seem to be the most successful regimens as they both have the lowest tumor volume over time points, lowest Var/STD/SEM, as well as the highest mice count overall, indicating high survival rate.
# 2. Deeper dive in Capomulin shows that there's some correlation between mouse weight and tumor volume (as tumor volume goes up, so does mouse weight), but there's also big range of tumor volume within same mouse weight. 
# 3. The outliers seem very far from the median, including a negative value, so would be worth looking deeper into these to see if there's bad datathat could affect overall analysis of the promising regimens. 
    
```


```python
%matplotlib inline
```


```python
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)
```


```python
mouse_metadata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug Regimen</th>
      <th>Sex</th>
      <th>Age_months</th>
      <th>Weight (g)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s185</td>
      <td>Capomulin</td>
      <td>Female</td>
      <td>3</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>x401</td>
      <td>Capomulin</td>
      <td>Female</td>
      <td>16</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>m601</td>
      <td>Capomulin</td>
      <td>Male</td>
      <td>22</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>g791</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>11</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
study_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b128</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>f932</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>g107</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a457</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>c819</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Combine the data into a single dataset
combined_data = pd.merge(mouse_metadata, study_results, how='outer', on='Mouse ID')
combined_data.head() #1893 rows

combined_data.isnull().values.any()
```




    False




```python
# Checking the number of mice in the DataFrame.
raw_count = {"Mice Count: Raw Data": [len(combined_data["Mouse ID"])]}
raw_count_df = pd.DataFrame(raw_count)
raw_count_df


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mice Count: Raw Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1893</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a clean DataFrame by dropping the duplicate mouse by its ID.
combined_clean = combined_data.drop_duplicates(subset=['Mouse ID', 'Timepoint'], keep=False)
combined_clean #1883 rows

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Drug Regimen</th>
      <th>Sex</th>
      <th>Age_months</th>
      <th>Weight (g)</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>0</td>
      <td>45.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>5</td>
      <td>38.825898</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>10</td>
      <td>35.014271</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>15</td>
      <td>34.223992</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>k403</td>
      <td>Ramicane</td>
      <td>Male</td>
      <td>21</td>
      <td>16</td>
      <td>20</td>
      <td>32.997729</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>z969</td>
      <td>Naftisol</td>
      <td>Male</td>
      <td>9</td>
      <td>30</td>
      <td>25</td>
      <td>63.145652</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1889</th>
      <td>z969</td>
      <td>Naftisol</td>
      <td>Male</td>
      <td>9</td>
      <td>30</td>
      <td>30</td>
      <td>65.841013</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>z969</td>
      <td>Naftisol</td>
      <td>Male</td>
      <td>9</td>
      <td>30</td>
      <td>35</td>
      <td>69.176246</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1891</th>
      <td>z969</td>
      <td>Naftisol</td>
      <td>Male</td>
      <td>9</td>
      <td>30</td>
      <td>40</td>
      <td>70.314904</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1892</th>
      <td>z969</td>
      <td>Naftisol</td>
      <td>Male</td>
      <td>9</td>
      <td>30</td>
      <td>45</td>
      <td>73.867845</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>1883 rows × 8 columns</p>
</div>




```python
# Checking the number of mice in the clean DataFrame.
clean_count = {"Mice Count: Clean Data": [len(combined_clean["Mouse ID"])]}
clean_count_df = pd.DataFrame(clean_count)
clean_count_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mice Count: Clean Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1883</td>
    </tr>
  </tbody>
</table>
</div>



## Summary Statistics


```python
# Generate a summary statistics table of mean, median, variance, standard deviation, 
# and SEM of the tumor volume for each regimen
summary_data = combined_clean[['Drug Regimen', 'Tumor Volume (mm3)']]
summary_table = summary_data.groupby('Drug Regimen').agg(['mean', 'median', 'var', 'std', 'sem'])
summary_table

# This method produces everything in a single groupby function.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Tumor Volume (mm3)</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>median</th>
      <th>var</th>
      <th>std</th>
      <th>sem</th>
    </tr>
    <tr>
      <th>Drug Regimen</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capomulin</th>
      <td>40.675741</td>
      <td>41.557809</td>
      <td>24.947764</td>
      <td>4.994774</td>
      <td>0.329346</td>
    </tr>
    <tr>
      <th>Ceftamin</th>
      <td>52.591172</td>
      <td>51.776157</td>
      <td>39.290177</td>
      <td>6.268188</td>
      <td>0.469821</td>
    </tr>
    <tr>
      <th>Infubinol</th>
      <td>52.884795</td>
      <td>51.820584</td>
      <td>43.128684</td>
      <td>6.567243</td>
      <td>0.492236</td>
    </tr>
    <tr>
      <th>Ketapril</th>
      <td>55.235638</td>
      <td>53.698743</td>
      <td>68.553577</td>
      <td>8.279709</td>
      <td>0.603860</td>
    </tr>
    <tr>
      <th>Naftisol</th>
      <td>54.331565</td>
      <td>52.509285</td>
      <td>66.173479</td>
      <td>8.134708</td>
      <td>0.596466</td>
    </tr>
    <tr>
      <th>Placebo</th>
      <td>54.033581</td>
      <td>52.288934</td>
      <td>61.168083</td>
      <td>7.821003</td>
      <td>0.581331</td>
    </tr>
    <tr>
      <th>Propriva</th>
      <td>52.458254</td>
      <td>50.854632</td>
      <td>44.053659</td>
      <td>6.637293</td>
      <td>0.540135</td>
    </tr>
    <tr>
      <th>Ramicane</th>
      <td>40.216745</td>
      <td>40.673236</td>
      <td>23.486704</td>
      <td>4.846308</td>
      <td>0.320955</td>
    </tr>
    <tr>
      <th>Stelasyn</th>
      <td>54.233149</td>
      <td>52.431737</td>
      <td>59.450562</td>
      <td>7.710419</td>
      <td>0.573111</td>
    </tr>
    <tr>
      <th>Zoniferol</th>
      <td>53.236507</td>
      <td>51.818479</td>
      <td>48.533355</td>
      <td>6.966589</td>
      <td>0.516398</td>
    </tr>
  </tbody>
</table>
</div>



## Bar Plots


```python
# Groupby clean df by 'Timepoint' and 'Drug Regimen'
drug_timepoint_group = combined_clean.groupby(['Timepoint', 'Drug Regimen'])

# Calculate count of mice by 'Timepoint' and 'Drug Regimen'
mice_count = drug_timepoint_group['Mouse ID'].count()
mice_count.sort_values(ascending=True)

# Generate a bar plot showing the number of mice per time point for each treatment throughout the course of the study using pandas.
mice_timepoint_bar = mice_count.plot(kind='bar', figsize=(40,15), grid=True, label='index', fontsize=20)
mice_timepoint_bar.set_xlabel("Time Points")
mice_timepoint_bar.set_ylabel("Mice Count")
mice_timepoint_bar.set_title("Mice Count per Time Point for each Drug Regimen")
plt.show()

# Note to TA's: Different color bars for each group would help this visual. Analysis is that most drug regimens started with same number
# of mice, but as the timepoints progressed, some regimens kept high number of mice alive while others had varying degrees of mice death. 
```


![png](output_13_0.png)



```python
# Generate a bar plot showing the number of mice per time point for each treatment throughout the course of the study using 
# pyplot.

x_axis = np.arange(len(mice_count))

plt.bar(x_axis, mice_count)
plt.xticks(mice_count)
plt.xlabel("Time Points")
plt.ylabel("Mice Count")
plt.title("Mice Count per Time Point for each Drug Regimen")

# Note to TA's: I had trouble changing figsize and adding xticks to output on this version.
```




    Text(0.5, 1.0, 'Mice Count per Time Point for each Drug Regimen')




![png](output_14_1.png)


## Pie Plots


```python
# Generate a pie plot showing the distribution of female versus male mice using pandas

female_count = sum(combined_clean['Sex'] == 'Female')
male_count = sum(combined_clean['Sex'] == 'Male')
sex_count = pd.DataFrame({
    "": [female_count, male_count]},
    index = ["Female", "Male"])

sex_pie_chart = sex_count.plot(kind="pie", y="", startangle=90)
sex_pie_chart.set_title("Females vs Male Mice Counts")
plt.show()


```


![png](output_16_0.png)



```python
# Generate a pie plot showing the distribution of female versus male mice using pyplot

labels = "Female", "Male"
counts = [female_count, male_count]

plt.pie(counts, labels=labels, startangle=90)
plt.title("Females vs Male Mice Counts")
plt.legend(labels, loc="best", )
plt.show()

```


![png](output_17_0.png)


## Quartiles, Outliers and Boxplots


```python
# Calculate the final tumor volume of each mouse across four of the most promising treatment regimens. 

promising_regimens = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']

drug_filtered = combined_clean[combined_clean["Drug Regimen"].isin(promising_regimens)]
drug_filtered_2 = drug_filtered[drug_filtered["Timepoint"] == 45]
  
```


```python
# Calculate the IQR and quantitatively determine if there are any potential outliers.

quartiles = drug_filtered_2["Tumor Volume (mm3)"].quantile(q=[.25,.5,.75])
lowerq = quartiles[.25]
upperq = quartiles[.75]
iqr = upperq-lowerq

print(f"The lower quartile of tumor volume is: {lowerq}")
print(f"The upper quartile of tumor volume is: {upperq}")
print(f"The interquartile range of tumor volume is: {iqr}")
print(f"The the median of tumor volume is: {quartiles[0.5]} ")

lower_bound = lowerq - (1.5 * iqr)
upper_bound = upperq + (1.5 * iqr)
print(f"Values below {lower_bound} could be outliers.")
print(f"Values above {upper_bound} could be outliers.")


```

    The lower quartile of tumor volume is: 33.48002734
    The upper quartile of tumor volume is: 62.14221369
    The interquartile range of tumor volume is: 28.66218635
    The the median of tumor volume is: 40.1592203 
    Values below -9.513252184999999 could be outliers.
    Values above 105.135493215 could be outliers.



```python
# Generate a box plot of the final tumor volume of each mouse across four regimens of interest

drug_filtered_2.boxplot(by='Drug Regimen', column='Tumor Volume (mm3)', figsize= (6, 5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a212ad250>




![png](output_21_1.png)


## Line and Scatter Plots


```python
# Generate a line plot of time point versus tumor volume for a mouse treated with Capomulin
capomulin = combined_clean[combined_clean['Drug Regimen'] == 'Capomulin']
capomulin_mouse = capomulin[capomulin['Mouse ID'] == 's185']

x_axis = capomulin_mouse['Timepoint']
data = capomulin_mouse['Tumor Volume (mm3)']

plt.plot(x_axis, data)
plt.xlabel('Timepoints')
plt.ylabel('Tumor Volume (mm3)')
plt.title('Capomulin Mouse s185: Tumor Volume over Timepoints')
plt.show()
```


![png](output_23_0.png)



```python
# Generate a scatter plot of mouse weight versus average tumor volume for the Capomulin regimen

capomulin_df = pd.DataFrame(capomulin)
capomulin_groupby = capomulin_df.groupby(by='Mouse ID')

x_values = capomulin_groupby['Weight (g)'].mean()
y_values = capomulin_groupby['Tumor Volume (mm3)'].mean()

plt.scatter(x_values, y_values)
plt.xlabel('Mouse Weight')
plt.ylabel('Avg Tumor Volume')
plt.title('Capomulin Regimen - Mouse Weight vs Tumor Volume')
plt.show()

```


![png](output_24_0.png)


## Correlation and Regression


```python
# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen

(slope, intercept, rvalue, pvalue, stderr) = stats.linregress(x_values, y_values)
regress_values = x_values * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

plt.scatter(x_values,y_values)
plt.plot(x_values,regress_values,"r-")
plt.annotate(line_eq,(18,37),fontsize=15,color="red")
plt.xlabel('Mouse Weight')
plt.ylabel('Avg Tumor Volume')
plt.title('Capomulin Regimen - Mouse Weight vs Tumor Volume')
plt.show()

```


![png](output_26_0.png)



```python

```
