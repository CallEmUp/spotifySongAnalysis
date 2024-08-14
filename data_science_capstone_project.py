# -*- coding: utf-8 -*-
"""

# **Intro**

In this project, I aimed to analyze 52,000 songs from Spotify in order to gain a better understanding of what factors make music popular, as well as the audio features that make up specific genres of music.

The dataset, consisting of data on 52,000 songs, was first loaded into a Pandas DataFrame. Initial inspection was conducted to understand the structure and completeness of the data. Following the loading and preparation of the data, Missing values were checked and rows containing any missing values in the key features were removed to ensure the robustness of the statistical tests performed throguhout the project. In this instance, there were no NaN values so all data was able to be used in the project. To ensure reproducibility of the results, I seeded my N number (18098787) in a random number generator as a unique identifier. This not only matters for the specific train/test split or bootstrapping but also to protect my work from potential plagarism that can occur.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Set seed for reproducibility
np.random.seed(18098787)

df = pd.read_csv('spotify52kData.csv')
df.head()

"""# **Question 1**

In this question, I attempted to identify which of the ten song features—duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo—follow a normal distribution. The use of the identification of normally distributed features is important for subsequent statistical analyses that assume normality, such as parametric tests, which may be used later within the project. I employed a plot of the distributions to help with understanding the data and to see the distributions clearly. The results were as follows:
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

features = [
    'duration', 'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

plt.figure(figsize=(20, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 5, i)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(feature)

plt.tight_layout()
plt.show()

"""Based on visual inspection, none of the features shows a clear normal distribution. Some, like tempo, might approximate normality more than others, but each has distinct skewness. For formal analysis, statistical tests like Shapiro-Wilk were be used to assess normality. The test's null hypothesis is that a sample comes from a normally distributed population. The Shapiro-Wilk test is suitable for this purpose, but even small deviations from normality can result in the rejection of the null hypothesis."""

# Performing Shapiro-Wilk test for normality for each feature
normality_test_results = {}
for feature in features:
    stat, p_value = shapiro(df[feature].sample(n=5000, random_state=18098787))
    normality_test_results[feature] = p_value

for feature, p_value in normality_test_results.items():
    print(f"Shapiro-Wilk test p-value for {feature}: {p_value}")

"""The results from the Shapiro-Wilk tests for normality show extremely low p-values for all the features. Duration, Loudness, Speechiness, Acousticness, Instrumentalness, and Liveness have p-values that are 0.0, strongly suggesting that these distributions are not normal. Danceability, Energy, Valence, and Tempo although not zero, have p-values that are very close to zero (well below any conventional alpha level such as 0.05), indicating these features are also not normally distributed. Therefore, the results indicate the rejection of the null hypothesis and none of the song features in the dataset are normally distributed.

# **Question 2**

In this question, I attempted to identify if there is a relationship between the duration of a song and its popularity. Understanding this relationship is important for song production and marketing as it might inform decisions if longer or shorter songs tend to be more popular. To explore this relationship, I employed a scatterplot to visually assess the correlation between these two variables. A scatterplot was used as it is one of the most straightforward methods to visualize potential correlations or trends between two quantitative variables. To aid alongside the visual analysis, a statistical test was performed to quantify the strength and direction of the relationship between song duration and popularity. Pearson’s correlation coefficient was calculated as it measures the linear correlation between two variables, providing both the strength and direction of the relationship.
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='popularity', data=df, palette = 'pastel')
plt.title('Relationship Between Song Duration and Popularity')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity Score')
plt.show()

correlation = df['duration'].corr(df['popularity'])
print("Pearson correlation coefficient between song duration and popularity:", correlation)

"""The plot shows a wide dispersion of points without a clear linear trend. There is considerable variability in popularity scores across the range of song durations.
There does not appear to be a strong visual correlation between song length and popularity.

Pearson Correlation Coefficient:
The computed correlation coefficient is approximately -0.055. This value is close to zero and negative, indicating a very weak negative linear relationship between song duration and popularity. This implies that song length is not a strong predictor of its popularity on Spotify. The length of a song does not significantly influence how popular it will be, at least not in a linear manner.

However, there is a possible issue with including outliers to see the relationship. In order to try and remove the potential error that these outliers pose on the correlation, I first identified outliers as data points that lie beyond 1.5 time sthe interquartile range from the quartiles. Once removing the outliers, I performed another analysis via scatterplot and Pearson correlation coefficient to see if the relationship between duration and popularity changes significantly.
"""

Q1 = df['duration'].quantile(0.25)
Q3 = df['duration'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers as those beyond 1.5 times the IQR from Q1 and Q3
outliers = ((df['duration'] < (Q1 - 1.5 * IQR)) | (df['duration'] > (Q3 + 1.5 * IQR)))
filtered_data = df[~outliers]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='duration', y='popularity', data=filtered_data)
plt.title('Relationship Between Song Duration and Popularity (Without Outliers)')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity Score')
plt.grid(True)
plt.show()

new_correlation = filtered_data['duration'].corr(filtered_data['popularity'])
print("New Pearson correlation coefficient (without outliers):", new_correlation)

"""The scatterplot still shows a wide dispersion of popularity scores across the range of song durations, but without extreme values in duration. There is no obvious linear trend even with outliers removed.

The new correlation coefficient is approximately -0.011, which is even closer to zero than before. This change indicates an even weaker negative linear relationship between song duration and popularity after removing outliers.

Removing outliers has slightly altered the correlation coefficient, further weakening the already negligible relationship between song duration and popularity. This suggests that the length of a song, within a more typical range of durations, has very little to no impact on how popular it will be on Spotify, but shorter songs tend to lean to be slightly more popular.

# **Question 3**

In this analysis, I sought to determine whether explicitly rated songs are more popular than songs that are not explicit. The popularity of explicit content can influence decisions related to production, marketing, and distribution, particularly in targeting demographics that either prefer or avoid explicit material. This involved comparing the popularity of two groups: songs with explicit content and songs without explicit content. To do this, I first looked into the data and noticed a large amount of songs had 0 as their popularity (6374 to be exact). A popularity of 0 does not make much sense, as even songs with the least popularity would scratch at least 1. Because of this irregularity, I removed all songs that had a popularity of 0, as it was most likely a substitute to a NaN value. After the cleaning of the data, I created boxplots to visually compare the distribution of popularity scores between explicit and non-explicit songs and then furthered that with the use of the Mann-Whitney U test. This non-parametric test is very helpful, as the distributions are not normally distributed. The null hypothesis for this analysis is that explicit and non-explicit materials have the same popularity.
"""

num_popularity_zero = df[df['popularity'] == 0].shape[0]

print(f"Number of rows with popularity score of 0: {num_popularity_zero}")

dfc = df[df['popularity'] > 0]

from scipy.stats import mannwhitneyu

plt.figure(figsize=(8, 6))
sns.boxplot(x='explicit', y='popularity', data=dfc, palette=['skyblue', 'lightgreen'])
plt.title('Popularity Distribution by Explicit Content')
plt.xlabel('Explicit Content')
plt.ylabel('Popularity Score')
p_value_annotation = f"p = {p_value:.2e}"
plt.text(0.5, max(dfc['popularity']) - 5, p_value_annotation, horizontalalignment='center', color='red')
plt.show()

explicit_popularity = dfc[dfc['explicit'] == True]['popularity']
non_explicit_popularity = dfc[dfc['explicit'] == False]['popularity']

u_statistic, p_value = mannwhitneyu(explicit_popularity, non_explicit_popularity)
print("Mann-Whitney U statistic:", u_statistic)
print("P-value:", p_value)

"""The very small p-value of 5.93e -18 (far below the conventional alpha level of 0.05) indicates a statistically significant difference in popularity between explicit and non-explicit songs, with explicit songs tending to be more popular, therfore we can reject the null hypothesis.

# **Question 4**

In this analysis, I explored whether songs in a major key are more popular than songs in a minor key. This involved comparing the popularity scores of two groups: songs in major keys versus songs in minor keys. I first started by creating a boxplot to visually compare the distribution of popularity scores between songs in major vs minor keys. I then coupled it with a Mann-Whitney U test to see if there was a statistically significant difference in popularity based on key mode. It is important to note for this problem, the hypotheses are as follows:

Null - There is no difference in popularity between songs in major key and songs in minor key.

Alternative - There is a statistical significant difference in popularity based on key mode.
"""

plt.figure(figsize=(8, 6))
box_plot = sns.boxplot(x='mode', y='popularity', data=dfc, palette=['skyblue', 'lightgreen'])
plt.title('Popularity Distribution by Musical Key')
plt.xlabel('Musical Key (0: Minor, 1: Major)')
plt.ylabel('Popularity Score')

major_popularity = dfc[dfc['mode'] == 1]['popularity']
minor_popularity = dfc[dfc['mode'] == 0]['popularity']

u_statistic, p_value = mannwhitneyu(major_popularity, minor_popularity, alternative='two-sided')

p_value_annotation = f"p = {p_value:.2e}"
plt.text(0.5, max(dfc['popularity']) - 5, p_value_annotation, horizontalalignment='center', color='red')
plt.show()

print("Mann-Whitney U statistic:", u_statistic)
print("P-value:", p_value)

"""Following the result of the two-tailed test, it became clear that there was not a statistically significant difference between popularity based key mode, as displayed by the p value of 0.127. Therefore, due to the p value being >0.05, we fail to reject the null hypothesis and major key songs are not significantly more popular than minor key songs.

# **Question 5**

For this analysis, which investigates whether the energy of a song reflects its loudness, I performed a scatterplot to visually examine the relationship between energy and loudness, and calculated the Pearson correlation coefficient to quantify the relationship. The scatterplot allowed visualization of the relationship between energy and loudness to see if there's a linear trend or any pattern. The Pearson correlation coefficient helped in hand with the scatterplot to determine how strongly the energy and loudness metrics are correlated. The results are as follows:
"""

plt.figure(figsize=(10, 6))
sns.scatterplot(x='energy', y='loudness', data=df)
plt.title('Relationship Between Energy and Loudness')
plt.xlabel('Energy')
plt.ylabel('Loudness (dB)')
plt.show()

correlation = df['energy'].corr(df['loudness'])
print("Pearson correlation coefficient between energy and loudness:", correlation)

"""Based on the plot, the scatterplot shows a clear positive relationship between energy and loudness of songs. As the energy increases, the loudness also tends to increase. The plot displays a dense clustering of points, particularly towards higher levels of energy and loudness, suggesting a strong linear trend in these regions. This analysis is backed with the correlation coefficient of 0.775, which further indicates the strong positive linear relationship between the two variables. Based on the data provided, energy levels largely reflect the loudness of a song.

# **Question 6**

For this analysis, where the task was to determine which of the 10 song features from question 1 predicts popularity the best, I employed a regression analysis. I used linear regression analysis for each of the 10 features to receive R^2 values. From the R^2 values, I compared each feature's ability to predict the popularity of a song and identify which feature serves as the best predictor. Understanding the feature that most influences song popularity can guide artists and producers in optimizing song characteristics to enhance their appeal. However, before the linear regression analysis took place, I removed the outliers within the features based on the basis that an outlier is a value that resides outside the parameters of below Q1 - 1.5IQR or above Q3 + 1.5IQR. The results are as follows:
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

r2_scores = {}

model = LinearRegression()

def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

for feature in features:
    fdf = remove_outliers(df, feature)
    X = fdf[[feature]]  # Predictor
    y = fdf['popularity']  # Response variable

    model.fit(X, y)

    y_pred = model.predict(X)

    r2_scores[feature] = r2_score(y, y_pred)

plt.figure(figsize=(20, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 5, i)
    sns.scatterplot(x=fdf[feature], y=fdf['popularity'], alpha=0.3)
    sns.regplot(x=fdf[feature], y=fdf['popularity'], scatter=False, color='red')
    plt.title(f"{feature} (R²={r2_scores[feature]:.4f})")
    plt.xlabel(feature)
    plt.ylabel('Popularity')
plt.tight_layout()
plt.show()

print("R^2 values for each feature predicting popularity:")
sorted_scores = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
for feature, score in sorted_scores:
    print(f"{feature}: R² = {score:.4f}")

"""The red line depicts the perfect predictor line. Despite all of the features avaliable, no feature had a high R^2 score. Of the features, speechiness appeared to have the largest R^2 score of 0.0052, meaning that speechiness alone explains very little of the variability in popularity. The low scores are influenced by a multitude of outside factors that were not analyzed. Some of these characteristics include marketing efforts, public relations, artist popularity, and cultural/social factors. Not only are those factors contributors, but also the data may not be linear, resulting in low R^2 values for linear regression. To test this, I conducted a Random Forest Regressor because it can handle non-linear relationships better. The results of the regressor are as follows:"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

r2_scores = {}

for feature in features:
    X = fdf[[feature]]  # Predictor
    y = fdf['popularity']  # Response variable

    model = RandomForestRegressor(n_estimators=100, random_state=18098787)
    model.fit(X, y)

    y_pred = model.predict(X)

    r2_scores[feature] = r2_score(y, y_pred)

print("R^2 values for each feature predicting popularity:")
sorted_scores = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)
for feature, score in sorted_scores:
    print(f"{feature}: R² = {score:.4f}")

plt.figure(figsize=(20, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(2, 5, i)
    sns.scatterplot(x=fdf[feature], y=fdf['popularity'], alpha=0.3)
    sns.lineplot(x=fdf[feature], y=y_pred, color='red')  # Changed to lineplot for clarity with RF predictions
    plt.title(f"{feature} (R²={r2_scores[feature]:.4f})")
    plt.xlabel(feature)
    plt.ylabel('Popularity')
plt.tight_layout()
plt.show()

"""The red line in each plot represents the predicted popularity based on the Random Forest model's predictions for each specific feature, plotted against the actual feature values. By overlaying this line on top of the scatterplot of actual data points, you can see where the model predictions align with actual outcomes and where they diverge."""

# Converting R² values to a list of tuples for sorting and plotting
sorted_r2 = sorted(r2_scores.items(), key=lambda x: x[1], reverse=True)

# Separating the data for plotting
features, r2_values = zip(*sorted_r2)

plt.figure(figsize=(12, 6))
plt.bar(features, r2_values, color='blue')
plt.xlabel('Features')
plt.ylabel('R² Score')
plt.title('Comparison of R² Scores for Different Features')
plt.xticks(rotation=45)
plt.show()

"""The Random Forest Regressor excelled in comparison to the linear regression previously used. The random forest regressor yielded R^2 scores ranging from 0.05 to 0.64. The 'best' model was duration, as duration explains 64% of the variance in popularity. Closely following duration was tempo, yielding a score of .61. These R^2 scores show that the random forest regressor model is a fairly good fit for the data.

# **Question 7**

Building on the findings from Question 6, this analysis tests the hypothesis that a combination of features might explain more variability in popularity due to potential synergistic effects. Multiple linear regression was used to model the interactions. The results are as follows:
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = fdf[features]
y = fdf['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18098787)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_score_combined = r2_score(y_test, y_pred)
print(f"Combined features model R² score: {r2_score_combined:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs. Predicted Popularity Scores')
plt.show()

"""The red line in the visual represents a perfect predictor. The use of the multiple linear regression fails to explain much of the variance in the popularity scores, as shown with the R^2 score of 0.0467. This outcome suggests that while the features included might have some predictive power, they are insufficient to robustly predict popularity on their own. This can most likely be because of how complex popularity is, or because the relationship between features and popularity is not linear. In order to test this theory once again, I used a Random Forest Regressor, which can handle non-linearity much better than linear regression. The results are as follows:"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

random_forest_model = RandomForestRegressor(n_estimators=100, random_state=18098787)
random_forest_model.fit(X_train, y_train)
y_pred_rf = random_forest_model.predict(X_test)

r2_score_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest model R² score: {r2_score_rf:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.title('Actual vs. Predicted Popularity Scores')
plt.show()

"""The Random Forest model with combined features demonstrates an R² score of 0.4060, which, while showing an improvement over linear regression models by capturing more complex interactions, is still approximately 0.24 points lower than the R² scores achieved by the best-performing individual features, such as duration (R² = 0.6442) and tempo (R² = 0.6059), from the previous analysis in question 6. This relative underperformance of the combined model can be accounted for by its ability to incorporate multiple features simultaneously, therfore adding complexity by integrating various song aspects and their interactions. The model provides a wider understanding of song popularity, even though it does not reach the high predictive power of the best single-feature models. The combination of features can potentially dilute the impact of any single strong predictor by averaging out its effects with less predictive features, therefore resulting in a lower overall R² score than models using only the most predictive features.

# **Question 8**

For this analysis, I aimed to explore the effectiveness of Principal Component Analysis (PCA) in reducing the dimensionality of the dataset consisting of ten song features—duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. The goal was to determine how many meaningful principal components can be extracted and what proportion of the variance these components account for. PCA is a powerful tool for dimensionality reduction, used to simplify the data while retaining as much information as possible. It transforms the original variables into a new set of variables, which are orthogonal (uncorrelated) and ranked according to the variance they capture from the data. I performed PCA to extract the principal components and then looked into the proportion of variance explained by each component. The results are as follows:
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = df[features]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(features))
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(features) + 1), explained_variance_ratio, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(features) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component Index')
plt.title('Explained Variance by PCA Components')
plt.legend(loc='best')
plt.show()

n_components_85 = next(i for i, total_var in enumerate(cumulative_variance, start=1) if total_var >= 0.85)

print(f"Number of components needed to explain at least 85% of the variance: {n_components_85}")
print("Explained variance by each component: ", explained_variance_ratio)
print("Cumulative variance explained by components: ", cumulative_variance)

"""The PCA results indicate that the first three principal components account for 60% of the total variance in the dataset (0.273, 0.162, 0.138). To capture at least 85% of the total variance, we found that seven principal components are required. This reduction from ten original features to seven principal components allows for a more efficient representation of the dataset without significant loss of information, enabling more effective subsequent analyses. The one issue with the PCA is that their interpretation is lost. In order to try and maintain an understanding of each component, I loaded them onto a heatmap as follows:"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(df[features])
pca = PCA()
components = pca.fit_transform(X)

loadings = pd.DataFrame(pca.components_.T, columns=['PC' + str(i) for i in range(1, len(features) + 1)], index=features)

plt.figure(figsize=(12, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title('PCA Component Loadings')
plt.show()

print(loadings)

"""From the heat map, the interpretation of the components are:

PC1: High values might indicate songs that are quieter and more acoustic, while low values suggest songs that are loud and energetic

PC2: High values indicate less danceable, less joyful music with more instrumental elements.

PC3: High values suggest songs that likely feature live performance aspects and more speech

PC4: Highly influenced by the duration of the songs and inversely by tempo, suggesting a dimension where longer songs tend to have slower tempos

PC5: Contrasts fast-tempo songs against acoustic characteristics, indicating that songs with a higher score are faster and less acoustic

PC6: High values could indicate longer songs that are less instrumental, potentially pointing towards more vocal or lyrical content

PC7: High values suggest a greater emphasis on speechiness in the track, whereas low values could indicate a focus on live performance attributes

PC8: High values indicate songs that are emotionally positive but less danceable, which could characterize certain genres like ballads or slower, more melodic music.

PC9: Higher values might be associated with songs that are less acoustic and quieter, potentially indicating more subdued or softer music styles.

PC10: Songs with high values may be louder but have less energetic content.

# **Question 9**

For this analysis, I aimed to determine if song valence (a measure of musical positiveness) can predict whether a song is in a major or minor key. Understanding this relationship can provide insights into how the emotional content of music correlates with musical theory concepts like key modes. In order to do this, I used a logistic regression to model the relationship between valence and key note. The results are as follows:
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X = df[['valence']]
y = df['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18098787)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC score: {roc_auc:.4f}")


valence_range = np.linspace(X_test['valence'].min(), X_test['valence'].max(), 300).reshape(-1, 1)
predicted_probabilities = model.predict_proba(valence_range)[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(valence_range, predicted_probabilities, color='red', label='Predicted Probability')
plt.scatter(X_test['valence'], y_test, alpha=0.1, label='Actual Data')
plt.xlabel('Valence')
plt.ylabel('Probability of Major Key')
plt.title('Predicted Probability of Major Key by Valence')
plt.legend()
plt.show()

"""The model does has a precision of 0.62, implying that when the model predicts a song is in a major key, it is correct 62% of the time. At the surface level, that seems like the model is working better than randomly guessing, however the major key recall is 1.00 and all of the minor key classification reports are 0. THe confusion matrix predicted all songs as being in a major key (1), as shown by the zeros in the first column (actual minor) and zeros in the first row of the second column (predicted minor). And the ROC-AUC score of 0.5029 indicates a model with no discriminative ability whatsoever between major and minor keys, essentially performing no better than random guessing. The model may have an issue due to the imbalance of major and minor keys within the data set. In order to investigate if this was the case, I used a resampling technique called SMOTE (Synthetic Minority Over-sampling Technique). This will synthesize new minority class instances rather than duplicating existing ones until it is balanced. The results are as follows:"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

X = df[['valence']]
y = df['mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18098787)

smote = SMOTE(random_state=18098787)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

"""The model's performance, as captured by the ROC-AUC score, was found to be 0.51, indicating a marginal improvement over random guessing, but still showing limited predictive accuracy. For songs in a minor key (labelled as 0 in the confusion matrix), the model achieved a precision of 0.38 and a recall of 0.52, resulting in an F1-score of 0.44. This suggests that while the model is relatively moderate at identifying true minor key songs, it lacks precision and falsely identifies major key songs as minor key songs quite frequently. For songs in a major key (labelled as 1), the precision was somewhat higher at 0.62 with a recall of 0.48, leading to an F1-score of 0.54. This indicates that while the model is more precise when predicting major key songs, it fails to identify just over half of them correctly. Based on this information, it appears that it is feasible but not very effective to predict the songs key from valence by itself. I wanted to find a better predictor, so I took a look at the features that may elude to a better model. Features I selected were: energy, danceability, acousticness, instrumentalness, and loudness. I followed the same method as previous and added a combined feature predictor (a combination of the features I hand selected), and received the following results:"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

features = ['energy', 'danceability', 'acousticness', 'instrumentalness', 'loudness']
results = []

def evaluate_model(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18098787)

    smote = SMOTE(random_state=18098787)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression()
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'feature': feature_name,
        'roc_auc': roc_auc,
        'report': report,
        'confusion_matrix': cm
    })

    print(f"Feature: {feature_name}")
    print(f"ROC-AUC score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {feature_name}')
    plt.legend(loc="lower right")
    plt.show()

for feature in features:
    evaluate_model(df[[feature]], df['mode'], feature)

evaluate_model(df[features], df['mode'], 'combined_features')

best_feature = max(results, key=lambda x: x['roc_auc'])
print(f"Best predictor: {best_feature['feature']} with ROC-AUC score: {best_feature['roc_auc']:.4f}")
print("Best feature's Classification Report:")
print(best_feature['report'])
print("Best feature's Confusion Matrix:")
print(best_feature['confusion_matrix'])

"""As observed from the results, all of the selected features yielded higher ROC-AUC values than valence, thus indicating that they are all better predictors. However, the "best" predictor from the data analysis was the combined feature predictor, with the ROC-AUC of 0.59.

# **Question 10**

For this analysis, I aimed to determine if song duration or principal components (extracted from various song features in question 8) can predict whether a song is classified as classical music. Understanding this relationship can provide insights into how different musical characteristics contribute to the classification of music genres, particularly classical music. In order to do this, I used logistic regression models to assess the predictive power of both duration and principal components. The results are as follows:
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

df['classical'] = df['track_genre'].apply(lambda x: 1 if x.lower() == 'classical' else 0)

features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = df[features]
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=len(features))
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

n_components_85 = next(i for i, total_var in enumerate(cumulative_variance, start=1) if total_var >= 0.85)

principal_components = pca.transform(X_scaled)[:, :n_components_85]
df_pca = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(n_components_85)])
df_pca['classical'] = df['classical']

def evaluate_model(X, y, feature_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18098787)

    smote = SMOTE(random_state=18098787)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression()
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_probs)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        'feature': feature_name,
        'roc_auc': roc_auc,
        'report': report,
        'confusion_matrix': cm
    })

    print(f"Feature: {feature_name}")
    print(f"ROC-AUC score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {feature_name}')
    plt.legend(loc="lower right")
    plt.show()

results = []

evaluate_model(df[['duration']], df['classical'], 'duration')

evaluate_model(df_pca.drop('classical', axis=1), df_pca['classical'], 'principal_components')

best_feature = max(results, key=lambda x: x['roc_auc'])
print(f"Best predictor: {best_feature['feature']} with ROC-AUC score: {best_feature['roc_auc']:.4f}")
print("Best feature's Classification Report:")
print(best_feature['report'])
print("Best feature's Confusion Matrix:")
print(best_feature['confusion_matrix'])

"""Based on the analysis, principal components are a significantly better predictor of whether a song is classical music compared to using duration alone. The ROC-AUC score for the principal components model was 0.9564, which is much higher than the 0.5651 for the duration model. The classification report for the principal components model shows a substantial improvement in recall for the classical music class (0.90) compared to the duration model (0.00). Additionally, the principal components model has a much higher precision for the classical music class (0.14) than the duration model (0.00), although it is still quite low. The ROC curve for the principal components model illustrates a far superior performance, with an area under the curve close to 1, indicating excellent discriminative ability. In contrast, the ROC curve for the duration model is much closer to the diagonal line, indicating performance close to random guessing. These results suggest that the combination of various song features captured by principal components provides a more comprehensive understanding and significantly better predictive power for classifying classical music compared to using duration alone.

# **Extra Credit**

For this analysis, I aimed to determine whether there are statistically significant differences in the average tempo of songs between different musical keys. Understanding these differences can provide insights into how key selection might influence the tempo of a composition, which is valuable for both music theory and practical composition. To achieve this, I used an Analysis of Variance (ANOVA) test followed by Tukey's Honestly Significant Difference (HSD) test for post-hoc analysis. ANOVA is useful in this context as it allows us to determine whether there are any statistically significant differences in the means of multiple groups (in this case, the different keys), and Tukey's HSD test helps to identify which specific pairs of keys differ. The results are as follows:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

columns_of_interest = ['tempo', 'key']
df = df[columns_of_interest].dropna()

Q1 = df['tempo'].quantile(0.25)
Q3 = df['tempo'].quantile(0.75)
IQR = Q3 - Q1
filter = (df['tempo'] >= (Q1 - 1.5 * IQR)) & (df['tempo'] <= (Q3 + 1.5 * IQR))
df = df.loc[filter]

# Perform ANOVA
anova_result = stats.f_oneway(*[df[df['key'] == k]['tempo'] for k in df['key'].unique()])

print(f"ANOVA result: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")

# Post-hoc analysis with Tukey's HSD
tukey_result = pairwise_tukeyhsd(endog=df['tempo'], groups=df['key'], alpha=0.05)

print(tukey_result)

tukey_result.plot_simultaneous()
plt.title('Tukey HSD Post-Hoc Test for Tempo Differences Between Keys')
plt.xlabel('Tempo')
plt.ylabel('Key')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='key', y='tempo', data=df)
plt.title('Boxplot of Tempo by Key')
plt.xlabel('Key')
plt.ylabel('Tempo')
plt.show()

"""Based on the analysis, there are statistically significant differences in the average tempo of songs between different keys. The ANOVA test resulted in an F-statistic of 8.8028 with a p-value of 7.84e-16, indicating that the variations in tempo across keys are not due to random chance. The post-hoc analysis using Tukey's HSD test further identified which specific keys have significant differences in average tempo. The results showed that certain key pairs have significantly different average tempos, as indicated by the 'True' values in the reject column of the Tukey HSD output. These significant differences highlight that some keys are associated with faster or slower tempos compared to others.Some of these include: keys C and E, C# and A#, D and F#, D# and E, and F and G. The Tukey HSD plot visually illustrates these significant differences, with some keys showing clear separation in average tempo. Additionally, the boxplot of tempo by key provides a visual representation of the distribution of tempos across different keys, supporting the statistical findings. These results suggest that the key of a song does influence its average tempo, and certain keys tend to be associated with faster or slower tempos."""
