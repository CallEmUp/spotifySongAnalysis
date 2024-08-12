# Spotify Data Analysis Capstone Project

## Project Overview
In this project, I aimed to analyze 52,000 songs from Spotify to gain a better understanding of what factors make music popular and the audio features that define specific genres. The dataset, consisting of data on 52,000 songs (see file directory), was first loaded into a Pandas DataFrame. Initial inspection was conducted to understand the structure and completeness of the data. Missing values were checked, and rows containing any missing values in the key features were removed to ensure the robustness of the statistical tests performed throughout the project. To ensure the reproducibility of the results, I seeded the number 18098787 as a unique identifier.

## Questions and Analysis

### Question 1: Normal Distribution of Song Features
**Objective:** Determine if any of the ten song features (duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo) are reasonably normally distributed.

**Findings:**  
Based on visual inspection and Shapiro-Wilk tests, none of the features showed a clear normal distribution. The Shapiro-Wilk tests yielded extremely low p-values for all features, strongly suggesting that these distributions are not normal.

### Question 2: Relationship Between Song Length and Popularity
**Objective:** Identify if there is a relationship between the duration of a song and its popularity.

**Findings:**  
The Pearson correlation coefficient was approximately -0.055, indicating a very weak negative linear relationship between song duration and popularity. After removing outliers, the correlation coefficient was approximately -0.011, further weakening the already negligible relationship. This suggests that the length of a song has very little to no impact on its popularity on Spotify.

### Question 3: Popularity of Explicit Songs
**Objective:** Determine whether explicitly rated songs are more popular than non-explicit songs.

**Findings:**  
A Mann-Whitney U test showed a statistically significant difference in popularity between explicit and non-explicit songs, with explicit songs tending to be more popular. The p-value was far below the conventional alpha level of 0.05, leading to the rejection of the null hypothesis.

### Question 4: Popularity of Songs in Major vs Minor Keys
**Objective:** Explore whether songs in a major key are more popular than songs in a minor key.

**Findings:**  
A Mann-Whitney U test revealed no statistically significant difference in popularity based on key mode, with a p-value of 0.127. Therefore, major key songs are not significantly more popular than minor key songs.

### Question 5: Relationship Between Energy and Loudness
**Objective:** Investigate whether the energy of a song reflects its loudness.

**Findings:**  
The scatterplot showed a clear positive relationship between energy and loudness, supported by a Pearson correlation coefficient of 0.775. This indicates a strong positive linear relationship between the two variables, substantiating that energy levels largely reflect the loudness of a song.

### Question 6: Best Predictor of Song Popularity
**Objective:** Identify which of the 10 song features predicts popularity the best and evaluate the quality of this model.

**Findings:**  
Linear regression analysis revealed that speechiness had the highest R² score of 0.0052, but this value is still very low, suggesting that none of the features alone is a good predictor of popularity. A Random Forest Regressor performed better, with duration being the best predictor, explaining 64% of the variance in popularity.

### Question 7: Predicting Popularity with All Song Features
**Objective:** Evaluate how well a model using all song features can predict popularity and compare it to the best single-feature model.

**Findings:**  
Multiple linear regression yielded an R² score of 0.0467, indicating that the features alone do not robustly predict popularity. The Random Forest model improved the prediction with an R² score of 0.4060, but it was still lower than the best single-feature models. The combination of features can dilute the impact of any single strong predictor, leading to a lower overall R² score.

### Question 8: Principal Component Analysis (PCA)
**Objective:** Determine the number of meaningful principal components that can be extracted and their variance contribution.

**Findings:**  
PCA results indicated that the first three principal components accounted for 60% of the total variance. Seven principal components were required to capture at least 85% of the variance. A heatmap was used to interpret each component.

### Question 9: Predicting Song Key from Valence
**Objective:** Assess whether valence can predict whether a song is in a major or minor key and identify a better predictor if applicable.

**Findings:**  
Logistic regression using valence alone showed limited predictive accuracy. The ROC-AUC score was 0.5029, indicating no discriminative ability. A better predictor was found by using a combination of features, yielding a higher ROC-AUC of 0.59.

### Question 10: Predicting Classical Music
**Objective:** Compare the predictive power of duration versus principal components in classifying classical music.

**Findings:**  
Principal components proved to be a significantly better predictor of whether a song is classical music compared to using duration alone, with a ROC-AUC score of 0.9564 versus 0.5651 for the duration model.

### Extra Credit: Tempo Differences Between Musical Keys
**Objective:** Explore statistically significant differences in average tempo between different musical keys.

**Findings:**  
An ANOVA test followed by Tukey's HSD test revealed statistically significant differences in average tempo between certain key pairs. This suggests that the key of a song influences its average tempo.

---

This project demonstrates a comprehensive analysis of a large Spotify dataset, highlighting key insights into factors influencing music popularity and genre characteristics.
