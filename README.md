# Spotify Song Analysis

In this project, I aimed to analyze 52,000 songs from Spotify in order to gain a better understanding of what factors make music popular, as well as the audio features that make up specific genres of music. The dataset, consisting of data on 52,000 songs, was first loaded into a Pandas DataFrame. Initial inspection was conducted to understand the structure and completeness of the data. Following the loading and preparation of the data, missing values were checked and rows containing any missing values in the key features were removed to ensure the robustness of the statistical tests performed throughout the project. In this instance, there were no NaN values so all data was able to be used in the project. To ensure reproducibility of the results, I seeded the number 18098787 in a random number generator as a unique identifier. Throughout the project, I aimed to answer the following 11 questions: 

1) Consider the 10 song features duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valenceand tempo. Are any of these features reasonably distributed normally?
2) Is there a relationship between song length and popularity of a song?
3) Are explicitly rated songs more popular than songs that are not explicit?
4) Are songs in major key more popular than songs in minor key?
5) Energy is believed to largely reflect the “loudness” of a song. Can I substantiate that?
6) Which of the 10 individual (single) song features from question 1 predicts popularity best?
7) Building a model that uses *all* of the song features from question 1, how well can it predict popularity now?
8) When considering the 10 song features above, how many meaningful principal components can be extracted?
9) Can I predict whether a song is in major or minor key from valence?
10) Which is a better predictor of whether a song is classical music –duration or the principal components extracted in question 8?
11) What is something interesting about this dataset that is not trivial and not already part of an answer (implied or explicitly) to the previous questions?

## Description of the Dataset
This dataset consists of data on 52,000 songs that were randomly picked from a variety of genres sorted in alphabetic order (a as in “acoustic” to h as in “hiphop”). For the purposes of this analysis, you can assume that the data for one song are independent for data from other songs.This data is stored in the file “spotify52kData.csv”, as follows:

Row 1: Column headers

Row 2-52001: Specific individual songs

Column 1: songNumber–the track ID of the song, from 0 to 51999.

Column 2: artist(s)–the artist(s) who are credited with creating the song.

Column 3: album_name–the name of the album

Column 4: track_name–the title of the specific track corresponding to the track ID

Column 5: popularity–this is an important metric provided by spotify, an integer from 0 to 100, where a higher number corresponds to a higher number of plays on spotify. 

Column 6: duration–this is the duration of the song in ms. A ms is a millisecond. There are a thousand milliseconds in a second and 60 seconds in a minute. 

Column 7: explicit–this is a binary (Boolean) categorical variable. If it is true, the lyrics of the track contain explicit language, e.g. foul language, swear words or otherwise content that some consider to be indecent. 

Column 8: danceability–this is an audio feature provided by the Spotify API.Ittries to quantify how easy it is to dance to the song (presumably capturing tempoand beat), and varies from 0 to 1.

Column 9: energy-this is an audio feature provided by the Spotify API. It tries to quantify how “hard” a song goes. Intense songs have more energy, softer/melodic songs lower energy, it varies from 0 to 1.

Column 10: key–what is the key of the song, from A to G# (mapped to categories 0 to 11).

Column 11: loudness–average loudness of a track in dB (decibels)

Column 12: mode–this is a binary categorical variable. 1 = song is in major, 0 –song is in minor

Column 13: speechiness–quantifies how much of the song is spoken, varying from 0 (fully instrumental songs) to 1 (songs that consist entirely of spoken words). 

Column 14: acousticness–varies from 0 (song contains exclusively synthesized sounds) to 1 (song features exclusively acoustic instruments like acoustic guitars, pianos or orchestral instruments)

Column 15: instrumentalness–basically the inverse of speechiness, varying from 1 (for songs without any vocals) to 0.

Column 16: liveness-this is an audio feature provided by the Spotify API. It tries to quantify how likely the recording was live in front of an audience (values close to 1) vs. how likely it was recorded in a studio without a live audience (values close to 0).

Column 17: valence-this is an audio feature provided by the Spotify API. It tries to quantify how uplifting a song is. Songs with a positive mood =close to 1 and songs with a negative mood =close to 0

Column 18: tempo–speed of the song in beats per minute (BPM)

Column 19: time_signature–how many beats there are in a measure (usually 4 or 3)

Column 20: track_genre–genre assigned by spotify, e.g. “blues” or “classica


## Question 1: 
Consider the 10 song features duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. Are any of these features reasonably distributed normally?

In this question, I attempted to identify which of the ten song features—duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo—follow a normal distribution. The use of the identification of normally distributed features is important for subsequent statistical analyses that assume normality, such as parametric tests, which may be used later within the project. I employed a plot of the distributions to help with understanding the data and to see the distributions clearly. The results were as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q1.png">

Based on visual inspection, none of the features shows a clear normal distribution. Some, like tempo, might approximate normality more than others, but each has distinct skewness. For formal analysis, statistical tests like Shapiro-Wilk were be used to assess normality. The test's null hypothesis is that a sample comes from a normally distributed population. The Shapiro-Wilk test is suitable for this purpose, but even small deviations from normality can result in the rejection of the null hypothesis.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q1.2.png">

The results from the Shapiro-Wilk tests for normality show extremely low p-values for all the features. Duration, Loudness, Speechiness, Acousticness, Instrumentalness, and Liveness have p-values that are 0.0, strongly suggesting that these distributions are not normal. Danceability, Energy, Valence, and Tempo although not zero, have p-values that are very close to zero (well below any conventional alpha level such as 0.05), indicating these features are also not normally distributed. Therefore, the results indicate the rejection of the null hypothesis and none of the song features in the dataset are normally distributed.

## Question 2:
Is there a relationship between song length and popularity of a song?

In this question, I attempted to identify if there is a relationship between the duration of a song and its popularity. Understanding this relationship is important for song production and marketing as it might inform decisions if longer or shorter songs tend to be more popular. To explore this relationship, I employed a scatterplot to visually assess the correlation between these two variables. A scatterplot was used as it is one of the most straightforward methods to visualize potential correlations or trends between two quantitative variables. To aid alongside the visual analysis, a statistical test was performed to quantify the strength and direction of the relationship between song duration and popularity. Pearson’s correlation coefficient was calculated as it measures the linear correlation between two variables, providing both the strength and direction of the relationship.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q2.png">

The plot shows a wide dispersion of points without a clear linear trend. There is considerable variability in popularity scores across the range of song durations. There does not appear to be a strong visual correlation between song length and popularity. Pearson Correlation Coefficient: The computed correlation coefficient is approximately -0.055. This value is close to zero and negative, indicating a very weak negative linear relationship between song duration and popularity. This implies that song length is not a strong predictor of its popularity on Spotify. The length of a song does not significantly influence how popular it will be, at least not in a linear manner. 

However, there is a possible issue with including outliers to see the relationship. In order to try and remove the potential error that these outliers pose on the correlation, I first identified outliers as data points that lie beyond 1.5 times the interquartile range from the quartiles. Once removing the outliers, I performed another analysis via scatterplot and Pearson correlation coefficient to see if the relationship between duration and popularity changes significantly.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q2.2.png">

The scatterplot still shows a wide dispersion of popularity scores across the range of song durations, but without extreme values in duration. There is no obvious linear trend even with outliers removed. The new correlation coefficient is approximately -0.011, which is even closer to zero than before. This change indicates an even weaker negative linear relationship between song duration and popularity after removing outliers. Removing outliers has slightly altered the correlation coefficient, further weakening the already negligible relationship between song duration and popularity. This suggests that the length of a song, within a more typical range of durations, has very little to no impact on how popular it will be on Spotify, but shorter songs tend to lean to be slightly more popular.

## Question 3:
Are explicitly rated songs more popular than songs that are not explicit?

In this analysis, I sought to determine whether explicitly rated songs are more popular than songs that are not explicit. The popularity of explicit content can influence decisions related to production, marketing, and distribution, particularly in targeting demographics that either prefer or avoid explicit material. This involved comparing the popularity of two groups: songs with explicit content and songs without explicit content. To do this, I first looked into the data and noticed a large number of songs had 0 as their popularity (6374 to be exact). A popularity of 0 does not make much sense, as even songs with the least popularity would scratch at least 1. Because of this irregularity, I removed all songs that had a popularity of 0, as it was most likely a substitute to a NaN value. After the cleaning of the data, I created boxplots to visually compare the distribution of popularity scores between explicit and non-explicit songs and then furthered that with the use of the Mann-Whitney U test. This non-parametric test is very helpful, as the distributions are not normally distributed. The null hypothesis for this analysis is that explicit and non-explicit materials have the same popularity.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q3.png">

The very small p-value of 5.93e -18 (far below the conventional alpha level of 0.05) indicates a statistically significant difference in popularity between explicit and non-explicit songs, with explicit songs tending to be more popular, therefore we can reject the null hypothesis.

## Question 4:
Are songs in major key more popular than songs in minor key?

In this analysis, I explored whether songs in a major key are more popular than songs in a minor key. This involved comparing the popularity scores of two groups: songs in major keys versus songs in minor keys. I first started by creating a boxplot to visually compare the distribution of popularity scores between songs in major vs minor keys. I then coupled it with a Mann-Whitney U test to see if there was a statistically significant difference in popularity based on key mode. It is important to note for this problem, the hypotheses are as follows:

Null - There is no difference in popularity between songs in major key and songs in minor key.

Alternative - There is a statistically significant difference in popularity based on key mode.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q4.png">

Following the result of the two-tailed test, it became clear that there was not a statistically significant difference between popularity based key mode, as displayed by the p value of 0.127. Therefore, due to the p value being >0.05, we fail to reject the null hypothesis and major key songs are not significantly more popular than minor key songs.

## Question 5:
Energy is believed to largely reflect the “loudness” of a song. Can I substantiate that?

For this analysis, which investigates whether the energy of a song reflects its loudness, I performed a scatterplot to visually examine the relationship between energy and loudness, and calculated the Pearson correlation coefficient to quantify the relationship. The scatterplot allowed visualization of the relationship between energy and loudness to see if there is a linear trend or any pattern. The Pearson correlation coefficient helped in hand with the scatterplot to determine how strongly the energy and loudness metrics are correlated. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q5.png">

Based on the plot, the scatterplot shows a clear positive relationship between energy and loudness of songs. As the energy increases, the loudness also tends to increase. The plot displays a dense clustering of points, particularly towards higher levels of energy and loudness, suggesting a strong linear trend in these regions. This analysis is backed with the correlation coefficient of 0.775, which further indicates the strong positive linear relationship between the two variables. Based on the data provided, energy levels largely reflect the loudness of a song.

## Question 6:
Which of the 10 individual (single) song features from question 1 predicts popularity best?

For this analysis, where the task was to determine which of the 10 song features from question 1 predicts popularity the best, I employed a regression analysis. I used linear regression analysis for each of the 10 features to receive R^2 values. From the R^2 values, I compared each feature's ability to predict the popularity of a song and identify which feature serves as the best predictor. Understanding the feature that most influences song popularity can guide artists and producers in optimizing song characteristics to enhance their appeal. However, before the linear regression analysis took place, I removed the outliers within the features based on the basis that an outlier is a value that resides outside the parameters of below Q1 - 1.5IQR or above Q3 + 1.5IQR. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q6.png">

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q6.2.png">

The red line depicts the perfect predictor line. Despite all of the features available, no feature had a high R^2 score. Of the features, speechiness appeared to have the largest R^2 score of 0.0052, meaning that speechiness alone explains very little of the variability in popularity. The low scores are influenced by a multitude of outside factors that were not analyzed. Some of these characteristics include marketing efforts, public relations, artist popularity, and cultural/social factors. Not only are those factors contributors, but also the data may not be linear, resulting in low R^2 values for linear regression. To test this, I conducted a Random Forest Regressor because it can handle non-linear relationships better. The results of the regressor are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q6.3.png">

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q6.4.png">

The red line in each plot represents the predicted popularity based on the Random Forest model's predictions for each specific feature, plotted against the actual feature values. By overlaying this line on top of the scatterplot of actual data points, you can see where the model predictions align with actual outcomes and where they diverge.

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q6.5.png">

The Random Forest Regressor excelled in comparison to the linear regression previously used. The random forest regressor yielded R^2 scores ranging from 0.05 to 0.64. The 'best' model was duration, as duration explains 64% of the variance in popularity. Closely following duration was tempo, yielding a score of .61. These R^2 scores show that the random forest regressor model is a fairly good fit for the data.

## Question 7:
Building a model that uses *all* of the song features from question 1, how well can it predict popularity now?

Building on the findings from Question 6, this analysis tests the hypothesis that a combination of features might explain more variability in popularity due to potential synergistic effects. Multiple linear regression was used to model the interactions. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q7.png">

The red line in the visual represents a perfect predictor. The use of the multiple linear regression fails to explain much of the variance in the popularity scores, as shown with the R^2 score of 0.0467. This outcome suggests that while the features included might have some predictive power, they are insufficient to robustly predict popularity on their own. This can most likely be because of how complex popularity is, or because the relationship between features and popularity is not linear. In order to test this theory once again, I used a Random Forest Regressor, which can handle non-linearity much better than linear regression. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q7.2.png">

The Random Forest model with combined features demonstrates an R² score of 0.4060, which, while showing an improvement over linear regression models by capturing more complex interactions, is still approximately 0.24 points lower than the R² scores achieved by the best-performing individual features, such as duration (R² = 0.6442) and tempo (R² = 0.6059), from the previous analysis in question 6. This relative underperformance of the combined model can be accounted for by its ability to incorporate multiple features simultaneously, therefore adding complexity by integrating various song aspects and their interactions. The model provides a wider understanding of song popularity, even though it does not reach the high predictive power of the best single-feature models. The combination of features can potentially dilute the impact of any single strong predictor by averaging out its effects with less predictive features, therefore resulting in a lower overall R² score than models using only the most predictive features.

## Question 8:
When considering the 10 song features above, how many meaningful principal components can be extracted?

For this analysis, I aimed to explore the effectiveness of Principal Component Analysis (PCA) in reducing the dimensionality of the dataset consisting of ten song features—duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. The goal was to determine how many meaningful principal components can be extracted and what proportion of the variance these components account for. PCA is a powerful tool for dimensionality reduction, used to simplify the data while retaining as much information as possible. It transforms the original variables into a new set of variables, which are orthogonal (uncorrelated) and ranked according to the variance they capture from the data. I performed PCA to extract the principal components and then looked into the proportion of variance explained by each component. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q8.png">

The PCA results indicate that the first three principal components account for 60% of the total variance in the dataset (0.273, 0.162, 0.138). To capture at least 85% of the total variance, we found that seven principal components are required. This reduction from ten original features to seven principal components allows for a more efficient representation of the dataset without significant loss of information, enabling more effective subsequent analyses. The one issue with the PCA is that their interpretation is lost. In order to try and maintain an understanding of each component, I loaded them onto a heatmap as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q8.2.png">

From the heat map, the interpretation of the components are:
- **PC1:** High values might indicate songs that are quieter and more acoustic, while low values suggest songs that are loud and energetic.
- **PC2:** High values indicate less danceable, less joyful music with more instrumental elements.
- **PC3:** High values suggest songs that likely feature live performance aspects and more speech.
- **PC4:** Highly influenced by the duration of the songs and inversely by tempo, suggesting a dimension where longer songs tend to have slower tempos.
- **PC5:** Contrasts fast-tempo songs against acoustic characteristics, indicating that songs with a higher score are faster and less acoustic.
- **PC6:** High values could indicate longer songs that are less instrumental, potentially pointing towards more vocal or lyrical content.
- **PC7:** High values suggest a greater emphasis on speechiness in the track, whereas low values could indicate a focus on live performance attributes.
- **PC8:** High values indicate songs that are emotionally positive but less danceable, which could characterize certain genres like ballads or slower, more melodic music.
- **PC9:** Higher values might be associated with songs that are less acoustic and quieter, potentially indicating more subdued or softer music styles.
- **PC10:** Songs with high values may be louder but have less energetic content.

This interpretation can help with understanding each component, leading to a more beneficial finding of results.

## Question 9:
Can I predict whether a song is in major or minor key from valence?

For this analysis, I aimed to determine if song valence (a measure of musical positiveness) can predict whether a song is in a major or minor key. Understanding this relationship can provide insights into how the emotional content of music correlates with musical theory concepts like key modes. In order to do this, I used a logistic regression to model the relationship between valence and key note. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.png">

The model does has a precision of 0.62, implying that when the model predicts a song is in a major key, it is correct 62% of the time. At the surface level, that seems like the model is working better than randomly guessing, however the major key recall is 1.00 and all of the minor key classification reports are 0. The confusion matrix predicted all songs as being in a major key (1), as shown by the zeros in the first column (actual minor) and zeros in the first row of the second column (predicted minor). And the ROC-AUC score of 0.5029 indicates a model with no discriminative ability whatsoever between major and minor keys, essentially performing no better than random guessing. The model may have an issue due to the imbalance of major and minor keys within the data set. In order to investigate if this was the case, I used a resampling technique called SMOTE (Synthetic Minority Over-sampling Technique). This will synthesize new minority class instances rather than duplicating existing ones until it is balanced. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.2.png">

The model's performance, as captured by the ROC-AUC score, was found to be 0.51, indicating a marginal improvement over random guessing, but still showing limited predictive accuracy. For songs in a minor key (labelled as 0 in the confusion matrix), the model achieved a precision of 0.38 and a recall of 0.52, resulting in an F1-score of 0.44. This suggests that while the model is relatively moderate at identifying true minor key songs, it lacks precision and falsely identifies major key songs as minor key songs quite frequently. For songs in a major key (labelled as 1), the precision was somewhat higher at 0.62 with a recall of 0.48, leading to an F1-score of 0.54. This indicates that while the model is more precise when predicting major key songs, it fails to identify just over half of them correctly. Based on this information, it appears that it is feasible but not very effective to predict the songs key from valence by itself. I wanted to find a better predictor, so I took a look at the features that may elude to a better model. Features I selected were: energy, danceability, acousticness, instrumentalness, and loudness. I followed the same method as previous and added a combined feature predictor (a combination of the features I hand selected), and received the following results:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.3.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.4.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.5.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.6.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.7.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q9.8.png">

As observed from the results, all of the selected features yielded higher ROC-AUC values than valence, thus indicating that they are all better predictors. However, the "best" predictor from the data analysis was the combined feature predictor, with the ROC-AUC of 0.59.

## Question 10:
Which is a better predictor of whether a song is classical music – duration or the principal components extracted in question 8?

For this analysis, I aimed to determine if song duration or principal components (extracted from various song features in question 8) can predict whether a song is classified as classical music. Understanding this relationship can provide insights into how different musical characteristics contribute to the classification of music genres, particularly classical music. In order to do this, I used logistic regression models to assess the predictive power of both duration and principal components. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q10.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q10.2.png">

Based on the analysis, principal components are a significantly better predictor of whether a song is classical music compared to using duration alone. The ROC-AUC score for the principal components model was 0.9564, which is much higher than the 0.5651 for the duration model. The classification report for the principal components model shows a substantial improvement in recall for the classical music class (0.90) compared to the duration model (0.00). Additionally, the principal components model has a much higher precision for the classical music class (0.14) than the duration model (0.00), although it is still quite low. The ROC curve for the principal components model illustrates a far superior performance, with an area under the curve close to 1, indicating excellent discriminative ability. In contrast, the ROC curve for the duration model is much closer to the diagonal line, indicating performance close to random guessing. These results suggest that the combination of various song features captured by principal components provides a more comprehensive understanding and significantly better predictive power for classifying classical music compared to using duration alone.

## Question 11:
What is something interesting about this dataset that is not trivial and not already part of an answer (implied or explicitly) to the previous questions?

For this analysis, I aimed to determine whether there are statistically significant differences in the average tempo of songs between different musical keys. Understanding these differences can provide insights into how key selection might influence the tempo of a composition, which is valuable for both music theory and practical composition. To achieve this, I used an Analysis of Variance (ANOVA) test followed by Tukey's Honestly Significant Difference (HSD) test for post-hoc analysis. ANOVA is useful in this context as it allows us to determine whether there are any statistically significant differences in the means of multiple groups (in this case, the different keys), and Tukey's HSD test helps to identify which specific pairs of keys differ. The results are as follows:

<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q11.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q11.2.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q11.3.png">
<img src="https://github.com/CallEmUp/spotifySongAnalysis/blob/main/images/Q11.4.png">

Based on the analysis, there are statistically significant differences in the average tempo of songs between different keys. The ANOVA test resulted in an F-statistic of 8.8028 with a p-value of 7.84e-16, indicating that the variations in tempo across keys are not due to random chance. The post-hoc analysis using Tukey's HSD test further identified which specific keys have significant differences in average tempo. The results showed that certain key pairs have significantly different average tempos, as indicated by the 'True' values in the reject column of the Tukey HSD output. These significant differences highlight that some keys are associated with faster or slower tempos compared to others. Some of these include: keys C and E, C# and A#, D and F#, D# and E, and F and G. The Tukey HSD plot visually illustrates these significant differences, with some keys showing clear separation in average tempo. Additionally, the boxplot of tempo by key provides a visual representation of the distribution of tempos across different keys, supporting the statistical findings. These results suggest that the key of a song does influence its average tempo, and certain keys tend to be associated with faster or slower tempos.
