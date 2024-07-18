---
layout: page
title: Final Report
permalink: '/final_report/'
---
## Final Report Video
<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/LuKO74ikTkk?si=GLPDVlfYyMFnAWEU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe> -->

## Introduction
The capability of the nervous system to effectively control muscles to maintain balance in response to disturbances to the body is crucial for survival. Several neural pathways in the brainstem and spinal cord generate sensorimotor responses, but it is unclear how cortical activity from the brain contributes to these motor responses.

Recordings from the cortex using electroencephalography (EEG) have revealed a large, negative peak of cortical activity, N1, known as an error assessment signal evoked when external stimuli cause an unexpected error from the upright posture [1]. 

A general framework for decoding neural signals has been established by Saeidi, et al. This involves using Wavelet Transform (WT) to convert raw recordings into wavelets that localize features and remove noise. Since wavelets are often high dimensional, PCA and other dimensionality reduction algorithms are used to preprocess the data before it is fed into the machine learning algorithm of choice [2]. We plan on using a similar pipeline in our project. However, since our project is mainly focused on cortical N1 activity, our preprocessing will involve transforming raw EEG signals into a vector of characteristic parameters that describe the cortical N1 activity using a neuromechanical model as opposed to WT.

Our dataset includes 6 experimental recordings (6 conditions: 2 directions and 3 magnitudes of balance perturbation) of EEG and EMG per participant, collected from 36 participants (19 healthy old adults, 17 Parkinson’s Disease patients) in the Emory Rehabilitation Hospital. This results in 216 data points in total. 

## Problem Definition
Our goal for this study is to develop predictive models in order to accurately forecast the characteristic parameters of muscle activity based on the characteristic parameters of the cortical N1 activity. This approach aims to enhance our ability to investigate changes in cortical contributions to balance control in aging and impairment.

## Methods
The EEG and EMG recordings were preprocessed using a neuromechanical model, which reconstructs the data into several characteristic parameters based on the participants’ center of mass (CoM) kinematics recordings [3]. After processing this model, we obtained 4 characteristic parameters for the N1 activity, including 3 CoM feedback gains and 1 time latency. Additionally, we obtained 8 characteristic parameters for muscle activity, including 6 CoM feedback gains and 2 time latencies.
In order to pre-process the characteristic parameters gained from the neuromechanical model, we used techniques which include PCA, and regularized canonical correlation analysis (CCA) to help us identify transformations between input and output that maximize the correlation [4].
For our supervised learning models, we applied a multiple linear regression model, and we hope to implement a sparse regression, and a feedforward neural network in the next phase.

Using Multiple Linear Regression will allow us to determine the relative contributions of each of the four characteristics in our input data to the output. It will also allow us to determine whether a linear relationship exists within the data.

### Multi-Linear Regression with rCCA
Upon using rCCA, we were able to find the canonical components that maximized the correlation between our input and output. Regularization in rCCA helps manage multicollinearity and overfitting. Through cross validation, we determined that one latent dimension was optimal to capture a relationship within our data. To cross validate our rCCA model, we split the training data into 5 folds at random. This model was fit and an r-squared score was calculated for each fold to evaluate the model’s ability to explain the variance in the data.  To validate the significance of our rCCA model, we implemented a null model. Our null model was trained on randomly shuffled data points from the training dataset using the same rCCA approach. The performance (r^2 values) of the null model was compared to the original model to assess the importance of the canonical components. This was repeated for 1000 iterations, and the median r^2 values across all iterations and latent dimensions was used to determine the optimal hyperparameters.  We fitted a linear regression model using the transformed canonical component to quantify the relationship between the canonical components and to predict the output. Here, the transformed canonical components from rCCA were split into training and testing sets. The model was trained using the training set and then used to predict the output of the test set.

### Multi-Linear Regression with PCA
The PCA model was implemented on the X and Y of the training and test data set to understand structure and reduce dimensionality. The data was split such that there was 70% training data and 30% testing data. The input and output features were scaled to ensure uniformity in the data using scikit learn. Then, PCA applied to find the first two principal components for the input and output training and testing data. After finding the principal components set for the training set, we used it to predict the output of the linear regression model for the test set.

## Results and Discussion
### Multi-Linear Regression with rCCA
![cross_validation](_images/cross_validation_cca.png)

*Fig. 1: Comparing the correlation between input and output that each of the latent dimensions captures*

![mlr_cca](_images/mlr_cca.png)

*Fig. 2: Result of applying multiple linear regression on canonical components*

Figure 1 was the result of our cross-validation. It shows the median r2 value over 1000 iterations of each of the latent dimensions and the null model. The r2 score of the null model is low, as expected because the data that it is trained on is randomly shuffled and thereby has little correlation. The r2 value of the first latent dimension is the highest and it was therefore chosen to be the latent dimension that would be used to train our model. 
The input arrays, X and Y of sizes (204, 4) and (204, 8) were transformed into U and V of sizes (204, 1) and (204, 1) using an rCCA model that extracted 1 latent dimension and used a regularization constant equal to 1. U and V were then split into a training and testing dataset. 70% of the data was used for training and 30% was used for testing. 
The mean squared error of our model was 1.0377 and the r2 score was 0.366. We suspect that this is because of the sporadic distribution of data and the inability of our linear model to capture its complex relationships. Additionally, the presence of outliers likely increases the inaccuracies of our model. 
Upon converting our predicted V-values from the canonical space into the input space and comparing it with the Y_test, we found that our r2 score was a large negative number. Upon further inspection, we printed and examined the reconstruction of the predicted Y. We found that our method failed to reconstruct the parameters with smaller values, possibly due to significant differences in scale between the first and subsequent characteristic parameters.

### Multi-Linear Regression with PCA
![PCA on X](_images/PCA_X.jpg)

*Fig. 3: PCA Space Feature Visualization for X*

Figure 3 is the result from completing PCA on X shows that the testing set for X has relatively few noise points in the PCA Space and created a cluster of data along the training set. Also, the cumulative variance ratio is 68% for X therefore 68% of the variance is captured from the original 4 features from X. This suggests that there are significant features in the dataset that could predict the main characteristic parameters. 

![PCA on Y](_images/PCA_Y.jpg)

*Fig. 4: PCA Space Feature Visualization for Y*

Figure 4 is the result from completing PCA on Y and it shows that the points in a two-dimensional space are scattered such that no clusters are created between training set and there is no specific clusters created from the test set. Also, the cumulative variance ratio is 45% for Y therefore the two principal components shows 45% of the variance captured from the dataset. The amount of variance captured may not have a pivotal role in the predictive model's outcome. 

![MLR_PCA](_images/mlr_pca.png)

*Fig 5: Result of Applying Multi-Linear Regression from PCA*

The results from the mean squared error was 1.38987 and the R2 score was 0.001345. From these metrics, it can be suspected that a linear model was not able to capture the complex relationships from the dataset. The presence of outliers may have increased the chance of the model not working and it may be possible that there was not a sufficient number of components signified for X and Y since the computed cumulative variance ratio did not reach an adequate threshold. Also, the results shown in Figure 5 could not be accurate for the prediction because the scope of the model was not converted bcck to the individual components from the original input and output after completing using PCA on the model. 

## Model Comparison

## Next Steps
We intend to use a wavelet transform instead of the neuromechanics model for preprocessing raw EEG and EMG recordings. It appears that only the first canonical component identified by rCCA may not adequately reconstruct the characteristic parameters of the EMG recordings possibly due to significant differences in scale between the first and subsequent characteristic parameters. We have applied the wavelet transform to both EEG and EMG recordings, and we are currently exploring the implementation of rCCA or PCA on the wavelet-transformed data. By comparing models on different preprocessing methods, we can identify the approach that best captures the relationships within the data. 

We also intend to remove outliers, specifically trials with noisy EEG or EMG recordings. These outliers could potentially introduce inaccuracies in our model training and prediction process. To do so, we plan to review our data quality and preprocessing to ensure the dataset is optimal for modeling. This will ensure there are no missing or erroneous values in the data set while also checking for outliers. 

Also, we hope to modify the results of PCA by attempting to maximize the number of principal components for 
X and Y. To do so, we will set a threshold of 85% and complete analysis in order to determine the best number of components. This will ensure that the variance is maximized such that the dimension of both the input and output dimension is reduced adequately. 

## References
[1] A. M. Payne, L. H. Ting, and G. Hajcak, “Do sensorimotor perturbations to standing balance elicit an error-related negativity?,” Psychophysiology, vol. 56, no. 7, p. e13359, Mar. 2019, doi: https://doi.org/10.1111/psyp.13359.

[2] M. Saeidi et al., “Neural Decoding of EEG Signals with Machine Learning: A Systematic Review,” Brain Sciences, vol. 11, no. 11, p. 1525, Nov. 2021, doi: https://doi.org/10.3390/brainsci11111525.

[3] S. Boebinger et al., “Precise cortical contributions to sensorimotor feedback control during reactive balance,” PLoS computational biology, vol. 20, no. 4, p. e1011562, Apr. 2024, doi: https://doi.org/10.1371/journal.pcbi.1011562.

[4] A. Mihalik et al., “Canonical Correlation Analysis and Partial Least Squares for Identifying Brain–Behavior Associations: A Tutorial and a Comparative Study,” Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, vol. 7, no. 11, pp. 1055–1067, Nov. 2022, doi: https://doi.org/10.1016/j.bpsc.2022.07.012.

## Contribution Table and Gantt Chart
[Click Here to View Gantt Chart](https://docs.google.com/spreadsheets/d/1LJo-kXLj1V64y5hSA2eHDMiAgkj5vY8V2jxa6E2smUs/edit?gid=0#gid=0)

<img src="_images/Midterm_Contributions.jpg" alt="Midterm Contribution">
