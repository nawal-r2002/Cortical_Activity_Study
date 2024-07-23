---
layout: page
title: Final Report
permalink: '/final/'
---
## Final Presentation Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/0MJGxjKrcYo?si=9Gk2e-VLGXyPeCGJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

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
For our supervised learning models, we applied a multiple linear regression model, sparse regressio by applying a ridge regression algorithm, a feedforward neural network (FNN) and a recurrent neural network (RNN). 

Using Multiple Linear Regression will allow us to determine the relative contributions of each of the four characteristics in our input data to the output. It will also allow us to determine whether a linear relationship exists within the data.

Sparse regression attempts to find a set of vectors that optimize the projection from the input to the output. The number of nonzero entries in the vectors is minimized which will allow us to determine which of the input features are most crucial to the output [5]. 

A feedforward neural network performs mathematical transforms on the input and applies a nonlinear activation function on it to transform it into the output. This will allow us to capture more complex, and potentially nonlinear relationships that may exist within the data. 

A recurrent neural network performs mathematical transformations on the input in order to perform tasjs that are related to tasks that involve time series data. This method will allow us to be able to help us to capture a sequential sequence of how the relationships are interacting with each other in order to capture the relationships between the existing features and the output.

### Multi-Linear Regression with rCCA
Upon using rCCA, we were able to find the canonical components that maximized the correlation between our input and output. Regularization in rCCA helps manage multicollinearity and overfitting. Through cross validation, we determined that one latent dimension was optimal to capture a relationship within our data. To cross validate our rCCA model, we split the training data into 5 folds at random. This model was fit and an r-squared score was calculated for each fold to evaluate the model’s ability to explain the variance in the data.  To validate the significance of our rCCA model, we implemented a null model. Our null model was trained on randomly shuffled data points from the training dataset using the same rCCA approach. The performance (r^2 values) of the null model was compared to the original model to assess the importance of the canonical components. This was repeated for 1000 iterations, and the median r^2 values across all iterations and latent dimensions was used to determine the optimal hyperparameters.  We fitted a linear regression model using the transformed canonical component to quantify the relationship between the canonical components and to predict the output. Here, the transformed canonical components from rCCA were split into training and testing sets. The model was trained using the training set and then used to predict the output of the test set.

### Multi-Linear Regression with PCA
The PCA model was implemented on the X and Y of the training and test data set to understand structure and reduce dimensionality. The data was split such that there was 70% training data and 30% testing data. The input and output features were scaled to ensure uniformity in the data using scikit learn. Then, PCA applied to find the first two principal components for the input and output training and testing data. After finding the principal components set for the training set, we used it to predict the output of the linear regression model for the test set.

### Sparse Regression
We also applied a sparse regression model to our data. Considering that we were using relatively small datasets, we chose to use ridge regression to avoid overfitting. Ridge regression also penalizes large weights therefore reducing variance. We compared the metrics of two preprocessing techniques used in conjunction with sparse regression. The first approach involved preprocessing raw EEG and EMG measurements using the neuromechanical model. The characteristic parameters gained from the neuromechanical model were then scaled using z-score normalization to ensure large weights were not skewing predictions due to their magnitude. Then rCCA with 1 latent dimension and a regularization coefficient of 0.2 was applied. The second approach involved preprocessing the raw EEG and EMG measurements with wavelet transform and applying rCCA. Finally, a ridge regression model was fit to both datasets.

### Feedforward and Recurrent Neural Network
We originally applied a feedforward neural network implementation for our dataset. Given that neural networks could handle more complicated data, we decided to use the EMG recordings and the center of mass recordings as our dataset. The center of mass recordings include time series recordings of acceleration, velocity, and distance of the participants during the balance perturbation. We also decided to use a recurrent neural network model since the data we were fitting was time series data for our 3 input features (acceleration, velocity, and distance) and EMG output. Both approaches involved reshaping the data and combining all input features in order to fit the model over all the inputs for each of the data points. They also both include having an architecture where there is 1 hidden layer 50 neurons with the ReLu Activation Function. After reshaping the data, we were able to fit the model on both neural network models to determine the trends in loss in order to determine the model that would show better results. Finally, after observing the trends in the loss, we made the prediction using the model that did not overfit significantly. 

## Results and Discussion
### Multi-Linear Regression with rCCA
![cross_validation](_images/cross_validation_cca.png)

*Fig. 1: Comparing the correlation between input and output that each of the latent dimensions captures*

![mlr_cca](_images/mlr_updated.png)

*Fig. 2: Result of applying multiple linear regression on canonical components*

Figure 1 was the result of our cross-validation. It shows the median R2 value over 1000 iterations of each of the latent dimensions and the null model. The R2 score of the null model is low, as expected because the data that it is trained on is randomly shuffled and thereby has little correlation. The R2 value of the first latent dimension is the highest and it was therefore chosen to be the latent dimension that would be used to train our model. 

The input arrays, X and Y of sizes (204, 4) and (204, 8) were transformed into U and V of sizes (204, 1) and (204, 1) using an rCCA model that extracted 1 latent dimension and used a regularization constant equal to 1. U and V were then split into a training and testing dataset. 70% of the data was used for training and 30% was used for testing. 

The mean squared error of our model was 0.8696 and the R2 score was 0.4025. We suspect that this is because of the sporadic distribution of data and the inability of our linear model to capture its complex relationships. Additionally, the presence of outliers likely increases the inaccuracies of our model. 

Upon converting our predicted V-values from the canonical space into the input space and comparing it with the Y_test, we found that our R2 score was 0.0623 and the RMSE was 0.9762. Upon further inspection, we printed and examined the reconstruction of the predicted Y. We found that our method failed to reconstruct the parameters with smaller values, possibly due to significant differences in scale between the first and subsequent characteristic parameters.

### Multi-Linear Regression with PCA
![PCA on X](_images/PCA_X.jpg)

*Fig. 3: PCA Space Feature Visualization for X*

Figure 3 is the result from completing PCA on X shows that the testing set for X has relatively few noise points in the PCA Space and created a cluster of data along the training set. Also, the cumulative variance ratio is 68% for X therefore 68% of the variance is captured from the original 4 features from X. This suggests that there are significant features in the dataset that could predict the main characteristic parameters. 

![PCA on Y](_images/PCA_Y.jpg)

*Fig. 4: PCA Space Feature Visualization for Y*

Figure 4 is the result from completing PCA on Y and it shows that the points in a two-dimensional space are scattered such that no clusters are created between training set and there is no specific clusters created from the test set. Also, the cumulative variance ratio is 45% for Y therefore the two principal components shows 45% of the variance captured from the dataset. The amount of variance captured may not have a pivotal role in the predictive model's outcome. 

The results from the RMSE was 1.1789 and the R2 score was 0.001345. From these metrics, it can be suspected that a linear model was not able to capture the complex relationships from the dataset. The presence of outliers may have increased the chance of the model not working and it may be possible that there was not a sufficient number of components signified for X and Y since the computed cumulative variance ratio did not reach an adequate threshold. Also, the results shown from the multi-linear regression may not be accurate for the prediction because the scope of the model was not converted back to the individual components from the original input and output after completing using PCA on the model. 

![PCA on X with 85% Variance in Componenents](_images/PCA_X_Updated.png)

*Fig. 5: PCA Space Feature Visualization for X with 85%*
![MLR from PCA](_images/mlr_pca_updated.png)

*Fig. 6: MLR from PCA Results*

With 85% variance, 3 principal components were found in the z-space for the training and testing dataset for the input data in the neuromechanical model. Therefore, we can see that the data has started to move further apart from each other when looking at Figure 5 which means that more variance can be seen from adding an extra component in the model.

When running the MLR model with the the updated versions of X and Y has shown that the prediction power of the model slightly increased since the R2 score became 0.07487 and the RMSE became 1.06808. However, these metrics also show that there is not a significant amount of predicting power in order to make conclusions about the characteristic parameters in the input. Also, from Figure 6, we can see that in comparison to the regression line there is significant errors for the predicted model so there may be outliers that have reduced the predictions accuracy. 

![MLR from PCA](_images/mlr_pca_updated.png)

### Sparse Regression with Scaling and rCCA on Characteristic Parameters
![Ridge Regression on Canonical Components](_images/ridge_regression_cca.png)

*Fig. 7:  Ridge Regression on Canonical Components of Characteristic Parameters of EEG and EMG*

Figure 7 is the result of scaling the input and output datasets, applying regularized CCA and fitting a ridge regression model to them. The input and output datasets consisted of time series wavedata that had been pre-processed using the neuromechanical model to yield characteristic parameters of EEG and EMG data.
First, We used the results of cross validation shown in Figure 1 to determine the optimal number of latent dimensions. Another round of cross validation was performed to determine the optimal regularization coefficient for regularized CCA. This was found to be 0.2.

The input and output data were normalized using Z-score normalization. Then, an rCCA model with latent dimensions = 1 and regularization coefficient = 0.2 was fit to the data. X (204, 4) and Y(204, 8) were transformed into U (204, 1) and V(204, 1). Data was split into training and testing sets using an 80/20 split. Finally, the ridge regression model was fit to the transformed data.

The results of using ridge regression on the scaled and transformed data were much better than that of basic regression on non-scaled data. Z-score normalization brings all the input features to a similar scale thereby preventing any one feature from dominating due to its larger scale. The use of a smaller regularization coefficient also seemed to yield better results. It is possible that the use of a regularization coefficient of 1 in our earlier model led to underfitting. 

The RMSE value for this model was 0.65 and the R2-score for the canonical prediction was 0.46. The R2-score for the canonical predictions converted back into X-space was 0.0637, and the RMSE was 0.9751. These values can be explained by the weak correlation between the features of our dataset. Additionally, having to predict eight features using a four dimensional input likely led the model to overgeneralize.

### Sparse Regression with rCCA on Wavedata
![Cross validation ](_images/wave_vis.png)

*Fig. 8: Visualizations of the EEG and EMG waves*

![Ridge Regression on canonical components](_images/ridge_wave_cca.png)

*Fig. 9: Ridge Regression on Canonical Components of Wavedata*

We attempted to improve the existing model by training it on raw EEG and EMG data, to see if reintroducing some features would lead to better results. We noticed that the EMG wave appeared to be almost an inverse of the EEG, so we wanted to see if the ridge regression model could capture this relationship.

Figure 8 shows a sample wave for a single trial. The orange wave shows EMG data and the blue wave shows EEG data. The wave data initially consisted of 1400 dimensions (measurements for 140 timesteps). We were able to cut this down to 200 because we noticed that large fluctuations in the wavelet only occurred until this point. It was difficult to determine a feasible number of latent dimensions that would adequately capture the relationships in our data. Ultimately, we decided to perform rCCA with a regularization term = 0.2 and latent dimension = 1. 

Figure 9 shows the result of fitting a ridge regression model to the canonical component of the wavelet data. For the canonical component, the RMSE value was 0.3636 and the  R2  was 0.8132. The R2 score was higher for this model but we faced problems when trying to convert back into XY-space. The R2 score after converting the predictions was -1555, and the RMSE was 0.6994.

The negative R2 score can likely be explained by the large loss of dimensionality when attempting to reduce 200 dimensions to 1 latent dimension. The wavedata likely also had a large amount of noise leading to a weak correlation between the input and the output.

### Feedforward Neural Network
![Feedforward Loss over 200 Epochs](_images/FeedforwardLoss.png)

*Fig. 10: Feedforward Model Loss Trend for Training and Testing Data over 200 Epochs*

With the feedforward neural network we attempted to use time-series data for the three features and the output from EMG data. From the initial feedforward neural network implementation with this dataset, we found that the model may be slightly overfitted due to the differences in the trends between the training and testing data in some of the iterations. This might be because the feedforward neural network was not able to hangle the temporal relationships that might be within the time series data. 

### Recurrent Neural Network
![Recurrent Loss over 200 Epochs](_images/RecurrentLoss.png)

*Fig. 11: Recurrent Model Loss Trend for Training and Testing Data over 200 Epochs*

After implementing the RNN with the time-series dataset, we found that there was not a significant loss difference between the training and testing data after the initial iteration as shown in Figure 11. Therefore, this model was shown not to be an overfitted model, which ,eans that we may be able to see more accurate predictions from our test data. 

From the metrics, we found that the R2 score from the RNN was 0.318225 and the RMSE was approximately 0.09304. Therefore, this model may be able to show effective results in terms of making an accurate prediction for some cases because the model is able to find a relationship between the inputs and the target data. However, there may be relationships that have not been found between the inputs and target data such that a fully accurate prediction because of the model architecture that was used. There is also a change that the model had this performance in terms of metrics because of the limitations from vanishing gradients from triaining the model. 

## Evaluation and Model Comparison
![Input Correlations](_images/input_corrs.png)

*Fig 12: Correlation matrix for input features*

![Output Correlations](_images/output_corrs.png)

*Fig 13: Correlation matrix for output features*

![All Features Correlations](_images/input_output_corrs.png)

*Fig 14: Correlation matrix for input and output features*

The above three figures show the correlations between input and out features, just output features and just input features, respectively. There is little correlation between the input and output features as is visible from the last correlation matrix (the first four rows/columns are input features and the rest are output features). This could be one of the reasons why our models failed to make good predictions. In the future, we could potentially attempt to increase correlation by experimenting with polynomial features and other feature engineering techniques. 

![Model Metrics Comparisons](_images/model_comps.png)

*Fig 15: Table comparing metrics for the implemented models*

For rCCA with multiple linear regression, R2  score was higher and RMSE was lower in the canonical space. But upon converting to XY-space  R2  score dropped and RMSE rose. PCA with multiple linear regression resulted in a higher R2  score than rCCA, but yielded a higher RMSE value. The R2  score was still relatively low, but this can likely be explained by the weak correlation between input features and output features in our dataset. It is also possible that the multiple linear regression model was too complex, and overfitting to our training data, especially considering that our dataset was relatively small. To address these issues, we also implemented a sparse regression model.

For the sparse regression models that used rCCA as a preprocessing technique, R2  score was typically high for predictions made in canonical space. However, when these were converted back into XY-space, it dropped significantly. The RMSE also tended to increase when predictions were converted. We suspect that this is because of issues with how we were converting our predictions back into XY-space. After implementing rCCA without scaling the data, we noticed that the canonical weights matrix had some very small numbers, and thereby gave very little weight to some features. Therefore, when we implemented sparse regression on the characteristic parameters, we used z-score normalization to ensure all the features would be weighted equally. This gave us better results in terms of R2 score for the canonical components. 

However, upon converting back to XY-space, it once again dropped. We then tried running rCCA and ridge regression on the raw wavedata to see if this would capture more detail. This model gave us the highest R2 score for canonical components of all the models that used rCCA. However, once again, upon converting back, it dropped. For this model it dropped to a large negative number, indicating that the model was making predictions that were worse than simply predicting the average of the function for any given test point. We suspected that this was because using 1 latent dimension to capture the relationships in our data led to underfitting. So, we tried experimenting with larger numbers of latent dimensions. Unexpectedly, this resulted in higher RMSE values and lower R2 scores. Another cause could have been the lack of data. The neuromechanical model dataset contained 208 datapoints while the wave dataset contained 107 datapoints. We plan on experimenting with wavelet transforms and different dimensionality reduction techniques in the future. 

The implementation of the neural network showed that it had a higher level of predicting power than MLR and Sparse Regression because of the differences between the RMSE and the R2 score. This is likely due to its ability to capture more complex relationships than simple regression. The regression models that were implemented were only capable of capturing linear relationships, but neural networks seek to introduce nonlinearity with the use of activation functions. Neural networks would also be a good place to start if we decide to reframe our problem as a classification one instead of regression.

## Next Steps
To improve our study in the future, several steps can be taken. First, we can refine our preprocessing techniques. We can do this by comparing the performance of models using wavelet transform and the neuromechanical model, optimizing the wavelet transform parameters, and implementing robust methods for outlier detection and removal. We can also explore dimensionality reduction techniques to capture complex relationships in the data. 

In terms of regression model improvement, we can experiment with different regularization techniques such as Lasso and Elastic Net, conduct extensive hyperparameter tuning, and gradually increase model complexity to better capture non-linear relationships. For neural network optimization, we can test various architectures, activation functions, dropout, and regularization techniques, and utilize adaptive learning rate schedules to enhance training.

We can ensure model validation and testing through k-fold cross-validation, tracking additional performance metrics, and evaluating models on a separate test set to assess real-world applicability. Feature importance analysis can be conducted using SHAP values, and based on this analysis, we can engineer new features to potentially enhance model performance. We can also explore a classification approach, including binary and multi-class classification models, and compare their performance to regression models. Additional considerations include implementing data augmentation techniques to increase the size and diversity of the training dataset, investigating domain adaptation techniques for better generalizability, and collaborating with neuroscientists and clinicians to validate the biological plausibility of the model predictions. By following these steps, we aim to enhance the accuracy and robustness of our predictive models, leading to better insights into the cortical contributions to balance control in aging and impairment.

## References
[1] A. M. Payne, L. H. Ting, and G. Hajcak, “Do sensorimotor perturbations to standing balance elicit an error-related negativity?,” Psychophysiology, vol. 56, no. 7, p. e13359, Mar. 2019, doi: https://doi.org/10.1111/psyp.13359.

[2] M. Saeidi et al., “Neural Decoding of EEG Signals with Machine Learning: A Systematic Review,” Brain Sciences, vol. 11, no. 11, p. 1525, Nov. 2021, doi: https://doi.org/10.3390/brainsci11111525.

[3] S. Boebinger et al., “Precise cortical contributions to sensorimotor feedback control during reactive balance,” PLoS computational biology, vol. 20, no. 4, p. e1011562, Apr. 2024, doi: https://doi.org/10.1371/journal.pcbi.1011562.

[4] A. Mihalik et al., “Canonical Correlation Analysis and Partial Least Squares for Identifying Brain–Behavior Associations: A Tutorial and a Comparative Study,” Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, vol. 7, no. 11, pp. 1055–1067, Nov. 2022, doi: https://doi.org/10.1016/j.bpsc.2022.07.012.

[5] G. Gordon, R. Tibshirani. Machine Learning 10-725. Class Lecture, Topic: “Lecture 2: Optimization”. School of Computer Science, Carnegie Mellon, Pittsburgh PA, Aug. 30, 2012.

## Contribution Table and Gantt Chart
[Click Here to View Gantt Chart](https://docs.google.com/spreadsheets/d/1LJo-kXLj1V64y5hSA2eHDMiAgkj5vY8V2jxa6E2smUs/edit?gid=0#gid=0)

![Final Contributions](_images/Final_Contributions.jpg)