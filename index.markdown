---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: page
title: Project Proposal
---
## Proposal Video

## Introduction
The capability of the nervous system to effectively control muscles to maintain balance in response to disturbances to the body is crucial for survival. Several neural pathways in the brainstem and spinal cord generate sensorimotor responses, but it is unclear how cortical activity from the brain contributes to these motor responses.

Recordings from the cortex using electroencephalography (EEG) have revealed a large, negative peak of cortical activity, N1, known as an error assessment signal evoked when external stimuli cause an unexpected error from the upright posture [1]. 

Our dataset includes 6 recordings (6 conditions: 2 directions and 3 magnitudes of balance perturbation) of EEG and EMG per participant, collected from 36 participants (19 healthy old adults, 17 Parkinson’s Disease patients). 

## Problem Definition
Our goal for this study is to develop predictive models in order to accurately forecast the characteristic parameters of muscle activity based on the characteristic parameters of the cortical N1 activity. This approach aims to enhance our ability to investigate changes in cortical contributions to balance control in aging and impairment.

## Proposed Methods
The EEG and EMG recordings will be preprocessed using a neuromechanical model, which reconstructs the data into several characteristic parameters based on the participants’ center of mass (CoM) kinematics recordings [2]. After processing this model, we will obtain 4 characteristic parameters for the N1 activity, including 3 CoM feedback gains and 1 time latency. Additionally, we will obtain 8 characteristic parameters for muscle activity, including 6 CoM feedback gains and 2 time latencies.

In order to pre-process the characteristic parameters gained from the neuromechanical model, we propose to use techniques which include PCA, and regularized canonical correlation analysis (CCA) to help us identify transformations between input and output that maximize the correlation [3].

For our supervised learning models we plan to apply multiple linear regression, sparse regression, and a feedforward neural network. 

Multiple Linear Regression will allow us to determine the relative contributions of each of the four characteristics in our input data to the output. It will also allow us to determine whether a linear relationship exists within the data. 

Sparse regression attempts to find a set of vectors that optimize the projection from the input to the output. The number of nonzero entries in the vectors is minimized which will allow us to determine which of the input features are most crucial to the output [4]. 

A feed forward neural network will allow us to capture non-linear relationships. We plan on using the scikit learn library’s MLPRegressor module to implement the neural network [5].

## Potential Results and Discussion
The metrics we plan to use to evaluate our models include accuracy, precision and recall, mean squared error (MSE), and r2 score. We will also compare our prediction results to a null model and use cross validation to evaluate our machine learning models. 

From our results, we hope to be able to discuss the characteristic factors from N1 activity that have a significant impact on muscle activity and we will be able to improve our understanding of how cortical contributions influence balance control.

## References
[1] A. M. Payne, L. H. Ting, and G. Hajcak, “Do sensorimotor perturbations to standing balance elicit an error-related negativity?,” Psychophysiology, vol. 56, no. 7, p. e13359, Mar. 2019, doi: https://doi.org/10.1111/psyp.13359.

[2] S. Boebinger et al., “Precise cortical contributions to sensorimotor feedback control during reactive balance,” PLoS computational biology, vol. 20, no. 4, p. e1011562, Apr. 2024, doi: https://doi.org/10.1371/journal.pcbi.1011562.

[3] A. Mihalik et al., “Canonical Correlation Analysis and Partial Least Squares for Identifying Brain–Behavior Associations: A Tutorial and a Comparative Study,” Biological Psychiatry: Cognitive Neuroscience and Neuroimaging, vol. 7, no. 11, pp. 1055–1067, Nov. 2022, doi: https://doi.org/10.1016/j.bpsc.2022.07.012.

[4] G. Gordon, R. Tibshirani. Machine Learning 10-725. Class Lecture, Topic: “Lecture 2: Optimization”. School of Computer Science, Carnegie Mellon, Pittsburgh PA, Aug. 30, 2012.

[5] “sklearn.neural_network.MLPRegressor — scikit-learn 0.21.3 documentation,” Scikit-learn.org, 2010. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html


## Gantt Chart and Contribution Table
[Click Here to View Gantt Chart](https://docs.google.com/spreadsheets/d/1LJo-kXLj1V64y5hSA2eHDMiAgkj5vY8V2jxa6E2smUs/edit?gid=0#gid=0)

<img src="_images/Proposal_Contributions.jpg" alt="Proposal Contribution">
