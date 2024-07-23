# Georgia Tech CS4641 - Machine Learning Group 2 Project
## Analyzing Cortical Relations to Muscle Activity
### Akshaya Arun, Manasa Golla, Nawal Reza, Jemmy Xiao, Kaitlyn Zhuang

### /_data/: Contains EEG and EMG readings used to train the model
* **/_data/HOA_PD_SRM_Output_StatsTable__05-Jun-2024.xlsx**: Contains characteristic parameters of the EEG and EMG readings for 204 patients
* **/_data/VelocityRecordings.xlsx**: Contains velocity recording data used for analysis
* **/_data/EMGRecording.xlsx**: Contains EMG recording data used for analysis
* **/_data/AccelerationRecordings.xlsx**: Contains acceleration recording data used for analysis
* **/_data/DistanceRecording.xlsx**: Contains distance recording data used for analysis

### /models/: Contains implementations of each of our proposed machine learning algorithms and preprocessing methods
* **/models/Dual.ipynb**: Contains implementation of regularized CCA and multiple linear regression
* **/models/Dual_wavelet.ipynb**: Contains implementation of wavelet transform on the data
* **/models/multi_linear_regression.ipynb**: Contains implementation of multiple-linear regression using PCA
* **/models/pca.ipynb**: Contains implementation of PCA on the data
* **/models/NeuralNetwork.ipynb**: Contains implementation of neural network models
* **/models/Sparse_regression_characteristic_params.ipynb**: Contains implementation of sparse regression on characteristic parameters
* **/models/Sparse_regression_wavedata.ipynb**: Contains implementation of sparse regression on wavelet-transformed data

### /_images/: Contains all images of machine learning implementation results and miscellaneous images needed for the report
