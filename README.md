# Fault Detection Methods: A Course Project

## Brief Description (for those who do not have time or who simply are not that invested)

A university course-level exploration of fault detection methods (PCA, K-means Clustering, Soft SVM) applied to the Tennessee Eastman dataset.

## Background

As mentioned above, this is just a university course level of work, done on a limited time frame, to explore multiple methods of detecting faults on a large dataset. These 3 methods (PCA, k-means clustering, soft SVM) were not chosen for any particular reason, simply the ones that were the most realistic to dive into based on what I've studied. (Other methods considered but not applied for this include: neural networks (ANNs), other SVM methods (hard and kernel), other clustering methods (kNN), CCA, ICA, FDA, and PLS)

## Data

The datasets used in this project are taken from the Tennessee Eastman Process. Below are some of the resources I used during research. The code can easily be used for other snippets of data, but the snippets I used are included in the files section of this project, including the fault-free testing set, and the faulty application set.
- [Tennessee Eastman Process: an open source benchmark](https://keepfloyding.github.io/posts/Ten-East-Proc-Intro/)
- [Loading and Exploring the TEP dataset](https://keepfloyding.github.io/posts/data-explor-TEP-1/)
- [TEP variables description and explanation](https://folk.ntnu.no/skoge/prost/proceedings/cpc6-jan2002/larsson.pdf)

## Methods

I am not even remotely an expert on any of these methods, so here are some resources that I read and referenced to study the applications of them

**PCA**
- [Principal component analysis for fault detection and diagnosis. Experience with a pilot plant](https://www.researchgate.net/publication/229022215_Principal_component_analysis_for_fault_detection_and_diagnosis_Experience_with_a_pilot_plant)
- [Fault Detection and Diagnosis Based on PCA and a New Contribution Plot](https://www.sciencedirect.com/science/article/pii/S1474667016358803)
- [A Step By Step Implementation of Principal Component Analysis](https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598)

**K-Means**
- [Distance-based K-Means Clustering Algorithm for Anomaly Detection in Categorical Datasets](https://vemanait.edu.in/pdf/cse/20-21-Paper/Mr.Noor-Basha-Distance-Based-k-Means-Clustering.pdf)
- [Mathematics Behind K-Means Clustering Algorithms](https://muthu.co/mathematics-behind-k-mean-clustering-algorithm/)
- [K-Means: Getting the Optimal Number of Clusters](https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/#:~:text=Silhouette%20Analysis,to%20other%20clusters%20(separation).)

**Soft SVM**
- [A literature review on one-class classification and its potential applications in big data](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00514-x)
- [Understanding One-Class Support Vector Machines](https://www.geeksforgeeks.org/understanding-one-class-support-vector-machines/)

**Note on accuracy metrics:**
In this case, confusion matrices and F1 scores are used to measure each method's accuracy. In the code, a lot of the figures are commented out just for ease of running while I was working on it, to stop it from printing 20 extra graphs each run. This is also noted in each individual folder's readMe.

## Implementation

This segment goes a little in depth about the implementation methods used, but the code is commented enough for it to have reasonable clarity

**PCA:**
The process begins with the preparation of the data, where information is loaded from two CSV files containing details about both faulty and non-faulty instances. These datasets are then concatenated to create a unified dataset. Features are extracted from this dataset, focusing specifically on instances labeled as non-faulty. Subsequently, the extracted features are standardized or mean centered using z-score normalization, which involves subtracting the mean and dividing by the standard deviation.
The core of this process is then performing PCA, in this case, manually implemented through SVD. This involves finding the principal component matrix and taking a subset for the amount chosen to retain, calculated through the explained variance array. These principal components are then used to reconstruct the data by projecting them onto the original data. After reconstruction, the difference between the original and reconstructed is calculated for each point, which gives us the reconstruction errors. These reconstruction errors represent how well the original data can be approximated by the reduced dimension components. The hope is that the anomalies in the dataset will yield much higher errors, which can be filtered out after determining an appropriate threshold.
A histogram of these errors is plotted, and a threshold is determined based on their distribution. The threshold for identifying anomalies is determined based on the distribution of distances calculated for fault-free instances. Initially, a histogram plot of these distances is generated to visualize their distribution. Then, statistical measures such as the mean and standard deviation of the distances are computed. The threshold is subsequently set as the mean distance plus three times the standard deviation. This threshold serves as a reference point for distinguishing normal instances from anomalies based on the magnitude of reconstruction error. Then, it iterates over different fault numbers, computing the F1-score and accuracy for each fault number. Results are combined based on the threshold to identify anomalies in the data. Performance evaluation is conducted using confusion matrices, F1-score, and accuracy metrics.

**K-Means:**
The process for anomaly detection begins by loading data from two CSV files containing details of both faulty and non-faulty instances. These datasets are combined into a unified dataset. Features are extracted from the non-faulty instances and standardized using z-score normalization (subtracting the mean and dividing by the standard deviation).
The core of the anomaly detection process involves K-means clustering which is manually implemented. This process initializes centroids randomly and updates them iteratively to minimize the distance between data points and centroids. In this case, the number of clusters (k) is set to 1, with a maximum of 100 iterations to ensure convergence. Once clustering is complete, the cluster center is used to compute distances of fault-free instances from this center.
These distances are then analyzed to identify anomalies. A histogram of the distances is plotted, and a threshold is determined based on the distribution of these distances. The threshold is set as the mean distance plus twice the standard deviation, marking the boundary beyond which instances are considered 'faulty'.
The method involves iterating over different fault numbers, computing the F1-score and accuracy for each, then combining results based on the threshold to identify anomalies in the data. Performance is evaluated using confusion matrices, F1-score, and accuracy metrics. Additional analysis includes filtering data for different fault numbers and performing random simulation runs to compute distances from the cluster center.

**Soft SVM:**
The process begins by loading fault-free and faulty training datasets, which are then combined into a single matrix. The fault-free portion is used to train a One-Class SVM (OCSVM) model, which is subsequently tested on the faulty training dataset. The OCSVM implementation is done using the ocsvm() function from MATLAB's "Statistical and Machine Learning" Toolbox.
The ocsvm() function requires a matrix of predictor data, with each column representing a predictor variable and each row an observation. Parameters for kernel generation and convergence conditions are set to 'auto', allowing MATLAB to determine these values heuristically. The data columns are standardized internally by the ocsvm() function, so no pre-processing is necessary. Initially, the contamination fraction is set to 0 because the training data contains only normal operating data.
The output of the ocsvm() function includes anomaly scores, a logical array indicating anomalies, and the One-Class-SVM model object. Anomaly scores range from [-inf, inf], with large negative scores indicating normal observations and large positive scores indicating outliers. To classify new data points, the isanomaly() function is used, requiring the new input data and the trained model object. This function classifies observations based on the Scorethreshold value from the trained model, identifying those with scores above the threshold as outliers.
After the initial training on fault-free data, the model parameters are fine-tuned using faulty training data. Anomaly scores are generated for the first simulation number of each fault type using the isanomaly() function. The F1 and accuracy scores are calculated for each fault number and averaged. The ContaminationFraction parameter is adjusted to optimize these scores, resulting in a final value of approximately 0.015.
