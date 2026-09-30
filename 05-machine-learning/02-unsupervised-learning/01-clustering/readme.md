# Clustering Algorithms

## Overview

Clustering is an unsupervised learning technique used to discover hidden patterns and structures in unlabeled data.

Unlike supervised learning, clustering algorithms do not use predefined target labels. Instead, they group similar samples together based on different concepts such as:

- Distance similarity
- Data density
- Hierarchical relationships
- Probability distributions

In this section, four major clustering algorithms are implemented and evaluated:

1. K-Means Clustering
2. DBSCAN
3. Hierarchical Clustering
4. Gaussian Mixture Model (GMM)

The implementations are compared with their Scikit-Learn equivalents and evaluated using visualization techniques and clustering metrics such as Adjusted Rand Index (ARI).

---

# 1. K-Means Clustering

## Concept

K-Means is a centroid-based clustering algorithm.

The main idea is:

> Each sample belongs to the cluster with the nearest centroid.

The algorithm tries to minimize the distance between data points and their assigned cluster centers.

The objective function is:

\[
J = \sum_{i=1}^{n} ||x_i-\mu_k||^2
\]

where:

- \(x_i\) is a data point
- \(\mu_k\) is the centroid of cluster k

---

## Algorithm Steps

### 1. Initialization

Randomly select K initial centroids.


Data Points
 ↓

Random Centroids

---

### 2. Assignment Step

Assign every point to the closest centroid.


Point → Nearest Centroid

---

### 3. Update Step

Calculate new centroids:

\[
\mu_k=\frac{1}{N_k}\sum x_i
\]

---

### 4. Repeat

Continue until centroids stop changing.

---

## Implementation

A custom K-Means implementation was created with:

- Distance calculation using Euclidean distance
- Centroid initialization
- Cluster assignment
- Centroid update
- Iterative optimization

The implementation was tested on:

- Synthetic datasets
- Iris dataset

and compared with:

```python
sklearn.cluster.KMeans

Results
On Iris dataset:
- Custom K-Means produced cluster assignments very close to Scikit-Learn.
- Centroids were almost identical.
- ARI evaluation showed consistent clustering behavior.
K-Means successfully separated Setosa but had difficulty distinguishing Versicolor and Virginica because these classes overlap.
2. DBSCAN
Concept
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups samples based on data density.
Unlike K-Means:
- No need to specify number of clusters.
- Can detect noise points.
- Can find non-spherical clusters.
Main Parameters
eps
Maximum distance between neighboring points.
Small eps:

Few neighbors

Large eps:

More neighbors

min_samples
Minimum number of points required to create a dense region.
Point Types
DBSCAN classifies points into:
Core Point
A point with enough neighbors.
Border Point
A point near a dense region but without enough neighbors.
Noise Point
A point that does not belong to any cluster.
Algorithm Steps
1. Select an unvisited point.
2. Find neighboring points within eps distance.
3. If enough neighbors exist:
   - Create a new cluster.
   - Expand the cluster.
4. Otherwise mark as noise.
Implementation
Custom DBSCAN was implemented with:
- Region query
- Neighborhood detection
- Cluster expansion
- Noise handling
The implementation was compared with:
sklearn.cluster.DBSCAN


Results
DBSCAN successfully identified:
- Dense regions
- Noise samples
- Arbitrary shaped clusters
On Iris:
- Both custom and sklearn implementations produced the same clustering structure.
- ARI comparison confirmed matching behavior.
DBSCAN performance depends heavily on selecting appropriate:
- eps
- min_samples
3. Hierarchical Clustering
Concept
Hierarchical clustering builds a hierarchy of clusters instead of directly assigning final groups.
The most common approach is:
Agglomerative Clustering
Starting point:
Each sample = one cluster

Then repeatedly:
Merge closest clusters

until reaching the desired number of clusters.
Linkage Methods
The distance between clusters can be calculated in different ways.
Single Linkage
Uses the closest pair of points:
\[
d(A,B)=min(distance)
\]
Advantage:
- Finds connected structures
Limitation:
- Chaining effect
Ward Linkage
Minimizes the increase in within-cluster variance.
Advantage:
- Produces compact clusters
Dendrogram
Hierarchical clustering can be visualized using a dendrogram.
The dendrogram shows:
- Order of merges
- Distance between clusters
- Possible number of clusters
Implementation
A custom Agglomerative Hierarchical Clustering implementation was created with:
- Distance calculation
- Cluster distance calculation
- Closest cluster search
- Cluster merging
- Merge history tracking
- Dendrogram generation
Compared with:
sklearn.cluster.AgglomerativeClustering


Results
Single Linkage
Single Linkage separated Setosa but merged Versicolor and Virginica.
Reason:
Chaining Effect

Results:
Cluster distribution:
1 / 49 / 100

ARI:
0.558

Ward Linkage
Ward produced more compact clusters.
Results:
Cluster distribution:
71 / 49 / 30

ARI:
0.615

Ward performed better because it reduces cluster variance during merging.
4. Gaussian Mixture Model (GMM)
Concept
GMM assumes that data is generated from a mixture of Gaussian distributions.
Unlike K-Means:
K-Means:

Point → One Cluster

GMM:
Point →

Cluster A: 70%

Cluster B: 25%

Cluster C: 5%

This is called:
Soft Clustering

Gaussian Distribution
Each cluster has:
Mean
Cluster center:
\[
\mu
\]
Covariance
Cluster shape:
\[
\Sigma
\]
Weight
Probability of selecting each Gaussian:
\[
\pi
\]
EM Algorithm
GMM is trained using:
Expectation Maximization

E-Step
Calculate probability of each sample belonging to each Gaussian.
Output:
Responsibilities Matrix

M-Step
Update:
- Mean
- Covariance
- Weights
using calculated probabilities.
Implementation
Custom GMM was implemented with:
- Gaussian probability calculation
- Parameter initialization
- Expectation step
- Maximization step
- EM optimization loop
- Probability-based cluster assignment
Compared with:
sklearn.mixture.GaussianMixture


Results
On synthetic Gaussian data:
Custom GMM successfully recovered all clusters.
Results:
ARI = 1.0

On Iris dataset:
Comparison:
Model	ARI
Custom GMM vs Sklearn	0.762
Custom GMM vs Ground Truth	0.543
Sklearn GMM vs Ground Truth	0.516


The lower ARI is caused by overlap between:
- Versicolor
- Virginica
GMM was still able to model uncertainty through probability assignments.
Final Comparison
Algorithm	Main Idea	Strength	Limitation
K-Means	Centroid based	Fast and simple	Requires K, spherical clusters
DBSCAN	Density based	Detects noise and arbitrary shapes	Sensitive to parameters
Hierarchical	Cluster merging tree	Dendrogram visualization	Computationally expensive
GMM	Probability based	Soft clustering and flexible shapes	Sensitive to initialization


Conclusion
The experiments demonstrated that each clustering algorithm has different assumptions about data structure.
- K-Means works well for compact spherical clusters.
- DBSCAN is useful when density and noise detection are important.
- Hierarchical clustering provides insight into relationships between samples through dendrograms.
- GMM provides probabilistic cluster assignments and can model more complex cluster shapes.
No single clustering algorithm is universally optimal. The appropriate method depends on the structure, distribution, and characteristics of the dataset.
```