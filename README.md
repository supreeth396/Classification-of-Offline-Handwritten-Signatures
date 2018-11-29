# Classification-of-Offline-Handwritten-Signatures
As signature is the primary mechanism for authentication and authorization in legal
transactions and documents, the need for an efficient automated solution for signature
verification has increased and more over fraud detection has been a real time problem in every
sector where the handwritten signatures are used. Therefore, developing a robust system that
automatically authenticates the signatures based on the sample images of original signatures of
the person is the objective of the project.
The robust system should neither be too sensitive nor too coarse. It should have an acceptable
trade-off between a low False Acceptance Rate (FAR) and a low False Rejection Rate (FRR).
The false rejection rate (FRR) and the false acceptance rate (FAR) are used as quality
performance measures.

# Signature Acquisition Phases
⚫ Pre-processing

⚫ Feature extraction

⚫ Classification

Once the featured are extracted for both genuine and forgery, it has to be classified using
classifiers.

-------Statistical Classifier----------

The pseudo code is as follows:

Step 1: Read the features from the excel sheet.

Step 2: Fix the threshold value for each feature.

Step 3: Given the test input, extract the feature and record them in the excel sheet. Again, read
the features. Compare each feature with each threshold value.

Step 4: If both matches then the given input signature is supposed to be “genuine” else
“forgery”.


---------Linear Classifier-----------

The pseudo code is as follows:

Step 1: Read the features from the excel sheet.

Step 2: All features are given to an array list, i.e x

Step 3: Define a label such that “1” is for genuine and “0” is for forgery, i.e y

Step 4: Using the function clf.fit(x,y), fit the points on the hyperplane where x is array and y is
label.

Step 5: Linear classifier plots the graph according to the points.


----------Logistic Regression-----------

The pseudo code is as follows:

Step 1: Read the features from the excel sheet.

Step 2: Each feature is given to an array namely x1, x2, x3…

Step 3: Define a label such that “1” is for genuine and “0” is for forgery.

Step 4: Append each array to the list.

Step 5: Initialize by fitting the number of features i.e theta

Step 6: Compute cost_function, gradient and sigmoid function. Display all the values.

⚫ Verification 
