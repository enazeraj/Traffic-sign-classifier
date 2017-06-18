#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Distribution_data_set.jpg "Distribution"
[image2]: ./examples/Comparison_norm.jpg "Normalization"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Test_figures/12.ppm "Traffic Sign 1"
[image5]: ./Test_figures/2.ppm "Traffic Sign 2"
[image6]: ./Test_figures/28.ppm "Traffic Sign 3"
[image7]: ./Test_figures/10.ppm "Traffic Sign 4"
[image8]: ./Test_figures/16.ppm "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set, the sizes are calculated using the method .shape. While the unique number of classes are calculate looking at the maximum label in the dataset and added with one as the starting index is 0:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data are devided per class 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The only pre-processing I used was to normalize the the pixels as suggested by the notes. I decided to use the VGGnet architecture that I found relatively easy to implement and could achieve a satisfying accuracy. Therefore, I decided to avoid spending time in the pre-processign task. I haven't investigated data augmentation.

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is based on the idea of the VGGnet where several convolutional layers are stacked together and intermediated by a max pooling layer. The idea to use VGGnet came from: https://www.youtube.com/watch?v=u6aEYuemt0M

The VGGnet architecture was used for images of size of 224x224x3 and I rearranged here for figure of size 32x32x3. In addition, to improve the regularization, I added a dropout with descending keeping probabilities before each max pooling  

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x24 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x24 	|
| RELU					|												|
| Dropout				| keeping probability = 90 %					|
| Max pooling	      	| 2x2 stride,  outputs 16x16x24 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x48 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 16x16x48 	|
| RELU					|												|
| Dropout				| keeping probability = 80 %					|
| Max pooling	      	| 2x2 stride,  outputs 8x8x48 					|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 		|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 		|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x96 		|
| RELU					|												|
| Dropout				| keeping probability = 70 %					|
| Max pooling	      	| 2x2 stride,  outputs 4x4x96 					|
| Fully connected		| Input = 1536. Output = 384.       			|
| RELU					|												|
| Fully connected		| Input = 384. Output = 192.       				|
| RELU					|												|
| Dropout				| keeping probability = 60 %					|
| Fully connected		| Input = 192. Output = 43.       				|	
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 64. an AdamOptimizer. A learning rate = 0.001.

The number of epochs were set at 50, but the model training would stop earlier if no improvement are not occurring in the next 5 epochs. The model at which the validation accuracy peaks is kept.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.2 %
* validation set accuracy of 96.8 %
* test set accuracy of 94.7 %

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The initial architecture was a LeNet-5. I could reach just an accuracy of 89%. The choice of LeNet-5 was due to its semplicity.
* What were some problems with the initial architecture?
Perharps, some more advanced pre-processing techniques could have helped to increase the accuracy with this architecture
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I decided to go for a VGGnet due to its semplicity and the advantage to get a higher accuracy compared to LeNet-5. So I stacked together various convnets. I reduced first the batch size, i noticed a smaller batch size was giving me better accuracies during the validation. In addition, I had to tune the neurons of the first layer. In fact, I am always doubling the filer size, so the final amount of neurons is related to the number chosen in the first layer. I tuned it to be 24.
However, I was facing problems of overfitting. So I introduced several dropout layers after each stack of convnet and relu. This helped me to reduce the overfitting, but I was wondering what was the best epochs number, to make sure I was not overfitting I used early-stopping.
* Which parameters were tuned? How were they adjusted and why?
As mentioned, the parameters tuned were the batch size, the neurons in the filter size and the number of layers of VGGnet. I used the standard 3x3convnets-relu as the building block of my net. I kept the value of learning rate as the LeNet-5.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The dropout helped to avoid overfitting

If a well known architecture was chosen:
* What architecture was chosen? 
A modified VGGnet
* Why did you believe it would be relevant to the traffic sign application?
It was in the past successful in other classification problems, so I thought was going to be successful also for the traffic sign project
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
It avoids overfitting and reach an accuracy above 94% in the test dataset
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road     	| Priority road   								| 
| Speed limit (50km/h)  | Speed limit (50km/h) 							|
| Children crossing		| Children crossing								|
| No passing for vehicles over 3.5 metric tons| Bumpy Road					 				|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model predicts with a confidence of almost 1.0 the german traffic signs. This is reasonable as the figures were taken by the german traffic sign dataset, where all the images have a size close to 32x32. Other images taken from internet were losing too much resolution when scaling down again to a size of 32x32 and were indistinguishable also to human eye. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.         			| Priority road   								| 
| 1.     				| Speed limit (50km/h) 										|
| 1.					| Yield											|
| 1.	      			| Children crossing					 				|
| 1.				    | Vehicles over 3.5 metric tons prohibited      							|

 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


