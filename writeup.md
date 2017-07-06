#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/x-train-stats.png "Stats for training set"
[image2]: ./images/x-valid-stats.png "Stats for validation set"
[image3]: ./images/x-test-stats.png "Stats for test set"
[image4]: ./images/samples-speed-20.png "Sample images from test set"
[image5]: ./examples/grayscale.jpg "Grayscale"
[image6]: ./images/30.jpg "Traffic Sign 1"
[image7]: ./images/80.jpg "Traffic Sign 2"
[image8]: ./images/120.jpg "Traffic Sign 3"
[image9]: ./images/kids-crossing.jpg "Traffic Sign 4"
[image10]: ./images/right.jpg "Traffic Sign 5"
[image11]: ./images/stop.jpg "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

I used panda library to draw bar charts showing number of records for the training, validation and test set

![alt text][image1]
![alt text][image2]
![alt text][image3]

I also randomly sampled 5 images of each type of traffic sign and drew them in the notebook. Here's an example of speed limit sign (20km/h):

![alt text][image4]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is irrelevant to determining the type of the traffic sign. So by reducing to grayscale, we can eliminate unnecessary features to make the program train faster and not being distracted by the color variations of the signs. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image5]

As a last step, I normalized the image data because we want values during the optimization not to get too big or to small. Ideally, they should have zero means and equal variance. 

*Update 07/05*

I've now added data augmentation by enhancing existing training images. I've added more images by flipping and rotating (randome degrees between -30 degrees and 30 degrees) the original X_train image set.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image 						| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max Pooling     		| 2x2 kernel, 2x2 stride, output 14x14x6		|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 kernel, 2x2 stride,  outputs 5x5x16 		|
| Fully connected		| input 400, output 120       					|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 120, output 84       					|
| RELU					|												|
| Dropout				|												|
| Fully connected		| input 84, output 43       					|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is based LeNet-5 architecture implemented in lesson 8. To train the model, I used `tf.train.AdamOptimizer` with learning rate of 0.001, epochs being 10 and batch size 128.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

With batch size 128 and EPOCH 10, I got around 0.90 validation accuracy:

	EPOCH 8 ... Training Accuracy = 0.991 Validation Accuracy = 0.912

	EPOCH 9 ... Training Accuracy = 0.989 Validation Accuracy = 0.893

	EPOCH 10 ... Training Accuracy = 0.992 Validation Accuracy = 0.901

As it shows high training accuracy but low validation accuracy, we know the model is overfitting. To address the overfitting, I've add a drop out layer after fully connected layer 1 and 2 with the keep probability of 0.5 for training and 1.0 for validation and test. It got me to around 0.92 validation accuracy if I remember correctly. Then I added L2 regularization. And tweaked values for the hyper parameters and finally settled with learning_rate=0.0005, EPOCH=60 and BATCH_SIZE=256. Then I finally was able to get around 0.94 validation accuracy.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.943
* test set accuracy of 0.932

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I've chosen the LeNet-5 architecture as it's a well known architecture for image classification purpose. And identifying traffic sign is a good example of image classification, for which LeNet-5 should work well. Without any modification, I was able to get around 90% validation accuracy, which proves that LeNet-5 architecture works well for our problem. And by adding dropout and L2 regularization and extend EPOCH to 60, I was able to get above 93% test accuracy without much difficulty.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

The 120km/h sign image might be difficult to classify because it's not well lit.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed 30km/h      	| Speed 30km/h 									| 
| Speed 80km/h     		| Speed 60km/h 									|
| Speed 120km/h     	| Speed 120km/h 								|
| Children crossing		| Children crossing								|
| Turn right ahead	    | Turn right ahead					 		    |
| Stop      			| Stop               							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

First Image: Speed Limit 30km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .19         			| 30km/h       									| 
| .11     				| 50km/h 										|
| .07					| 70km/h										|
| .05	      			| 20km/h    					 				|
| .03				    | 80km/h              							|

Second Image: Speed Limit 80km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .096         			| 60km/h       									| 
| .096     				| 50km/h 										|
| .092					| 80km/h										|
| .037	      			| 30km/h    					 				|
| -.02				    | End of speed limit (80km/h)             		|


Third Image: Speed Limit 120km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .10         			| 120km/h       								| 
| .07     				| 20km/h 										|
| .05					| 70km/h										|
| -.01	      			| 100km/h    					 				|
| -.03				    | NO ENTRY             		                    |

Fourth Image: Children crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .23         			| Children crossing       						| 
| .07     				| Dangerous curve to the right 					|
| .06					| Bicycles crossing								|
| .05	      			| Beware of ice/snow    					 	|
| .04				    | Slippery road             		            |

Fifth Image: Turn right ahead

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .18         			| Turn right ahead       						| 
| .04     				| Roundabout mandatory       					|
| .03					| Keep left     								|
| -.00	      			| Right-of-way at the next intersection    		|
| -.01				    | Stop                       		            |

Sixth Image: Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .24         			| Stop                     						| 
| .04     				| No entry                     					|
| .02					| Turn left ahead     						    |
| .00	      			| Yield                                 		|
| -.01				    | Keep right                       		        |

*Update 7/5*

After doing data augumentation, the test accuracy didn't improve much (in fact, it stayed flat at 0.932). But the softmax probability has got a significant bump. Here're the numbers after doing data augmentation:

First Image: Speed Limit 30km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .30        			| 30km/h       									| 
| .15     				| 50km/h 										|
| .02					| 20km/h										|
| .01	      			| 80km/h    					 				|
| .00				    | 70km/h              							|

Second Image: Speed Limit 80km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .37         			| 30km/h       									| 
| .20    				| 20km/h 										|
| .04					| 70km/h										|
| .01	      			| 50km/h    					 				|
| -.02		    	    | Bicycles crossing                        		|


Third Image: Speed Limit 120km/h

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .22         			| 20km/h       		    						| 
| .14   				| 30km/h 										|
| .10					| 120km/h										|
| .09      		    	| 70km/h    					 				|
| .07				    | Keep left             		                |

Fourth Image: Children crossing

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .24        			| Children crossing       						| 
| .18     				| Bicycles crossing 			        		|
| .07					| Dangerous curve to the right					|
| .06	      			| Right-of-way at the next intersection    	    |
| .05				    | Road narrows on the right             		|

Fifth Image: Turn right ahead

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .14        			| Turn right ahead       						| 
| .07     				| Keep left                    					|
| .02					| Double curve     								|
| .01	      			| Go straight or left                   		|
| .00				    | Speed limit (50km/h)                          |

Sixth Image: Stop

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .26         			| Stop                     						| 
| .08     				| Turn right ahead                     			|
| .07					| Speed limit (30km/h)     						|
| .05	      			| Yield                                 		|
| .02				    | Bumpy road                       		        |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


