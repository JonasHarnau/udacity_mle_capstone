
This is the capstone project for my Udacity Machine Learning Engineer Nanodegree. 
The full project report in pdf form is included in this repository ([click](https://github.com/JonasHarnau/udacity_mle_capstone/blob/master/Project%20Report.pdf)).

# Abstract

We propose a two-stage model that improves classiﬁcation accuracy compared to a standard convolutional neural network by making use of bounding box data that is available for a small subset of classes where the classes are relatively homogeneous. Speciﬁcally, we apply the model to a dataset of about 22,000 turtle and tortoise images from 22 species; we have bounding box information for some 2,000 images distributed over ﬁve species. The ﬁrst stage is a binary implementation of Faster R-CNN (Ren et al., 2017) which has the purpose to detect the turtles or tortoises in the image and puts forward a single region proposal. The second stage model never gets to see the full image but rather only the cropped region proposed by the ﬁrst stage. The model outperforms the benchmark signiﬁcantly, improving top-1 and top-3 error by about 2pp and 1.25pp, respectively.

# Example Output
Below we show the classification results for the two-stage model as well as for the benchmark for several images. 
For the application, the first stage looks for the boxes containing turtles or tortoises (turtletoises for short).
Then, we pick the most likely box as ranked by the turtletoiseness and crop the image to that box. The cropped image is 
then warped and fed to the second stage which classifies the species. The benchmark essentially consists of the second stage only.
That is, the full image is warped and fed to the CNN. We point out that we did not have any bounding box information for
*Pacific Ridley*, *Giant Tortoise*, or *Desert Tortoise*. Still, the model draws a box and classifies it just like 
an R-CNN.

![examples](https://user-images.githubusercontent.com/25103918/32015605-942c6a98-b98f-11e7-8c9b-086feb8f0526.png)



# Required Packages

Apart from standard packages (Keras, pandas, matplotlib, cv2) we made use of ``keras-frcnn`` from https://github.com/yhenon/keras-frcnn. We adjusted this repository to our needs. 
This modified repository is provided in the folder keras-frcnn.

# Data 

We downloaded the image data in archive form from [ImageNet](http://www.image-net.org). 
Unfortunately, for legal purposes, we cannot make the image data directly available. 
However, the results should be roughly replicable if the images are downloaded directly 
from their sources (links are in the table). For the code to run, the images should then be added to a tar-archive 
and integrated so the folder structure is  ``./data/archives/images/*.tar`` where ``*`` 
is replaced with the WnID. 
Then, running ``01. Data -preparing.ipynb`` extracts the data into the correct folders.


| Species                       | WnID      | Bounding box data |
| ----------------------------- | --------- |:-----------------:|
| Loggerhead	                | [n01664065](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01664065) | x                 |
| Green Turtle	                | [n01663782](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01663782) |                   |
| Pacific Ridley	            | [n01664674](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01664674) |                   |
| Atlantic Ridley	            | [n01664492](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01664492) |                   |
| Hawksbill Turtle	            | [n01664990](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01664990) |                   |
| Leatherback Turtle	        | [n01665541](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01665541) | x                 |
| Common Snapping Turtle	    | [n01666228](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01666228) |                   |
| Alligator Snapping Turtle	    | [n01666585](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01666585) |                   |
| Mud Turtle	                | [n01667114](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01667114) | x                 |
| Terrapin	                    | [n01667778](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01667778) | x                 |
| Red-Bellied Terrapin	        | [n01668436](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01668436) |                   |
| Slider	                    | [n01668665](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01668665) |                   |
| Cooter	                    | [n01668892](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01668892) |                   |
| Box Turtle	                | [n01669191](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01669191) | x                 |
| Painted Turtle	            | [n01669654](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01669654) |                   |
| European Tortoise	            | [n01670535](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01670535) |                   |
| Giant Tortoise	            | [n01670802](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01670802) |                   |
| Gopher Tortoise	            | [n01671125](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01671125) |                   |
| Desert Tortoise	            | [n01671479](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01671479) |                   |
| Texas Tortoise	            | [n01671705](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01671705) |                   |
| Spiny Softshell	            | [n01672432](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01672432) |                   |
| Smooth Softshell	            | [n01672611](http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n01672611) |                   |



For those with imagenet access, the archives can directly be downloaded through 
``www.image-net.org/download/synset?wnid=[wnid]&username=[username]&accesskey=[accesskey]&release=latest&src=stanford``.

# Saved Models

The saved models are too large for GitHub. Instead, we made those available [here](https://1drv.ms/f/s!AqtF5RTosLYjpPdYn7NZJfsldKV5DQ).
