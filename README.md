## Project Overview 


#### Project Workflow 
We will begin with a video in mp4 format. The video contains dash cam footage
of a vehicle driving in downtown San Francisco. Our goal is to predict the
speed of the car at each frame ( 20, 400 total ).

Our workflow will be as follows: 

> - Video -> 20,400 images -> Segmentation
>
> - Segmentation -> optical flow matrix set 
>
> - Segmentation -> one hot encoding matrix set 
>
> - Combine one hot encoding with optical flow 
>
> - Long Short Term Memory ( LSTM ) model 
>
## Preparing the dataset
#### <ins>Segmentation</ins>  

After separating the video into 20,400 images we will process each image through segmentation.

Segmentation is a form of image classification. A typical image classifier has 
a set number of classes. For example we can make a binary classifier that 
distinguishes between Adam Sandler or Brad Pitt. Our classifier will label an image as Adam Sandler if it thinks the entire image resembles Adam Sandler. Likewise with Brad Pitt. 

<p align="center">
<img width="460" height="300" src="images/AdamBradTogether.jpg">
</p>

But what if the image contains both Brad Pitt *and* Adam Sandler? How should the
image classifier categorize the image? Segmentation is a way to work around
this problem. This technique anticipates that both our classes ( Brad Pitt and
Adam Sandler ) may be in an image at the same time and therefore approaches
classification differently. Instead of classifying an *entire* image as Brad or Adam,
segmentation classifies each pixel as either Adam or Brad. 

<p align="center">
<img width="460" height="300" src="images/SegmentationFastai.png">
</p>

So taking the information we gleaned from the Brad and Adam example, let's apply it to our current image set. We have driving footage so we need to figure out a way to categorize each object that appears in our images. [Fast.Ai](https://medium.com/analytics-vidhya/image-segmentation-using-fastai-ddded25f811e) created a segmentation model with the following model. We can see each of the 31 classes below. 

```
classes = [ 'Animal', 'Archway', 'Bicyclist', 'Building', 'Car', 'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 
'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_bus', 'Tunnel', 'VegetagtionMisc', 'Void', 
'Wall' ] 
```

Once the image is segmented each of the 31 classes are represented in two ways. If the image is in matrix format the class is shown as a number ( e.g., 0 - 30 ), and in image format we differentiate classes by color ( e.g., sidewalks are purple, buildings are orange ). 

Segmenting all our images will allow us to implement other techniques that will
help us detect the speed at each frame. 

In the following steps we take our image set and process in two separate ways
( optical flow and one-hot encoding ), later we will combine both image sets
again. 

#### <ins>Optical Flow</ins> 
<table>
<tr>
<th> dash cam footage </th>
<th> optical flow example </th>
</tr> 

<tr>
<td> <img src="images/dash_cam.gif">
</td>
<td>
<img src="images/optical_flow_2.gif">
</td>
</tr>

</table>

Optical flow is a computer vision tool used for sequential data. In our case
we're using it to track any movement from image to image. This allows us to see
which of our classes ( pedestrians, street signs ) are moving through 
our sequence of images. 

Let's imagine again our image as a matrix. Our matrix has dimensions of **20 x
30**. Optical flow uses small filters ( i.e., a smaller matrix, usually **3 x 3** or
    **5 x 5** ) that hovers over every part of the **20 x 30** matrix and identifies
small dense clusters of similar pixels and then compares it to the next
matrix in the sequence. 

Visually this ends up looking like the image above on the right, only movement is
tracked. Comparing the optical flow gif to the video frame gif on the left we
can see that lane markings are being picked up by the optical flow model!
That's great news for us because we want to give our LSTM model an object to
reference so that it can develop a speed estimate for each frame. 



For more information and an implementation of optical flow check [this](
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
) out.



#### <ins>One Hot Encoding</ins> 
<p align="center">
<img width="460" height="300" src="images/SegmentationFastai.png">
</p>

Recall our segmented image example, we have 31 classes total. Since our image
set spans 20,400 images, not all classes will be represented in each image.
For instance, sometimes there will be buildings in the image and other times
not.

One hot encoding is a way of formatting our image in matrix format so that we
know when each of the 31 classes is present or not. We will take our image as a
matrix of size **W x H**, where **W** = width and **H** = height. Inside the matrix
representation of the above image each pixel is labeled as 0-30 for each class. 

One hot encoding will create 31 layers of this image, so now our image will be
**30 x W x H**, one for each class. It is important to emphasize that each class
will get it's own layer. So for example lane markings are labeled as 4, so
therefore lane markings will be in layer 4 of the matrix. In that layer
wherever there are lane markings there will be 1's, everywhere else will be
labeled as 0's. This process is repeated for each class at each layer of the
matrix. 

[Jeremy Jordan](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.00-PM.png) has created a great illustration of this concept, shown below. 
<p align="center">
<img width="660" height="400" src="images/oneHotExample.png">
</p>

It is vital to do this because we will combine this matrix set with our optical
flow matrix set through matrix multiplication. If the object class is not
present in the image it will be labeled as 0, and therefore that optical flow
data will not be relevant or present in our final matrix set. If the object
class is present, we end up preserving the movement in the matrix
multiplication. 
