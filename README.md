## Project Overview 


#### Project Workflow 

[ Insert image of the overall workflow ]

## Preparing the dataset
#### Segmentation 

After separating the video into 20,400 images I decided to run each image
through segmentation. 

Segmentation is a form of image classification. A typical image classifier has 
a set number of classes. For example we can make a binary classifier that 
distinguishes between Adam Sandler or Brad Pitt. 

<p align="center">
<img width="460" height="300" src="images/AdamBradTogether.jpg">
</p>

But what if the image contains both Brad Pitt and Adam Sandler? How should the
image classifier categorize the image? Segmentation is a way to work around
this problem. This technique anticipates that both our classes ( Brad Pitt and
Adam Sandler ) may be in an image at the same time and therefore approaches
classification differently. Instead of classifying an image as Brad or Adam,
segmentation classifies each pixel as Adam or Brad. 

In the context of our speed detection problem we categorized everything in the
image of the dash cam footage. We used 31 classes to categorize everything from
buildings ( shown in orange ), to lane markings ( shown in red ). We see all
the separate classes as distinct colors but the computer sees the image as a
matrix of numbers. Since we have 31 categories all numbers in the matrix are
from 0 to 30 ( e.g., lane markings are all labeled as 4 ).

Segmenting all our images will allow us to implement other techniques that will
help us detect the speed at each frame. 

In the following steps we take our image set and process in two separate ways
( optical flow and one-hot encoding ), later we will combine both image sets
again. 

#### Optical Flow 
<table>
<tr> <td> <img src="images/dash_cam.gif"></td><td><img
src="images/optical_flow_2.gif"></td></tr>
</table>

For more information and an implementation of optical flow check [this](
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
) out.

