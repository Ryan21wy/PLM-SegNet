# PLM-SegNet: An end-to-end method for palm-leaf manuscript segmentation based on U-Net
## Background
The cultural heritage suffer from the inevitable destruction more or less over time. It is of great necessity to carry out the preservation and the restoration of cultural heritage to prolong their life span. Image acquisition of these cultural heritage is one of the most commonly used technique due to its non-destruction to the fragile relics. The current status of the cultural heritage can be recorded and and saved as electronic data. The electronic data can be further permanently stored in databases for other applications such as digital display and information mining.  

Palm-leaf manuscripts are one of the most valuable relics in the world. However, there are various factors including change of the climate or damages from the microorganism, which contribute jointly to the inevitable destruction of the palm-leaf manuscripts. Image acquisition becomes significant for preservation and restoration of these manuscripts. The acquired images are not always ideal for the existence of the background, which seriously affects the aesthetics of displaying and the subsequent processing of the images. 

The **p**alm-**l**eaf **m**anuscript **seg**mentation **net**work (*PLM-SegNet*) is proposed to segment palm-leaf manuscript from raw image. PLM-SegNet follows the typical U-Net where a image of palm-leaf manuscript can be fed into the network and a foreground distribution map can be consequently and automatically generated.

![overview](https://user-images.githubusercontent.com/81405754/233940518-0af70622-e5a2-42aa-b53b-d3abb1f81683.png)

## Depends
[Anaconda for Python 3.8](https://www.python.org/)  
[conda install PyTorch](https://pytorch.org)  
[conda install OpenCV](https://opencv.org/)  
[conda install Pillow](https://pypi.org/project/Pillow/)  
[conda install numpy](https://numpy.org/)  
[conda install scipy](https://scipy.org/)  

## Dataset
The images of palm-leaf manuscripts would not be available for its status being one of national first-class cultural relics. Moreover, we have already signed a confidentiality agreement that we have no rights to make these data open.  

## Usage
The PLM-SegNet model is public at [release](https://github.com/Ryan21wy/PLM-SegNet/releases/download/v1.0.0/model.zip), every user can download and use it.  
A test ipython notebook at [demo](https://github.com/Ryan21wy/PLM-SegNet/blob/master/test.ipynb) is available.
