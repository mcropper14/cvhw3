# 18-794 HW3 Semantic Segmentation

In this assignment, we'll be building our own DeepLab network, a framework designed for high resolution, precise image segmentation, and using it to predict a categorical label for every single pixel in an image.

This type of image classification is called semantic image segmentation. It's similar to object detection in that both ask the question: "What objects are in this image and where in the image are those objects located?," but where object detection labels objects with bounding boxes that may include pixels that aren't part of the object, semantic image segmentation allows you to predict a precise mask for each object in the image by labeling each pixel in the image with its corresponding class. 

To train a segmentation network, you will need an annotated dataset where a training pair contains an RGB image and the annotated segmentation map. Each pixel is labeled by a categorical number similar to the classification.

* To finish this assignment, you need to submit a zip file containing both the finished code and a report.

## Your tasks
* Build a data processing pipeline and visualize the annotation
* Build your DeepLabV3 network
* Train and evaluate your own network on PascalVOC dataset
* Apply the latest segment-anything-model.

### 0. environments and packages
We recommend you start with an Anaconda environment but feel free to use anything. Then in your environment, run 

```bash
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```
Your system should be equipped with an Nvidia GPU, and plz follow the official PyTorch instruction to install the GPU version properly. We have tested the code with Pytorch 1.10 & TorchVision 0.11.0 for this assignment.

In the ``datasets/`` folder, create a subdirection ``data``. Download and unzip the PascalVOC dataset. We will be using the [2012 branch](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), if this is too slow or unavailable, try mirror [link](https://dataset.bj.bcebos.com/voc/VOCtrainval_11-May-2012.tar). Your path should be:

```
/datasets
    /data
        /VOCdevkit 
            /VOC2012 
                /SegmentationClass
                /JPEGImages
                ...
            ...
        /VOCtrainval_11-May-2012.tar
        ...
```



## Reference

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[3] [Segment Anything](https://arxiv.org/abs/2304.02643)
