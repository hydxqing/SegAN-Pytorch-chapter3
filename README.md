# SegAN-Pytorch-chapter3

The chapter3 of the segmentation network summary: 
### Combine other mature structures with segmentation networks.
External links: SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation [paper](https://arxiv.org/pdf/1706.01805.pdf).

##### These papers' ideas are combining the basic network with GAN.
Network structure in the paper:![image](https://github.com/hydxqing/SegAN-Pytorch-chapter3/blob/master/picture_in_paper/picture.png)

This network structure is a reconstruction of loss of the basic segmentation network combined with GAN.

First, prediction map and label map of basic segmented network (equivalent to generator in this case) were multiplied with input to get prediction-mask and GT-mask respectively. Then, prediction was made by feeding prediction-mask and GT-mask into the discriminant network, and then loss of multi-scale was calculated.

**References**

This code borrows from [xue](https://github.com/YuanXue1993)'s [work](https://github.com/YuanXue1993/SegAN) and is modified to use on my own dataset.

### Environment: 
  
            Pytorch version >> 0.4.1
