## Pixel2PixelHD -一种基于分割标签与边界标签相结合的图像风格迁移模型 –Pytorch实现
---

## 目录  
1. [所需环境 Environment](#所需环境) 
2. [注意事项 Attention](#注意事项) 
3. [效果展示 Effect](#效果展示)
4. [数据下载 Download](#数据下载) 
5. [训练步骤 Train](#训练步骤) 
6. [参考文献 Reference](#参考文献) 

## 所需环境  
1. Python3.7
2. Pytorch>=1.10.1+cu113  
3. Torchvision>=0.11.2+cu113
4. timm>=0.6.11
5. numpy==1.19.5
6. Pillow==8.2.0
7. Opencv-contrib-python==4.5.1.48
8. CUDA 11.0+
9. Cudnn 8.0.4+

## 注意事项  
1.实现基于segNet与UNet的两种风格迁移生成器 
2.实现多感受野、多尺度输出的对抗器
3.可参考SelfAttentionGAN将对抗器的卷积操作替换为谱归一化(SpecNorm)卷积
4.可参考SelfAttentionGAN将注意力机制应用于生成器与对抗器
5.实现增强器模型， 用于改变生成图像的风格，此处并未使用
6. 数据与标签路径、训练参数等均位于config.py  

## 效果展示
![image]()  
![image]()  

## 数据下载    
Cityscapes  
链接：Cityscapes Dataset – Semantic Understanding of Urban Street Scenes (cityscapes-dataset.com)
下载解压后将数据集放置于config.py中指定的路径。 

## 训练步骤  
运行train.py  

## 参考文献  
Chrome-extension://dnkjinhmoohpidjdgehjbglmgbngnknl/pdf.js/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1711.11585.pdf 
