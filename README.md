# Camouflaged Object Segmentation based on Matching-Recognition-Refinement Network
## Camouflaged Object Segmentation based on Matching-Recognition-Refinement Network

> Authors: Xinyu Yan, Meijun Sun, Yahong Han, Zheng Wang.

## Abstract
In the biosphere, camouflaged objects take the advantage of visional wholeness by keeping the color and texture of the objects highly consistent with the background, thereby confusing the visual mechanism of other creatures and achieving a concealed effect. This is also the main reason why the task of camouflaged object detection is challenging. In this paper, we break the visual wholeness and see through the camouflage from the perspective of matching the appropriate field of view. We propose a matching-recognition-refinement network (MRR-Net), which consists of two key modules, i.e., the visual field matching and recognition module (VFMRM) and the step-wise refinement module (SWRM). In the VFMRM, various feature receptive fields are used to match candidate areas of camouflaged objects of different sizes and shapes, and adaptively activate and recognize the approximate area of the real camouflaged object. The SWRM then uses the features extracted by the backbone to gradually refine the camouflaged region obtained by VFMRM, and thus yielding the complete camouflaged object. In addition, a more efficient deep supervision method is exploited, making the features from the backbone input into the SWRM more critical and not redundant. Extensive experimental results demonstrate that our MRR-Net runs in real-time (82.6 FPS) and significantly outperforms 30 state-of-the-art models on three challenging datasets under three standard metrics. Furthermore, MRR-Net is applied to 4 downstream tasks of camouflaged object segmentation, and the results validate its practical application value. Our code is publicly available at: https://github.com/XinyuYanTJU/MRR-Net.

## Framework Overview
![Image text](https://raw.githubusercontent.com/yxy452710960/MRR-Net/master/img/Framework.png)

<p align="center">Figure 1. Overview of our matching-recognition-refinement network (MRR-Net) and its two main building blocks: a visual field matching and recognition module (VFMRM) and a step-wise refinement module (SWRM).</p>

## Qualitative Results
![Image text](https://raw.githubusercontent.com/yxy452710960/MRR-Net/master/img/QualitativeResults.png)

<p align="center">Figure 2. Qualitative Results.</p>


## Proposed Baseline

### Prerequisites
- Python 3.6
- Pytorch 1.7.1
- OpenCV 4.5
- Numpy 1.19
- Apex

### Download dataset
Download the following datasets and unzip them into `data` folder

- [COD10K](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view)
- [CAMO](https://sites.google.com/view/ltnghia/research/camo)
- [NC4K](https://drive.google.com/file/d/1EgfD_GtxTlP7CSJI9RRQuKhjhbsg2DZy/view)


You should rename the folders of each dataset's Image and GT to Imgs and Masks, respectively. Then please run this command to get list.txt for every dataset.
```
python3 dataset_list.py
```

### Download pths
Download the following pths and put them into `pths` folder

- [ResNet-50](https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view)
- [Res2Net-50](https://sites.google.com/view/ltnghia/research/camo)
- [Trained MRR-Net-ResNet-50](https://drive.google.com/file/d/1EgfD_GtxTlP7CSJI9RRQuKhjhbsg2DZy/view)
- [Trained MRR-Net-Res2Net-50](https://drive.google.com/file/d/1EgfD_GtxTlP7CSJI9RRQuKhjhbsg2DZy/view)

### Training
You can revise the datapath、savepath、batchsize、lr、epoch defined in the train_MRRNet_ResNet50.py or train_MRRNet_Res2Net50.py. After the preparation, run this command 
```
python3 train_MRRNet_ResNet50.py 
or
python3 train_MRRNet_Res2Net50.py
```
make sure  that the GPU memory is enough.

### Test
You can revise the model_pth, test_paths in the test_MRRNet_ResNet50.py or test_MRRNet_Res2Net50.py. Then run this command 
```
python3 test_MRRNet_ResNet50.py
or
python3 test_MRRNet_Res2Net50.py
```

### Results
You can also download the results - [Google Drive link]() or - [Baidu Pan link](https://pan.baidu.com/s/16IIRBrQTHdlc08OwYxT1ug) with the fetch code:gt35.
