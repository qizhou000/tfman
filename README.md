
# TFMAN

Official pytorch implementation of paper [*Single image super-resolution based on trainable feature matching attention network (TFMAN)*](https://www.sciencedirect.com/science/article/pii/S0031320324000402) [(Google Scholar)](https://scholar.google.com.hk/scholar?hl=zh-CN&as_sdt=0%2C5&q=Single+image+super-resolution+based+on+trainable+feature+matching+attention+network&btnG=). 

# Fast Run
The organization structure of the code is very simple. You can directly run 
```python test.py```
to test the SR performance of TFMAN model trained with x2 Bicubic degradation model on Set5 and Set14 datasets.

# Train
Please unzip the downloaded training dataset **(DIV2K / DIV2K_BD / DIV2K_DN**, see below) into the directory **SRTrain**. Then, you can train TFMAN by just running:
```python train.py```


# Checkpoints and Datasets

You can download the TFMAN **checkpoints** and the **training/testing datasets** from the following website:
|Download| website  |
|---------------|----------|
| **checkpoints**   |[*BaiduYun*](https://pan.baidu.com/s/1aQO_dhzwt6R07jqejtBcTw?pwd=tfma)|
| **datasets** |[*BaiduYun*](https://pan.baidu.com/s/1vH_Sq4LQ65B4m4TJTBLAyA?pwd=tfma)|

# Other Training or Testing Requirements
If you want to train or test custom dataset, please organize the directory structure of the dataset into a format consistent with our datasets and run `train.py` or `test.py`.

# Cite
If this paper is fortunate enough to help improve your work, please cite us.
[[Code]](https://github.com/qizhou000/tfman)[[Paper]](https://www.sciencedirect.com/science/article/pii/S0031320324000402)
```
@article{TFMAN,
    title = {Single image super-resolution based on trainable feature matching attention network},
    journal = {Pattern Recognition},
    year = {2024},
    doi = {https://doi.org/10.1016/j.patcog.2024.110289},
    url = {https://www.sciencedirect.com/science/article/pii/S0031320324000402},
    author = {Qizhou Chen and Qing Shao},
}
```
