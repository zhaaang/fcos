### train for PASCAL VOC
You can run the train_voc.py, train 30 epoch and you can get the result. You need to change the PASCAL07+12 path, you can reference to this repo:https://github.com/YuwenXiong/py-R-FCN

### predict
You can run the predic.py to predict the box/score/class in a new picture.

###  AP Result
| PASCAL VOC (800px) | COCO(800px) |
| :-----------: | :-----------------: |
|     78.7 (IoU.5)      |      **37.2**       |

### Requirements  
* opencv-python  
* pytorch >= 1.0  
* torchvision >= 0.4. 
* matplotlib
* cython
* numpy == 1.17
* Pillow
* tqdm
* pycocotools


### Results in Pascal Voc
You can download the 78.7 ap result in [Baidu driver link](https://pan.baidu.com/s/1aB0irfcJQM5WTlmiKFOfEA), password:s4cp, then put it in checkpoint folder, then run the eval_voc.py and
you can get the result as follows:
```python
ap for aeroplane is 0.8404275844161726
ap for bicycle is 0.8538414069634157
ap for bird is 0.8371043868766486
ap for boat is 0.6867630943895144
ap for bottle is 0.7039276923755678
ap for bus is 0.8585650817738049
ap for car is 0.8993155911437366
ap for cat is 0.919100484692941
ap for chair is 0.5575814527810952
ap for cow is 0.8429926423801004
ap for diningtable is 0.6596296818110386
ap for dog is 0.8896160095323242
ap for horse is 0.8436443710873067
ap for motorbike is 0.8114359299817884
ap for person is 0.8525903122141745
ap for pottedplant is 0.47628914937925404
ap for sheep is 0.8257834833986701
ap for sofa is 0.7000391892293902
ap for train is 0.8664281745198105
ap for tvmonitor is 0.8186715890179656
mAP=====>0.787
```
I also use the cosine lr to train the voc, and it got 76.7mAP, which is lower than linear. I think the cosine lr matches Adam is better.

