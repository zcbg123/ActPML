# PML-CD:https://github.com/zcbg123/PMLNet.git
### Requirement  
```bash
-Pytorch 1.8.0+  
-torchvision 0.9.0+  
-python 3.8+  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1+  
-Cudnn 11.3+  
```
## Training,Test and Visualization Process   

```bash
python train.py

python test.py 
```
You can change data_name for different datasets like "LEVIR", "WHU", "SYSU", "S2Looking". Pay attention to the location of the weight update.



## Dataset Path Setting
```
 LEVIR-CD or WHU-CD or SYSU-CD or S2Looking-CD
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  ```


