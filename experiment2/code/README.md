### 一、运行环境

OS：Windows 10 旗舰版

IDE：Pycharm 2020.2

主要Python环境：

- Python 3.7.9
- Numpy 1.19.2
- Pytorch 1.6.0
- Tensorboard 2.3.0
- Tensorflow 2.3.1
- scikit-learn 0.23.2
- pickle

### 二、目录结构

data

​		-THUCNews

​		-aclImdb

​		-run.py

​		-TextCNN.py

​		-train_eval.py

​		-utils.py

### 三、资源下载

**aclImdb与THUCNews**文件夹由于过大，未上传至Github上托管。为了方便，放在百度网盘上方便下载。

------

链接：https://pan.baidu.com/s/17RwAWKnrJgki2-6aK0mM3g 
提取码：6666 

-------

按目录结构，将下载到的**aclImdb与THUCNews**解压放在`data`目录下即可。

### 四、运行

#### 4.1 训练中文数据集

```bash
python run.py --language zh
python run.py --language en
```



