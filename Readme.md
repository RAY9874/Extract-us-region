This work is used in paper <Multimodal Feature Attention for Cervical Lymph Node Segmentation in Ultrasound and Doppler Images>. If you find this work useful for your work, star us before paper published. Cite us after published~

### 功能
从超声机上采集的图片往往存在一些型号、深度等信息，本项目能够自动从超声图中裁剪超声成像区域。

### 使用说明

1. 拷贝US_SSD至本地
1. copy US_SSD.
2. 读取权重
2. set weight path
3. 设置类别为2（背景+超声区）
3. set classes to 2(bg + us region)
4. 调用extract_us_region
4. call extract_us_region method
5. 样例代码如detect.py的main函数中，假设路径为---./
5. a sample code is in detect.py, assume your path is ---./
  ​							---data
  ​							---US_SSD
  ​							---yourcode.py

### tips
1.一般情况下，置信度>0.99，在我本地数据集中，13000例超声图，<0.99的有160个 

### 模型权重 model weights
链接：https://pan.baidu.com/s/1jYEO0M3S1jk159xO6ggf4g 
提取码：yxzn

