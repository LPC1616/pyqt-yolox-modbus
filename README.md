## 建筑废弃物智能分拣系统 

该系统分为 **识别** 模块和 **通信** 模块，识别模块用到的是 **YOLOX** 算法，通信网络用的是工业界常用的 **Modbus** 通信网络。

采用 **Qt** 工具搭建了检测可视化界面，界面有三个功能按钮，起作用分别是图像检测、摄像头检测和视频检测。

### 环境搭建

**Anaconda**（环境管理） + **Cudnn**-8.0.5.39（NVIDIA 专门负责管理分配运算单元的框架） + **CUDA**-11.0（深度神经网络的GPU加速库） + **pytorch**-1.7.1（python版本的神经网络框架）

安装torch==1.7.1，指令如下

`# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

### 训练步骤
