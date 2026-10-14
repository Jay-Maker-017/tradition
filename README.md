# tradition：传统方案 + 数字神经网络装甲板类型识别

在原传统方案的基础上，迁移深大开源工程 的 **NNdeployment 部署库**与 **Infantry 装甲板模型**，用数字神经网络识别装甲板的类型

## 1. 工作流程

```
图片/视频帧
   │
   |-> [传统方案] 灯条检测 -> 装甲板配对
   │
   |-> [神经网络] ArmorModel 整帧推理 -> 网络装甲板（角点 + 类型/颜色/尺寸）
   │
   |-> [匹配] 按中心距将网络类型写回传统装甲板 -> 输出类型（数字）+ 可视化

```

思路参考 "26_Detection_Models-main"(深大开源工程),
一句话说就是"**传统方案负责定位，神经网络负责分类**":
用 "NNdeployment" 的 "ArmorModel" 对整帧图像做一次推理，
再与传统方案检测出的装甲板按中心距匹配，
把类型信息（armor_id , class_name , 置信度）补充到传统装甲板上


类型 ID 对应关系：

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 类型 | 哨兵 | 英雄 | 工程 | 3号步兵 | 4号步兵 | 5号步兵 | 前哨站 | 基地(底部) | 基地(顶部) |

颜色 ID：0=蓝 1=红 2=白 3=紫

## 2. 环境依赖

| 依赖 | 版本要求 |
| --- | --- |
| Ubuntu | 22.04 |
| GCC | ≥ 11（C++20） |
| CMake | ≥ 3.22 |
| OpenCV | 4.x |
| OpenVINO | C++ Runtime |

### OpenVINO 安装方式（二选一）

**方式一：pip 安装**

```bash
pip install openvino
```

**方式二：apt 安装**

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update
sudo apt install -y openvino-2025.4.1
```


> 注意：如果你像我一样使用的虚拟机,没有 Intel GPU，模型配置中的 "device" 必须为 "CPU"（已配置好）
> 只有物理机具备 Intel 核显/独显时才可改为 "GPU"

## 3. 构建

```bash
cd /home/jay/Desktop/auto_aim_homework/tradition
cmake -S . -B build
cmake --build build -j$(nproc)
```
> 注意：记得把路径改成自己的!
> main函数默认调用视频路径，但那个路径我为了方便，是用绝对路径写的，运行的时候也记得改一下~

## 4. 运行

```bash
./build/main                                # 默认视频 video/video_combat.avi + armor_v8(是的，我一共放了三个模型)
./build/main img/test_1.jpg                 # 指定图片，使用 armor_v8(不写就默认这个)
./build/main video/video_combat.avi         # 指定 AVI 视频
./build/main img/test_1.jpg armor_v5        # 指定图片和模型（armor_v5 / armor_v8）
```

- 程序先尝试将输入按图片读取；读取失败时自动尝试作为视频打开，
- 因此就可以兼容兼容 png/RGBA 图片啦
- 图片输入时，终端打印灯条中心、尺寸、装甲板角点、类型和置信度；结果图保存到 "lightbar_result.jpg"
- 结果图/视频中的绿框为传统方案装甲板四边形，
- 黄色标签为神经网络识别出的类型 , 未识别时标签为绿色"?"(但是为了结果图看起来比较干净，我把这种情况的显示关掉了)

## 5. 部分文件说明


| 文件 | 说明 |
| --- | --- |
| NNdeployment/ | 部署库（完整复制自 深大开源工程,仅注释掉启动横幅打印） |
| models/openvino/ | Infantry-v5n / v8n 模型（完整复制自 深大开源工程） |
| configs/detect.json | 模型配置 |
| include/digit_recognizer.hpp | 数字识别模块接口 |
| src/digit_recognizer.cpp | 整帧推理 + 中心距匹配实现 |
| src/visualize.cpp | 装甲板类型与置信度 |
| src/main.cpp | 图片/视频智能切换；逐帧执行检测、配对、数字识别、打印与可视化 |
| lightbar_result.avi | 视频检测结果 |

## 6. 常见参数调整

- **灯条识别阈值**:"include/lightbar.hpp"里调
- **灯条配对阈值**:"include/armor.hpp"里调
- **装甲板数字神经网络识别阈值**:"include/digit_recognizer.hpp"里调
- **置信度阈值**:"configs/detect.json" 中调 "score_threshold"(默认 0.5),
- **换模型**：把新模型的 ".xml" / ".bin" 放入 "models/openvino/"，
  并修改 "configs/detect.json" 中对应节点的 "xml' 相对路径即可


