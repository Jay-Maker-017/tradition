# tradition：传统方案 + 数字神经网络装甲板类型识别

在原传统方案（灯条检测 → 装甲板配对 → 四角点 + 中心）的基础上，迁移
`26_Detection_Models-main` 的 **NNdeployment 部署库**与 **Infantry 装甲板模型**（OpenVINO 后端），
用数字神经网络识别每块装甲板的类型。

## 1. 工作流程

```
图片/视频帧
   │
   ├─► [传统方案] 灯条检测 ─► 装甲板配对（四角点 + 中心）
   │
   ├─► [神经网络] ArmorModel 整帧推理 ─► 网络装甲板（角点 + 类型/颜色/尺寸）
   │
   └─► [匹配] 按中心距将网络类型写回传统装甲板 ─► 输出类型（数字）+ 可视化
```

思路参考 `26_Detection_Models-main`：即用 `NNdeployment` 的 `ArmorModel` 对整帧图像做一次
推理，再与传统方案检测出的装甲板按中心距匹配（距离 ≤ 0.5 倍装甲板宽度才认为是同一块，
一对一匹配），把类型信息（`armor_id` / `class_name` / 置信度）补充到传统装甲板上。
即：**传统方案负责定位，神经网络负责分类**。

类型 ID 与名称（与开源模型 Class ID 定义一致）：

| ID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 类型 | 哨兵 | 英雄 | 工程 | 3号步兵 | 4号步兵 | 5号步兵 | 前哨站 | 基地(底部) | 基地(顶部) |

颜色 ID：`0=蓝 1=红 2=白 3=紫`。

## 2. 环境依赖

| 依赖 | 版本要求 | 本机状态 |
| --- | --- | --- |
| Ubuntu | 22.04 | ✔ |
| GCC | ≥ 11（C++20） | ✔ 11.4 |
| CMake | ≥ 3.22 | ✔ 3.22.1 |
| OpenCV | 4.x（apt `libopencv-dev` 即可） | ✔ 4.5.4 |
| OpenVINO | C++ Runtime（2025.4+，二选一安装） | ✔ 2026.4.0（pip） |

### OpenVINO 安装方式（二选一）

**方式一：pip 安装（推荐，无需 root，本机已用此方式）**

```bash
pip install openvino
```

pip 包自带 C++ 所需的头文件、库与 CMake 配置，无需 `setupvars.sh`。
顶层 `CMakeLists.txt` 会自动探测 pip 包位置（通过 `python3 -c "import openvino ..."`），
无需手动设置任何环境变量。

**方式二：apt 安装（需要 sudo）**

```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo gpg --output /etc/apt/trusted.gpg.d/intel.gpg --dearmor GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino.list
sudo apt update
sudo apt install -y openvino-2025.4.1
```

apt 安装的 OpenVINO 自带 CMake 配置，会被 `find_package(OpenVINO)` 直接找到，
顶层 CMakeLists 的自动探测逻辑不受影响。

> 注意：本工程使用的虚拟机通常没有 Intel GPU，模型配置中的 `device` 必须为 `CPU`
> （已配置好）；只有物理机具备 Intel 核显/独显时才可改为 `GPU`。

## 3. 构建

```bash
cd /home/jay/Desktop/auto_aim_homework/tradition
cmake -S . -B build
cmake --build build -j$(nproc)
```

或在 VS Code 中：`Ctrl+Shift+B` → `CMake: 配置并构建 main`。

## 4. 运行

```bash
./build/main                                      # 默认视频 video/video_combat.avi + armor_v8
./build/main img/test_1.jpg                       # 指定图片，使用 armor_v8
./build/main video/video_combat.avi               # 指定 AVI 视频
./build/main img/test_1.jpg armor_v5              # 指定图片和模型（armor_v5 / armor_v8）
```

- 程序先尝试将输入按图片读取；读取失败时自动尝试作为视频打开，不使用视频帧数猜测文件类型，因此兼容 PNG/RGBA 图片。
- 图片输入时，终端打印灯条中心、尺寸、装甲板角点、类型（中文）和置信度；结果图保存到 `lightbar_result.jpg`。
- 视频输入支持 AVI（OpenCV 后端支持的其他视频格式也可尝试）；模型只初始化一次，然后逐帧检测。终端每帧打印检测摘要，视频结束后打印处理帧数、耗时、平均帧耗时和 FPS。
- 带图形显示环境时按视频帧率节流播放，按 `q` 或 `Esc` 提前结束；若单帧识别耗时超过帧间隔，则播放速度受实际处理速度限制。无 `DISPLAY` 时跳过窗口并尽快处理完整视频。视频处理结果以 MJPG 编码保存到 `lightbar_result.avi`。
- 结果图/视频中的绿框为传统方案装甲板四边形，
  **黄色标签为神经网络识别出的类型**（未识别时标签为绿色 `?`）。

## 5. 新增/修改的文件

| 文件 | 说明 |
| --- | --- |
| `NNdeployment/` | 部署库（复制自 26_Detection_Models-main，与上游完全一致，勿修改） |
| `models/openvino/` | Infantry-v5n / v8n 模型（复制自 26_Detection_Models-main） |
| `configs/detect.json` | 模型配置：xml 路径、推理模式、设备、置信度阈值 |
| `include/digit_recognizer.hpp` | 数字识别模块接口（DigitParams / DigitRecognizer） |
| `src/digit_recognizer.cpp` | 整帧推理 + 中心距匹配实现 |
| `include/armor.hpp` | `Armor` 增加类型字段（has_type / armor_id / color_id / nn_score / class_name） |
| `src/armor.cpp` | 新增 `armorCenter()` 与 `armorTypeNameEn()`（putText 不能画中文） |
| `src/visualize.cpp` | 装甲板标签显示类型与置信度 |
| `src/main.cpp` | 图片/视频智能切换；逐帧执行检测、配对、数字识别、打印与可视化 |
| `video/video_combat.avi` | AVI 测试视频；默认视频输入 |
| `lightbar_result.avi` | 视频检测结果（运行视频输入后生成） |
| `CMakeLists.txt` | 集成 NNdeployment 子目录 + OpenVINO 自动探测 |
| `.vscode/tasks.json` | CMake 构建及运行任务 |

## 6. 常见参数调整

- **匹配阈值**（`src/main.cpp` 中 `DigitParams` 的 `0.5f`）：
  传统装甲板中心与网络装甲板中心的距离上限（单位：装甲板宽度）。
  调大更容易匹配（可能误配），调小更严格（可能漏配）。
- **置信度阈值**（`configs/detect.json` 中 `score_threshold`）：默认 `0.5`。
- **推理模式**（`configs/detect.json` 中 `infer_mode`）：
  单图离线处理用 `sync` 最简单；连续帧场景可改 `async`（流水线推理，需注意结果延迟）。
  > 若终端出现 `注：标准情况下v8步兵应该使用openvino部署，async推理`，这是部署库对 `sync`
  > 模式的常规提示，离线单图处理用 `sync` 是无误的（同步返回结果，不会丢帧）。
- **换模型**：把新模型的 `.xml` / `.bin` 放入 `models/openvino/`，
  并修改 `configs/detect.json` 中对应节点的 `xml` 相对路径即可，无需重新编译。

## 7. 实测示例

```
$ ./build/main img/test_1.jpg
Total armors: 1
Armor 1: Blue  left_center=(489, 533)  right_center=(827, 546)  dist=339.2  score=0.357
  Corners: LT=(482,472) RT=(829,479) RB=(826,612) LB=(495,594)
  Type: 4号步兵  id=4  color_id=0  score=0.945

$ ./build/main img/test_11.png
Total armors: 1
  Type: 哨兵  id=0  color_id=0  score=0.923

$ ./build/main video/video_combat.avi
输入源: 视频 ... (1440x1080, 30 FPS, 531 帧)
[帧 1/531] lightbars=... armors=...
视频处理结束：共处理 531 帧，耗时 ... 秒，平均 ... ms/帧（... FPS）。
```

## 8. 参考

- `26_Detection_Models-main`（NNdeployment 部署库、Infantry 模型与说明文档）
- `tradition/NNdeployment/README.md`（部署库接口文档）
