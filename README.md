# RaceVideoToLog Lite

从赛车游戏视频中提取速度数据，生成时间-速度-距离 CSV 文件。**CPU-only、CLI-only** 的精简版本，零 GUI 依赖，单文件 EXE 可部署。

与完整版 [RaceVideoToLog](https://github.com/chr431/RaceVideoToLog) 相比，本版本：
- 移除了 GUI 界面、数据分析、人工基准模式、GPU 加速
- 代码量减少 ~60%，EXE 体积减少 ~75%（81MB vs 333MB）
- 保留核心 OCR + 物理纠错流水线，精度一致

## 安装

```bash
pip install rapidocr_onnxruntime onnxruntime opencv-python numpy
```

启动时自动加载 PP-OCRv5 Mobile ONNX 模型（首次需下载至 `rapidocr_onnxruntime/models/`）。

## 使用方式

```bash
python lite.py video.mp4 --roi X1 Y1 X2 Y2 [options] -o output.csv
```

### 示例

```bash
# 基本用法
python lite.py race.mp4 --roi 1200 680 1350 740

# 每 4 帧采样一次，指定输出路径
python lite.py race.mp4 --roi 1200 680 1350 740 --div 4 -o lap1.csv

# 处理视频片段
python lite.py race.mp4 --roi 1200 680 1350 740 --frame-start 300 --frame-end 3000
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `video` | 路径 | 必填 | 视频文件路径 |
| `--roi X1 Y1 X2 Y2` | 4×int | 必填 | 仪表盘像素区域 |
| `--div` | int (1-10) | 2 | 帧采样分母，1=逐帧 |
| `--frame-start` | int | 0 | 起始帧号 |
| `--frame-end` | int | 末尾 | 结束帧号 |
| `-o, --output` | 路径 | video.csv | 输出 CSV 路径 |

### ROI 选取

用图片查看器测量仪表盘速度数字的像素坐标，建议比数字区域略大（上下左右各留 2-3px）。

## 纠错算法

全自动流程，无需人工干预：

```
视频帧提取 → OCR 三级后备链 → 自适应锚点筛选 → 5 阶段物理纠错 → 距离积分 → CSV
```

### OCR 三级后备链

每帧依次尝试，成功即跳过后续步骤：

```
1. 灰度化 + 缩放(24px) → RapidOCR 标准识别
2. OTSU 二值化 + 缩放(24px) → RapidOCR 标准识别          (步骤 1 失败时)
3. ocr_digital_fallback:
   a. CLAHE+OTSU × 3 高度(28/32/48px) → 标准识别
   b. 4 变体 × 2 高度(32/48px) → use_det=False         (步骤 3a 失败时)
```

### 自适应锚点筛选

局部中值滤波，窗口自适应覆盖 ≈0.3 秒数据。帧值偏离中位数 ≤4 km/h 且与邻帧偏差 ≤10 km/h 者标记为锚点，作为后续纠错硬约束。

### 5 阶段物理纠错

| 阶段 | 功能 |
|------|------|
| 1. 错误检测 | 6 种检测器并行扫描：邻帧跳变、V 字形、悬崖、锚点趋势偏离、孤立离群、局部趋势偏离 |
| 2. 重 OCR | 4 种预处理变体重新识别，含图像哈希缓存去重 |
| 3. 最优选择 | 评分 = 邻帧一致性(0.40) + 锚点插值(0.35) + 平滑度(0.25) |
| 4. 多轮迭代 | 重新检测修复，最多 3 轮，空集提前退出 |
| 5. 级联填充 | 不可恢复帧线性插值 + 加速度裁剪，while 循环收敛 |

### 候选速度值生成

纠错阶段对 OCR 原始文本进行 3 策略扩展：
1. **原值保留**：OCR 结果如在合理范围内则保留
2. **后缀扩展**：处理 OCR 丢位（如 "60" → 候选 60/160/260 km/h）
3. **字符混淆替换**：19 种常见 OCR 混淆映射（O→0, S→5, 8↔0, 3↔8 等）

## 输出格式

```csv
# RaceVideoToLog Lite
# video_hash=709674b9c34665ea, video=test.mp4
# roi=876,933,962,982, format=km/h
# max_speed=400.0, max_accel=50.0, div=2
0.00,0.00,0.00,2
0.03,0.12,15.30,0
0.07,0.58,32.10,0
0.10,1.25,48.70,1
```

| 列 | 名称 | 说明 |
|----|------|------|
| timestamp | 时间戳 | 秒，精确 0.01 |
| distance | 累计距离 | 梯形法积分，单位米 |
| speed_kmh | 速度 | km/h |
| flag | 置信度 | 0=原始OCR, 1=纠错修复, 2=锚点 |

元数据头（`#` 行）包含视频 SHA-256 指纹，用于数据溯源。

## 项目结构

```
RaceVideoToLogLite/
├── lite.py                # CLI 入口
├── ocr_engine.py           # OCR 引擎、预处理、锚点选择
├── correction.py           # 5 阶段纠错流水线
├── requirements.txt        # Python 依赖
├── RaceVideoToLogLite.spec # PyInstaller 打包配置
└── README.md
```

## 打包

```bash
pip install pyinstaller
pyinstaller RaceVideoToLogLite.spec --noconfirm
```

生成 `dist/RaceVideoToLogLite.exe`（≈81MB），可在无 Python 环境的 Windows 机器直接运行。EXE 仅包含 PP-OCRv5 Mobile 模型，Windows 原生 DShow/MSMF 视频解码，无需额外安装。

## License

MIT
