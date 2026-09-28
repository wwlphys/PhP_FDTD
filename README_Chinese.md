# fdtd — 三维 MPI 并行电磁 / 声子极化激元 FDTD 求解器

**中文** | [English](README.md)

一个用 C 语言编写、基于 MPI 并行的三维时域有限差分（FDTD）全波电磁求解器，面向**各向异性介质**、**负介电常数介质**以及**光场与极性晶体光学声子强耦合**（声子极化激元，Phonon Polariton / PhP）问题的时域模拟，可用于散射型近场光学显微镜（s-SNOM）等近场问题的建模。

求解器采用 Yee 网格蛙跳格式，配合 CPML 吸收边界、张量形式的辅助微分方程（ADE）本构关系，以及由第一性原理数据（Born 有效电荷 + 动力学矩阵）驱动的离子运动方程，实现 Maxwell 方程与晶格动力学的双向耦合。

```
   ┌───────────────────┐   E    ┌─────────────────────┐   Z*·E   ┌────────────────┐
   │   Maxwell 方程     │ ─────► │  离子运动方程        │ ───────► │ 晶格位移 r, v   │
   │  ∂D/∂t = ∇×H      │        │  a = Z*·E/m − DMM·r │          └───────┬────────┘
   │  ∂H/∂t = −∇×E     │ ◄───── │                     │                  │
   └───────────────────┘ P_ion  └─────────────────────┘ ◄────────────────┘
                        = Σ Z*·r
```

---

## 目录

- [1. 主要特性](#1-主要特性)
- [2. 目录结构](#2-目录结构)
- [3. 依赖与编译](#3-依赖与编译)
- [4. 快速开始](#4-快速开始)
- [5. 配置文件详解](#5-配置文件详解)
- [6. 单位与物理约定](#6-单位与物理约定)
- [7. 输入文件格式](#7-输入文件格式)
- [8. 输出文件格式](#8-输出文件格式)
- [9. 数值方法](#9-数值方法)
- [10. 后处理工作流](#10-后处理工作流)
- [11. 并行与性能](#11-并行与性能)
- [12. 已知限制与注意事项](#12-已知限制与注意事项)
- [13. 常见问题排查](#13-常见问题排查)
- [14. 参考与致谢](#14-参考与致谢)
- [15. 许可证](#15-许可证)

---

## 1. 主要特性

| 类别 | 说明 |
| --- | --- |
| 求解器 | 三维 Yee 网格显式蛙跳 FDTD，等距网格；每个方向可使用不同的空间步长 `delta_x/y/z` |
| 吸收边界 | **CPML**（卷积完全匹配层，Roden & Gedney 2000），支持 `σ`、`κ`、`α` 三参数多项式分布，可通过 `pml_reflection` 指定目标反射率 |
| 本构关系 | 完整的 3×3 **各向异性复介电张量**（实部 + 虚部，共 12 个独立分量），介电函数虚部按电导损耗处理（`σ`、`σ_m` 亦可显式给出） |
| 负介电常数 | **ADE（辅助微分方程）方法**，张量形式的 Drude / Drude-Lorentz 色散模型，处理介电函数实部为负的介质 |
| 晶格耦合 | 由第一性原理数据驱动：Born 有效电荷张量（`QeZV` / `QeZM`）+ 动力学矩阵（`DMM`），逐格点求解离子运动方程，与 Maxwell 方程双向耦合，可模拟 **PhP / 光学声子** |
| 坐标变换 | 材料坐标系可绕 `z` 轴（`anglez`）或绕 `x` 轴（`anglex`）旋转，也可直接对调 XZ 轴（`switchxz`），便于放置任意晶向 |
| 激励源 | 高斯脉冲、正弦波、Ricker 小波、偶极子（dipole）；频率可用 Hz（`frequency`）或波数 cm⁻¹（`wavenumber`）指定 |
| 并行 | MPI 一维域分解（沿 `z` 轴）+ halo 交换，边界数据打包后批量通信 |
| 输出 | 场切片（二进制）、单点全分量时域监测、全量监测（场 + 全部原子的位移/速度） |
| 工程性 | 内存占用统计（`[MEM]` 日志）、每步进度与剩余时间（ETA）估计 |

配套的 Python / MATLAB / C 后处理工具可完成：介质几何生成、介质分布与几何检查、场分布快照与动画、时域信号 FFT 提取近场频率切片、Drude-Lorentz 介电函数表生成。

---

## 2. 目录结构

```
.
├── README.md                   # 英文说明（GitHub 默认展示）
├── README_Chinese.md           # 中文说明（本文档）
├── LICENSE                     # MIT 许可证
├── .gitignore                  # 排除 文档/ 目录与运行产物
├── fdtd.c                      # 主求解器（C + MPI，单文件，约 4000 行）
├── frequency_analysis.c        # 近场频率切片提取：对场切片做逐点 FFT（依赖 FFTW3）
├── generate_medium5.py         # 由配置文件中的 [medium] 段生成 medium_map.bin
├── visualize_results2.py       # 场切片二进制 → 快速浏览图 + max|field| 列表
├── visualize_results2_switchxy.py  # 同上，绘图时交换 x/y 轴
├── visualize_input3.m          # (MATLAB) 检查配置文件的几何与源位置
├── visualize_medium.m          # (MATLAB) 检查 medium_map.bin 介质分布
├── v_source.m                  # (MATLAB) 激励源时域波形快速预览
├── Material_data/              # 材料数据（hBN、MoO3）
│   ├── hBN/                    # OUTCAR.txt, POSCAR, QeZV_clean.txt,
│   │                           # QeZM_clean.txt, DMM_clean.txt, README.txt
│   └── MoO3/                   # 同上（注意：MoO3 的文件名为 QeZm_clean.txt，小写 m）
├── Lorentz_dielectric/         # Drude-Lorentz 介电函数表 + 生成它们的 MATLAB 脚本
└── examples/                   # 示例算例（见 4.1 / 4.2）
    ├── example.cfg             #   最小示例：80×80×48，MoO3 方块，几分钟可跑完
    ├── Lorentz_zdipole_650cm/  #   研究级算例：Lorentz 色散模型 MoO3，z 偶极子 @650 cm⁻¹
    │   ├── simulation.cfg
    │   └── dielectric_MoO3_Lorentz.txt
    └── MoO3_zdipole_800cm/     #   研究级算例：声子耦合 MoO3（Born + 动力学矩阵）@800 cm⁻¹
        ├── simulation.cfg
        ├── OUTCAR.txt
        ├── QeZV_clean.txt
        ├── QeZM_clean.txt
        └── DMM_clean.txt
```

> 说明：本仓库不包含任何 `.docx` 理论推导文档，相关目录已被 `.gitignore` 排除。

---

## 3. 依赖与编译

### 3.1 依赖

| 组件 | 用途 | 说明 |
| --- | --- | --- |
| C 编译器 | 编译 `fdtd.c`、`frequency_analysis.c` | GCC / Clang / ICC 均可 |
| MPI | `fdtd.c` 必需 | OpenMPI、MPICH、Intel MPI 均可（代码只使用标准 MPI-2 API） |
| math / rt | 链接数学库 | `-lm -lrt` |
| FFTW3 | **仅** `frequency_analysis.c` 需要 | `libfftw3-dev` |
| Python 3 + NumPy | `generate_medium5.py` | 只用到 `numpy` 与标准库 |
| Python 3 + Matplotlib | `visualize_results2*.py` | 仅用于画图 |
| MATLAB | `*.m` 脚本 | Octave 可能可用，但未做兼容性测试 |

Debian / Ubuntu 一键安装：

```bash
sudo apt update
sudo apt install build-essential libopenmpi-dev openmpi-bin libfftw3-dev
pip3 install numpy matplotlib
```

### 3.2 编译主求解器

```bash
mpicc -o fdtd.x fdtd.c -lm -lrt -Ofast -flto -funroll-loops
```

若使用 Intel 编译器，可用 `mpiicc -o fdtd.x fdtd.c -lm -O3 -xHost -ipo`。

> **提示**：`-Ofast` 会启用 `-ffast-math`。本程序的所有常量与物理量都是有限值、不依赖 IEEE 特殊值语义，因此通常安全；若你修改代码引入了 NaN/Inf 依赖逻辑（例如用 `isnan()` 做判断），请改用 `-O3`。

### 3.3 编译频率分析工具

```bash
# 若 FFTW3 安装在标准路径
gcc -o frequency_analysis frequency_analysis.c -lfftw3 -lm

# 若 FFTW3 安装在自定义前缀（例如 /opt/fftw-3.3.6）
gcc -o frequency_analysis frequency_analysis.c \
    -I/opt/fftw-3.3.6/include -L/opt/fftw-3.3.6/lib -lfftw3 -lm
```

> **FFTW3 装在自定义前缀时，这个前缀需要在两个不同的时刻各指定一次：编译时（`-I`/`-L`）和运行时。** `frequency_analysis` 链接的是**动态库** `libfftw3.so`，编译选项并不会告诉动态加载器去哪里找它。因此「编译通过」并不等于「能运行」——若没有设置 `LD_LIBRARY_PATH`（或写入 `/etc/ld.so.conf` 后执行 `ldconfig`），程序会在启动瞬间直接报错退出：
>
> ```
> ./frequency_analysis: error while loading shared libraries: libfftw3.so.3: cannot open shared object file: No such file or directory
> ```
>
> ```bash
> # 1) 编译时：告诉编译器头文件与库的位置
> gcc -o frequency_analysis frequency_analysis.c \
>     -I/opt/fftw-3.3.6/include -L/opt/fftw-3.3.6/lib -lfftw3 -lm
>
> # 2) 运行时：告诉动态加载器同一个库的位置
> export LD_LIBRARY_PATH=/opt/fftw-3.3.6/lib:$LD_LIBRARY_PATH
> ./frequency_analysis simulation.cfg
> ```
>
> 可用 `ldd ./frequency_analysis` 查看该程序在找哪个库：若某项显示 `not found`，说明运行时路径仍未设置。把这条 `export` 写进作业脚本（或 `~/.bashrc`）即可免去每次运行都要重新设置。

### 3.4 编译期常量

以下常量在 `fdtd.c` 顶部宏定义，修改后需要**重新编译**：

| 宏 | 默认值 | 含义 |
| --- | --- | --- |
| `EPS_INF` | `20.0` | ADE 方法中的**高频介电常数 ε∞**（标量，仅用于 `use_ADE=1` 的材料） |
| `N` | `3` | 介电张量/矩阵求逆的维度，固定为 3 |
| `TRACK_MIN_PRINT` | `1e8` | 内存日志的最小打印阈值（字节） |

---

## 4. 快速开始

### 4.1 最小示例

`examples/example.cfg` 是一个几分钟就能跑完的小算例（80 × 80 × 48 网格 + MoO3 小方块；64 核机器实测：4 进程约 4.5 分钟、16 进程约 1.5 分钟），用于验证编译、数据文件与后处理链路。在仓库根目录下执行：

```bash
# 1) 编译
mpicc -o fdtd.x fdtd.c -lm -lrt -Ofast

# 2) 由配置文件的 [medium] 段生成介质分布文件 medium_map.bin
python3 generate_medium5.py examples/example.cfg medium_map.bin

# 3) 运行 FDTD（数字为 MPI 进程数，建议 ≤ nz）
mpirun -np 4 ./fdtd.x examples/example.cfg > out.log 2>&1

# 4) 快速查看结果
python3 visualize_results2.py          # 场切片快照

# 5) 提取近场频率切片（需要先删除源仍在振荡期间的场切片）
#    （FFTW3 在自定义前缀时：此处补上 -I/-L，并在运行时设置 LD_LIBRARY_PATH，见 3.3）
gcc -o frequency_analysis frequency_analysis.c -lfftw3 -lm
./frequency_analysis examples/example.cfg
```

> **以上五步请在仓库根目录执行，或者把数据文件与配置文件一起复制过去。**
> `examples/example.cfg` 中引用材料数据用的是**相对路径**（`Material_data/MoO3/...`），而 `medium_file = ./medium_map.bin` 是相对于**当前工作目录**（不是配置文件所在目录）解析的。因此在别处执行上面的命令会因为找不到 `medium_map.bin` 或 `QeZV`/`QeZM`/`DMM` 而报错（`Error opening medium file` 等）。
>
> 又因为所有输出（`field_slice_*.bin`、`monitor_point.dat`、`full_monitor.dat`、`source.txt`）同样写在当前工作目录，通常更希望在一个独立目录里运行——此时需要把配置文件**和 `Material_data/` 目录一起**复制过去：
>
> ```bash
> mkdir -p run && cp examples/example.cfg run/ && cp -r Material_data run/
> cd run
> python3 ../generate_medium5.py example.cfg          # 生成 ./medium_map.bin
> mpirun -np 4 ../fdtd.x example.cfg > out.log 2>&1
> python3 ../visualize_results2.py                    # 扫描当前目录
> ../frequency_analysis example.cfg                   # 同样扫描当前目录
> ```
>
> （4.2 节的两个研究级算例本身已是自包含目录，需在各自目录内运行；把任何算例复制到别处时同理。）

运行过程中，标准输出会给出形如：

```
it 1200/20000 (6.0%), t = 4.57383e-14 s,|E|_max = 3.42e+05 V/m,r_max = 1.23e-03 A, elapsed: 000:00:37, ETA: 000:09:43
```

的进度信息（`r_max` 为**全部原子位移模的最大值**，单位 Å），标准错误会给出内存占用。单个数组只有在达到 0.1 GB 时才会逐条打印（`TRACK_MIN_PRINT`，见 [3.4](#34-编译期常量)），因此像本算例这样的小算例只会打印启动时的汇总行：

```
[MEM] Before time loop: per-rank = 0.017825 GB, total (4 ranks) = 0.061462 GB
```

研究级规模的计算还会逐条打印超过阈值的数组，例如：

```
[MEM] +1.234567 GB (Ex), total=1.234567 GB (est. all: 4.938268 GB)
```

### 4.2 研究级算例：偶极子激发下的 MoO3 表面近场

`examples/` 下另有两个规模接近实际研究的算例。两者**几何与源的位置完全相同**，只有"材料 4（MoO3 层）如何描述"这一点不同，因此可以成对使用，对比两种建模途径：

| 算例 | 材料 4 的建模方式 | 源频率 | 说明 |
| --- | --- | --- | --- |
| `MoO3_zdipole_800cm/` | **声子耦合**：`natom=16` + `born`/`QeZV`/`QeZM`/`DMM`，逐格点求解离子运动方程 | 800 cm⁻¹ | 显式包含光学声子自由度（PhP） |
| `Lorentz_zdipole_650cm/` | **等效色散介质**：`dielectric=dielectric_MoO3_Lorentz.txt` + `use_ADE=1` | 650 cm⁻¹ | 用 Drude-Lorentz 介电函数表替代声子自由度 |

两者的共同设置：

| 项 | 值 | 说明 |
| --- | --- | --- |
| 网格 | `400 × 800 × 800`，`Δx = 5 nm`、`Δy = Δz = 10 nm` | 域尺寸 2 µm × 8 µm × 8 µm；x 为表面法向，故在法向加密 |
| 时间 | `tsteps = 157000`，`dt = 9.5263e-18 s` | 总时长约 1.5 ps；`dt` 取在三维 CFL 极限附近（由 Δx = 5 nm 决定） |
| 边界 | `pml_thickness = 150`，`pml_reflection = 1e-6` | x 方向 150 × 5 nm = 750 nm，y/z 方向 1.5 µm |
| 几何（`cubes`） | `x ∈ [0,160]` → 材料 1（内置 SiO2 衬底）；`x ∈ [161,200]` → 材料 4（MoO3，厚 40 × 5 nm = 200 nm）；`x > 200` 为真空 | 材料一直延伸到 `x = 0`（穿过 PML），避免衬底界面产生额外反射 |
| 源 | `(220, 400, 400)`，z 偏振偶极子 → MoO3 表面外 100 nm 处 | 点偶极子近场激发 |
| 输出 | `output_component = Ex`，`output_slice_axis = x`，`output_slice_position = 201` | 即表面外 1 格（5 nm）处的 y-z 近场平面 |
| 监测点 | `(200, 400, 400)` | 最后一层 MoO3，`dir = x` |

两个算例目录都是**自包含**的：`MoO3_zdipole_800cm/` 中的 `QeZV_clean.txt`、`QeZM_clean.txt`、`DMM_clean.txt` 与 `Material_data/MoO3/` 下的同名文件完全一致，`OUTCAR.txt` 内容相同（换行符为 LF，便于版本管理）；`Lorentz_zdipole_650cm/` 中的 `dielectric_MoO3_Lorentz.txt` 与 `Lorentz_dielectric/` 下的同名文件一致。这样每个算例单独拷走也能运行。

两个算例的配置文件都使用**相对路径**（`medium_file = ./medium_map.bin`、材料数据用裸文件名），因此必须在**各自的目录内**运行：

```bash
cd examples/MoO3_zdipole_800cm            # 或 Lorentz_zdipole_650cm

# 生成介质分布（约 1.02 GB，不随仓库提供）
python3 ../../generate_medium5.py simulation.cfg

# 运行（进程数按可用内存调整，见下方"资源需求"）
mpirun -np 28 ../../fdtd.x simulation.cfg > out.log 2>&1

# 后处理（frequency_analysis 只需编译一次）
python3 ../../visualize_results2.py
gcc -o frequency_analysis ../../frequency_analysis.c -lfftw3 -lm
./frequency_analysis simulation.cfg
```

> **资源需求**（粗略估计，用于选择进程数与磁盘配额）
>
> | 项目 | `MoO3_zdipole_800cm` | `Lorentz_zdipole_650cm` |
> | --- | --- | --- |
> | 场数组（9 个分量） | \~18.4 GB | \~18.4 GB |
> | `medium` | \~1.0 GB | \~1.0 GB |
> | `r`/`v`（仅 MoO3 的 10% 格点） | \~19.7 GB | — |
> | `P_ADE`/`P_ADEo` | — | \~1.2 GB |
> | CPML 辅助场 `ψ` | \~20.5 GB | \~20.5 GB |
> | **内存合计** | **\~60 GB** | **\~41 GB** |
> | `medium_map.bin` | 1.02 GB | 1.02 GB |
> | `field_slice_*.bin` | \~4.0 GB（1571 个切片） | \~4.0 GB |
>
> 内存按 z 方向分解到各进程：28 进程时每进程约 1.8 GB，56 进程时约 0.9 GB；其中 z 区间与 z 方向 PML 重叠的进程还要额外分配约 0.3 GB 的 z 法向 `ψ` 数组。
>
> **输出说明**：`MoO3_zdipole_800cm` 的监测点落在 `natom > 0` 的材料上，程序会**自动开启** `full_monitor`，额外产生 `full_monitor.dat`（1571 行 × 106 列，含全部 16 个原子的位移与速度）；`Lorentz_zdipole_650cm` 的材料 4 不含原子，`monitor_point` 里的 `atom=`/`dir=` 不起作用，输出只有 10 个场分量列。
>
> **采样与频率分辨率**：`output_interval = 100` 对应切片间隔 9.5263e-16 s、共 1571 个切片，据此做 FFT 的频率分辨率为 **22.3 cm⁻¹**（Nyquist 约 17500 cm⁻¹）。需要更细的频率网格时，请减小 `output_interval` 或增加 `tsteps`；做近场频谱前记得删掉源仍在振荡期间（`t < t0 + 6·tau` 附近）的切片。

---

## 5. 配置文件详解

配置文件是**纯文本**，按行解析，规则如下：

- `#` 开头的行是注释；空行忽略。**行内注释（行尾 `#`）不支持**，会被当成数值的一部分。
- `[grid]`、`[medium]` 之类的段头对求解器而言只是**装饰**（解析时自动跳过），但 `generate_medium5.py` **依赖 `[medium]` 段头**来定位介质几何定义，因此请保留它们。
- `source ...` 行与 `material ...` 行必须**从第 0 列开始**（解析器用 `strstr(line, "source") == line` 判断），前面不能有空格。
- 普通参数行格式为 `key = value`（等号两侧空格可有可无，值可含空格）。
- `source` / `material` 行内以**逗号**分隔多个 `key=value`；`value` 中**不能含空格和逗号**（用 `%s` 解析），因此文件路径不要带空格。

### 5.1 网格与运行参数

| 键 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `nx` | int | 50 | x 方向网格数（全局） |
| `ny` | int | 50 | y 方向网格数（全局） |
| `nz` | int | 50 | z 方向网格数（全局，MPI 沿此方向分解） |
| `tsteps` | int | 500 | 时间步数 |
| `delta_x` | double | 0.01 | x 方向空间步长（**米**） |
| `delta_y` | double | 0.01 | y 方向空间步长（米） |
| `delta_z` | double | 0.01 | z 方向空间步长（米） |
| `dt` | double | 自动 | 时间步长（**秒**）。若 `dt ≤ 0` 或超过 CFL 极限，则自动取 `dt = 0.99 · min(Δx,Δy,Δz) / (c√3)` |
| `pml_thickness` | int | 10 | CPML 层厚度（格点数），六个面都使用同一厚度 |
| `pml_reflection` | double | 0.001 | CPML 目标反射率（用于计算 `σ_max`），越小吸收越强 |
| `output_interval` | int | 20 | 场切片与 `full_monitor` 的输出间隔（步）；`monitor_point` **每步都输出** |
| `output_component` | string | `Ex` | 输出分量，可取 `Ex`/`Ey`/`Ez`/`E`（模值）/`Dx`/`Dy`/`Dz` |
| `output_slice_axis` | char | `z` | 切片法向轴，`x`/`y`/`z` |
| `output_slice_position` | int | `nz/2` | 切片位置（格点索引） |
| `medium_file` | string | `medium_map.bin` | 介质分布二进制文件路径 |
| `monitor_point` | list | 关闭 | 监测点，见 [5.4](#54-监测点) |
| `monitor_output_file` | string | `monitor_point.dat` | 单点监测输出文件名 |
| `full_monitor` | int | 0 | 置 1 输出"场 + 全部原子位移/速度"到文件 |
| `full_monitor_output_file` | string | `full_monitor.dat` | 全量监测输出文件名 |

> **未实现**：配置文件示例中出现的 `bc_x` / `bc_y` / `bc_z`（周期性边界）**没有被解析**，程序会静默忽略它们。当前版本只支持 CPML 吸收边界。

**关于 `dt` 的重要提醒**：程序会把过大的 `dt` 静默替换为 CFL 极限值，但**不会**自动调整 `tsteps`。因此物理上想要模拟的总时长 `tsteps × dt` 会随之改变；如果你显式指定了 `dt`，请同时核对 `tsteps`。

> 举例：`delta_* = 10e-9 m` 时 CFL 极限为 `dt ≈ 1.9e-17 s`。若配置写 `dt = 1e-15`（大 50 倍），程序会静默改用 `1.9e-17 s`，此时 `tsteps = 50000` 只对应约 0.96 ps 而不是 50 ps。请以程序启动时打印的 `Time steps: ..., dt= ...` 为准。

### 5.2 激励源 `source`

```
source type=<...>, x=<int>, y=<int>, z=<int>, amplitude=<double>, component=<x|y|z>, \
       frequency=<Hz> | wavenumber=<cm^-1>, t0=<s>, tau=<s>, start_time=<s>, end_time=<s>
```

| 键 | 默认值 | 说明 |
| --- | --- | --- |
| `type` | `gaussian` | `gaussian`、`sinusoidal`、`ricker`、`dipole` |
| `x` / `y` / `z` | — | 源所在格点索引，请显式给出 |
| `amplitude` | 1.0 | 幅度 |
| `component` | `z` | 注入分量：`x`/`y`/`z` → 分别累加到 `Dx`/`Dy`/`Dz` |
| `frequency` | 1e9 | 频率（Hz），用于 `sinusoidal`/`ricker`/`dipole` |
| `wavenumber` | — | 波数（cm⁻¹），换算 `frequency = wavenumber × 0.03e12` Hz |
| `t0` | — | 脉冲中心时刻（s），请显式给出 |
| `tau` | — | 脉冲宽度（s），请显式给出 |
| `start_time` | 0 | 源开始工作的时刻（s） |
| `end_time` | `tsteps × dt` | 源停止工作的时刻（s） |

时域波形（`A` = `amplitude`，`f` = `frequency`）：

| 类型 | 表达式 |
| --- | --- |
| `gaussian` | `A · exp(−((t − t0)/tau)²)` |
| `sinusoidal` | `A · sin(2πft)` |
| `ricker` | `A · (1 − 2π²f²(t−t0)²) · exp(−π²f²(t−t0)²)` |
| `dipole` | `A · sin(2πft) · exp(−((t − t0)/tau)²)` |

`dipole` 类型**必须且只能**给出 `frequency` 或 `wavenumber` 之一，否则程序报错退出。

> **注意 1**：源是"软源"——直接把电流项累加到 `D` 上（`D += value · dt`），不会覆盖场值。
>
> **注意 2**：建议**逐项写全 `x`、`y`、`z`、`t0`、`tau`**。源行是在第一遍扫描中解析的，此时 `nx`、`dt` 等键尚未读入，因此这几项的默认值基于编译期初值而非你的配置。
>
> **注意 3**：**第一个** `source` 行的 `frequency` 会被记录为 `unique_source_frequency`，用于：
> 1. 从 `dielectric=` 指定介电函数表中挑选**最接近该频率**的一行；
> 2. 计算 `use_ADE=1` 材料的 ADE 系数（`γ`、`ω_p²`）。
>
> 因此请把**主频对应的源放在最前面**。多个源频率不同时，ADE/介电函数只对第一个源的频率精确成立。
>
> **注意 4**：源位置由 `x=`、`y=`、`z=` 三个独立键给出（`position=(10,10,10)` 这种写法不属于本程序的语法）。

### 5.3 材料 `material`

```
material id=<int>, name=<string>, eps_xx=..., ..., eps_zzi=..., sigma=..., sigma_m=..., \
         dielectric=<file>, use_ADE=0|1, anglez=<rad>, anglex=<rad>, switchxz=0|1, \
         natom=<int>, born=<file>|[数组], QeZV=<file>, QeZM=<file>, DMM=<file>
```

| 键 | 说明 |
| --- | --- |
| `id` | **必需**。材料编号；`0`–`3` 为程序内置材料，自定义材料**必须从 `4` 开始** |
| `name` | 材料名（最长 63 字符），仅用于打印 |
| `eps_xx` … `eps_zz` | 介电张量实部（9 个分量，**不要求**你输入对称项，代码会按对称性使用） |
| `eps_xxi` … `eps_zzi` | 介电张量虚部（9 个分量，约定 **≥ 0**），作为损耗处理 |
| `sigma` | 电导率 σ（S/m），默认 0 |
| `sigma_m` | 磁损耗 σ_m（Ω/m），默认 0 |
| `dielectric` | 介电函数表文件名。设置后会在 `unique_source_frequency` 处插值取一行**覆盖** `eps_*` 与 `eps_*i`（见 [7.4](#74-介电函数表-dielectric)） |
| `use_ADE` | `1` 启用 ADE 方法（介电函数实部为负时必须为 1），`0` 使用常规 `E = ε⁻¹D/ε₀` |
| `anglez` | 材料绕 **z 轴**旋转的角度，单位 **弧度**（顺时针/逆时针见 [9.4](#94-各向异性本构关系与坐标旋转)） |
| `anglex` | 材料绕 **x 轴**旋转的角度，单位弧度。`anglez` 优先于 `anglex`（二者同时非零时只用 `anglez`） |
| `switchxz` | `1` 时对调材料的 XZ 轴（同时作用于 `eps`、`born`、`QeZV`、`QeZM`、`DMM`） |
| `natom` | 原胞中的原子数。`> 0` 时程序进入"离子耦合"模式，**必须**同时给出 `QeZV`、`QeZM`、`DMM`，否则 `MPI_Abort` |
| `born` | 若值中不含 `[`，视为 **VASP OUTCAR 文件名**（读取 Born 有效电荷与宏观静态介电张量）；若形如 `[v1,v2,...]`，则直接给出 `3·natom·3` 个数 |
| `QeZV` | 文件，`9·natom` 个数（见 [7.2](#72-声子极化激元数据-qezv--qezm--dmm)） |
| `QeZM` | 文件，`9·natom` 个数 |
| `DMM` | 文件，`9·natom²` 个数 |

**内置材料（`id = 0…3`）**

| id | 名称 | 说明 |
| --- | --- | --- |
| 0 | Vacuum | ε = 1 |
| 1 | SiO2 | Drude-Lorentz 模型（在 `unique_source_frequency` 处取值），若该频率下 ε 实部 ≤ 0 会自动打开 `use_ADE` |
| 2 | Uniaxial Crystal | ε = diag(8, 7, 5) |
| 3 | Lossy Medium | ε = 9，σ = 0.1 S/m |

程序启动时会打印全部材料的完整信息（介电张量、σ、use_ADE、旋转角、原子数），可据此核对配置是否按预期解析。

> **关于 `born=<OUTCAR>` 的行为**：设置 `born=` 会让程序从 OUTCAR 中读取 `MACROSCOPIC STATIC DIELECTRIC TENSOR` 并**覆盖**该材料的 `eps_*`（虚部置 0）；读到的 Born 电荷数组本身**目前不参与任何场/离子更新**（离子耦合完全由 `QeZV`/`QeZM`/`DMM` 承担）。因此对于声子极化激元计算，请注意 OUTCAR 给出的是**静态**介电张量 ε(0)，而耦合式 `E = (D − P_ion)/(ε₀ε∞)` 中应当使用**高频（clamped-ion）介电张量 ε∞**；直接用静态值会重复计入晶格贡献。建议对 PhP 材料改用 `dielectric=` 或显式 `eps_xx=...` 给出 ε∞。

### 5.4 监测点

```
monitor_point = x=500, y=500, z=250, atom=0, dir=z
monitor_output_file = monitor_point.dat
```

- `x`、`y`、`z`：监测点格点索引（全局）。
- `atom`：原子序号（从 0 开始）；`dir`：位移方向，`x`/`y`/`z`。
- 若该点材料的 `natom > 0` 且 `atom < natom`，输出该原子的位移 `R` 与速度 `V`；否则只输出 9 个场分量。
- **不设 `monitor_point` 行**时该功能关闭（`monitor_enabled` 默认为 0）。
- 当监测点落在 `natom > 0` 的材料上时，程序会**自动打开** `full_monitor`。

### 5.5 完整示例

```ini
# ===== 网格与时间 =====
[grid]
nx = 400
ny = 400
nz = 200
delta_x = 10e-9
delta_y = 10e-9
delta_z = 10e-9
tsteps = 20000
# dt 省略则自动取 CFL 极限
pml_thickness = 20
pml_reflection = 1e-3
output_interval = 50
output_component = Ez
output_slice_axis = x
output_slice_position = 200

# ===== 激励源（第一个源的频率决定 ADE / 介电函数取值）=====
source type=dipole, x=200,y=200,z=100, amplitude=1.0, wavenumber=930, component=x, t0=0.2e-12, tau=0.05e-12

# ===== 介质几何（仅 generate_medium5.py 使用）=====
[medium]
cubes = [ {"xrange": (0, 399), "yrange": (0, 399), "zrange": (60, 140), "material": 4} ]

# ===== 材料 =====
material id=4, name=MoO3, natom=16, born=Material_data/MoO3/OUTCAR.txt, \
  QeZV=Material_data/MoO3/QeZV_clean.txt, QeZM=Material_data/MoO3/QeZm_clean.txt, \
  DMM=Material_data/MoO3/DMM_clean.txt
# 上面三行的反斜杠续行是无效的！material 必须写在一行内

material id=5, name=Lorentz_MoO3, dielectric=Lorentz_dielectric/dielectric_MoO3_Lorentz.txt, \
  use_ADE=1, anglex=0.523598, switchxz=1

medium_file = ./medium_map.bin
monitor_point = x=200,y=200,z=100,atom=0,dir=z
monitor_output_file = monitor_point.dat
full_monitor = 1
full_monitor_output_file = full_monitor.dat
```

> ⚠️ 上面的 `\` 续行**只是本文档的排版**。程序**不支持跨行的 `source` / `material` 定义**，每条 `material`、`source` 必须写在**单独一行**。请参考 `examples/example.cfg` 中的写法。

### 5.6 `[medium]` 几何定义（供 `generate_medium5.py` 使用）

`generate_medium5.py` 用正则 + `ast` 解析 `[medium]` 段，支持三种几何体（可同时使用，按 `cubes → hexahedra → cylinders` 顺序**依次覆盖**）：

```ini
[medium]
# 立方体（格点索引范围，闭区间）
cubes = [ {"xrange": (0, 999), "yrange": (0, 999), "zrange": (187, 250), "material": 4} ]

# 任意平行六面体（顶点按"先 z 小后 z 大，各在 xy 平面内逆时针"排列）
hexahedra = [
    { "vertices": [ (201,406,593), (204,406,593), (204,408,593), (201,408,593),
                    (201,406,595), (204,406,595), (204,408,595), (201,408,595) ],
      "material": 6 }
]

# 圆柱体：p1/p2 为两端点（格点索引），radius 单位是米，会按 delta_x/y/z 换算
cylinders = [ {"p1": (212, 500, 506), "p2": (212, 700, 494), "radius": 120e-9, "material": 6} ]
```

> 配置文件示例中可能出现的 `spheres`、`slabs`、`points` 关键字**当前版本未实现**，写了不会生效。
>
> `hexahedra` 与 `cylinders` 的 `[ ]` 中间**不要写注释、不要换行**（`cubes` 支持行内注释）。几何体之外（未被任何几何体覆盖）的格点自动为材料 0（真空）。

---

## 6. 单位与物理约定

整个程序使用 **SI 单位制**。

| 物理量 | 单位 | 备注 |
| --- | --- | --- |
| 空间步长 `delta_*`、`radius` | m | 格点索引 × 步长 = 物理坐标 |
| 时间 `dt`、`t0`、`tau` | s | |
| 频率 `frequency` | Hz | |
| 波数 `wavenumber` | cm⁻¹ | 换算：`f [Hz] = wavenumber × 0.03e12`（即取 c = 3×10¹⁰ cm/s；与真空真值 2.998×10¹⁰ cm/s 相差约 0.07%） |
| 电场 `E` | V/m | |
| 磁场 `H` | A/m | |
| 电位移 `D` | C/m² | |
| 电导率 `sigma` | S/m | |
| 磁损耗 `sigma_m` | Ω/m | |
| 原子位移 `r` | m | 打印时换算为 Å（`r/1e-10`） |
| 原子速度 `v` | m/s | |
| 介电张量 `eps_*` | 无量纲（相对介电常数） | 虚部约定 **≥ 0** |

---

## 7. 输入文件格式

### 7.1 介质分布 `medium_map.bin`

由 `generate_medium5.py` 生成的**二进制**文件（本机字节序，`int32`）：

```
偏移 0        : int32 nx
偏移 4        : int32 ny
偏移 8        : int32 nz
偏移 12 起    : int32 medium[nx][ny][nz]，x 最慢、y 次之、z 最快
                即第 (i,j,k) 个值位于  12 + 4·((i·ny + j)·nz + k)
```

- 文件头中的 `nx/ny/nz` 必须与配置文件中的网格尺寸**完全一致**，否则程序报错退出。
- 数值为**材料 id**；`0` 表示真空。
- 文件大小应为 `12 + 4·nx·ny·nz` 字节。
- 程序读取时会按 `z` 方向分解后分发给各 MPI 进程（等价于 `MPI_Scatterv`），只有 0 号进程真正读盘。

### 7.2 声子极化激元数据 `QeZV` / `QeZM` / `DMM`

三个文件都是**纯文本、每行一个数**（`fscanf("%lf")` 顺序读取，空白符分隔即可）。设 `natom = N`：

| 文件 | 元素个数 | 内存索引 | 含义与量纲 |
| --- | --- | --- | --- |
| `QeZV_*.txt` | `9N` | `[iatom][iE][idir]`，原子优先 | 使极化强度 `P_iE = Σ_{idir} QeZV·r_idir`（C/m²），量纲 **C/m³**；物理上 ≈ `Z*e/Ω_cell` |
| `QeZM_*.txt` | `9N` | `[iatom][iE][idir]`，原子优先 | 使加速度 `a_idir = Σ_{iE} QeZM·E_iE`（m/s²），量纲 **C/kg**；物理上 ≈ `Z*e/m_atom` |
| `DMM_*.txt` | `9N²` | `[iatom][idir][jatom][jdir]`，`iatom` 最慢、`jdir` 最快 | 质量加权的动力学矩阵，量纲 **s⁻²**（本征值为 ω²） |

即文件顺序为：

```
QeZV/QeZM : atom0 的 9 个数, atom1 的 9 个数, ...
            每个原子内部按 iE = x,y,z 分组，组内按 idir = x,y,z
DMM       : iatom 从 0 到 N-1
              对每个 iatom，idir = x,y,z
                对每个 idir，jatom 从 0 到 N-1
                  对每个 jatom，jdir = x,y,z
```

**自洽性要求**：同一材料的三份数据必须来自同一次第一性原理计算、使用同一原胞体积与原子质量约定，且满足

```
QeZV / QeZM = m_atom / Ω_cell
```

即"某个原子的质量除以其所属原胞体积"。用仓库中的数据可以自检：

| 材料 | 第一原子 | `QeZV/QeZM` | 该原子质量 / POSCAR 原胞体积 |
| --- | --- | --- | --- |
| hBN | B | 1.25086e10 / 2.41267e7 = 518.45 | 10.811 amu / 34.626 Å³ = 1.7956e-26 kg / 3.4626e-29 m³ = 518.5 |
| MoO3 | Mo | 794.60 | 95.96 amu / 200.54 Å³ = 1.5934e-25 kg / 2.0054e-28 m³ = 794.6 |

两者一致即说明 `QeZV`、`QeZM` 与 `POSCAR` 的原胞体积、原子质量是同一个约定。

> 注意 `DMM` 采用的是**运动方程形式**（`D = Φ(i,j)/m_i`，来自 Γ 点动力学矩阵的质量加权），因此它**不是对称矩阵**，不要直接当作对称矩阵做本征分解；其本征值为 `ω²`（rad²/s²），低频端应出现 3 个接近 0 的声学模，可用来验证数据完整性。

> 各材料数据文件的坐标约定不同（例如 hBN：`z` 轴垂直于 hBN 平面；MoO3：`x=[100]`、`y=[001]`、`z=[010]`），建模时请用 `switchxz` / `anglez` / `anglex` 把晶轴旋转到仿真坐标系。
>
> 文件名大小写**不统一**：hBN 目录是 `QeZM_clean.txt`，MoO3 目录是 `QeZm_clean.txt`。在 Linux 等区分大小写的系统上请照抄实际文件名。

**数据来源**：这些文件由第一性原理计算得到——Born 有效电荷张量来自 VASP 静态计算（`OUTCAR` 中的 `BORN EFFECTIVE CHARGES` 段），`DMM` 为 Γ 点动力学矩阵经质量加权后的结果。仓库中 `Material_data/` 下的 hBN、MoO3 数据可直接使用；**自行生成数据所需的完整流程与脚本不在本仓库内**，如需扩展到其他材料，请按上面给出的物理定义与量纲约定自行准备（或联系作者）。

> 坐标约定需与 `POSCAR` 保持一致：`QeZV`/`QeZM`/`DMM` 的坐标轴顺序与 `POSCAR` 的晶格取向相对应，若数据文件做过 `switchxz` 而 `POSCAR` 没有对应交换，两者描述的就不是同一个坐标系。


### 7.3 Born 有效电荷 `OUTCAR.txt`

VASP 的输出片段，解析器按两个关键字定位：

```
 MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects)
 ------------------------------------------------------
           4.923949     0.000000     0.000000
          -0.000000     4.923947     0.000000
           0.000000    -0.000000     2.890847
 ------------------------------------------------------

 BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cummulative output)
 ---------------------------------------------------------------------------------
 ion    1
    1     2.70365     0.00000     0.00000
    2     0.00000     2.70365     0.00000
    3     0.00000     0.00000     0.82073
 ion    2
    ...
```

- 介电张量节：连续的 3 行 × 3 个数（可带行号）。
- Born 节：`ion <n>` 开始一个原子（`n` 从 1 开始），随后 3 行、每行 `方向号 + 3 个数`。
- 该文件只会在 `born=` 键给出时读取，且要求 `natom` 与之匹配（`ion` 数 ≥ `natom`）。

### 7.4 介电函数表 `dielectric=`

纯文本，每行 **15 列**（`#` 开头为注释行）：

| 列 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 含义 | f (Hz) | λ | 波数 (cm⁻¹) | ε_xx | ε_xy | ε_xz | ε_yy | ε_yz | ε_zz | ε_xxi | ε_xyi | ε_xzi | ε_yyi | ε_yzi | ε_zzi |

- 程序扫描全表，选取 **f 最接近 `unique_source_frequency`（第一个源的频率）** 的一行；若相对偏差 > 10% 会给出警告但仍使用该行。
- 只读取 6 个独立分量（`xx, xy, xz, yy, yz, zz` 的实部与虚部），并按其对称张量填充（`yx=xy`、`zx=xz`、`zy=yz`）。
- 第 2 列（波长）程序不使用，可以填 0。
- `Lorentz_dielectric/` 目录中的文件由同目录的 MATLAB 脚本生成，频率范围 1–2000 cm⁻¹、步长 0.3 cm⁻¹：

```matlab
% Lorentz_MoO3.m
epsfx=4.0; epsfy=5.2; epsfz=2.4;
wLOx=972; wLOy=851; wLOz=1004;      % cm^-1
wTOx=820; wTOy=545; wTOz=958;       % cm^-1
gx=4; gy=4; gz=2;
% ε(ω) = ε∞·(1 + (ω_LO² − ω_TO²)/(ω_TO² − ω² − iγω))
```
只要改脚本顶部的参数即可生成新材料（各向异性 Drude-Lorentz）的介电函数表。

### 7.5 结构文件 `POSCAR`

`Material_data/hBN/` 与 `Material_data/MoO3/` 下各附一份 VASP 格式的 `POSCAR`（第 1 行注释、第 2 行缩放因子、3–5 行晶格矢量、第 6 行元素符号、第 7 行原子数、第 8 行起为分数坐标），用于核对原胞体积、原子种类与原子顺序——`QeZV`/`QeZM`/`DMM` 的坐标轴约定与原子编号都以它为准（见 [7.2](#72-声子极化激元数据-qezv--qezm--dmm)）。

---

## 8. 输出文件格式

| 文件 | 生成条件 | 频率 | 格式 |
| --- | --- | --- | --- |
| `field_slice_<轴><位置>_<步数:08d>.bin` | 总是 | 每 `output_interval` 步 | 二进制：两个 `int32` 尺寸 + `float32` 数据，布局随切片轴变化，见下 |
| `monitor_point.dat` | 设置 `monitor_point` 时 | **每一步** | 文本，`#` 表头 + 每行 `Time(s) Ex Ey Ez Dx Dy Dz Hx Hy Hz [R V]` |
| `full_monitor.dat` | `full_monitor=1`（或监测点在含原子材料上自动开启） | 每 `output_interval` 步 | 文本，`#` 表头 + 每行 `Time(s)` + 9 个场分量 + 每个原子的 `r{x,y,z}` 和 `v{x,y,z}` |
| `source.txt` | 总是 | 每 `output_interval` 步 | 文本，每个源一行（该时刻的源幅值）；列之间是**两个空格**而非制表符，请按空白字符切分 |
| `frequency_slice_<idx>_<f>Hz_<k>cm-1.txt` | `frequency_analysis` 生成 | — | 文本，见下 |

**场切片**文件名示例：`field_slice_x201_00000100.bin`、`field_slice_z100_00000500.bin`。文件内容为「2 个 `int32` 尺寸 + `float32` 数据」，其中数据按剩余两个坐标轴排列，**先写的轴变化慢、后写的轴变化快**：

| `output_slice_axis` | 文件头 | 数据长度 | 数据索引（慢 → 快） |
| --- | --- | --- | --- |
| `z` | `nx`, `ny` | `nx·ny` | `field[i][j]` → `i*ny + j` |
| `x` | `ny`, `nz` | `ny·nz` | `field[j][k]` → `j*nz + k` |
| `y` | `nx`, `nz` | `nx·nz` | `field[i][k]` → `i*nz + k` |

例如 `field_slice_x201_*.bin` 是 `x = 201` 处的 **y-z 平面**，`reshape((ny, nz))` 即可（`visualize_results2.py` 与 `frequency_analysis` 都是这样读取的）。

**`monitor_point.dat` 列**：

```
# Time(s)  Ex  Ey  Ez  Dx  Dy  Dz  Hx  Hy  Hz  R  V
```

当 `atom >= natom`（或该点材料无原子）时，末尾的 `R`、`V` 两列不存在。

**`full_monitor.dat` 列**：

```
# Time(s) Ex Ey Ez Dx Dy Dz Hx Hy Hz  r0_x r0_y r0_z v0_x v0_y v0_z  r1_x ... v{N-1}_z
```

列数 = `10 + 6·natom`；第 1 行为 `#` 开头的表头（制表符分隔的列名，列名形如 `r0_x`、`v0_z`），可直接用 `numpy.loadtxt(..., skiprows=1)` 或 MATLAB `dlmread(..., '\t', 1, 0)` 读取。

**`frequency_slice_*.txt` 列**（对每个空间点做时间 FFT）：

```
# Frequency slice at <f> Hz (index: <n>)
# ... FFT 参数、切片尺寸等注释行 ...
# Format: x_index y_index real_part imag_part magnitude phase(rad) phase(deg)
< i  j  Re  Im  |E|  φ(rad)  φ(deg) >
```

注意：只输出切片中心区域（`0.25·n < i,j < 0.75·n`）的点，以控制文件体积。

---

## 9. 数值方法

### 9.1 时间推进（蛙跳格式）

每个时间步依次执行：

1. **交换 E 场的 z 方向 halo**（供磁场更新使用边界外的值）；
2. **更新 H**（显式差分旋度 + CPML 修正）；
3. **交换 H 场的 z 方向 halo**；
4. **更新 D**（`∂D/∂t = ∇×H`，含 CPML 修正）；
5. **注入源**（`D += J·dt`，软源）；
6. **由 D 求 E**（各向异性本构关系 / ADE），并同步更新离子位移 `r`、速度 `v`；
7. **按 `output_interval` 输出场切片、`full_monitor`、`source.txt`；每步输出 `monitor_point`**。

时间步长由三维 CFL 条件钳制：

```
dt = 0.99 · min(Δx, Δy, Δz) / (c·√3)
```

（当配置的 `dt` 大于该值或为负时自动采用该值。）

### 9.2 CPML 吸收边界

采用 Roden & Gedney 的卷积 PML（CPML）实现，六个面各用一组 `σ(ρ)`、`κ(ρ)`、`α(ρ)` 分布，`ρ` 为到 PML 内界面的深度：

```
r      = (L − i)/L                       # 0 在内界面，1 在最外层
σ(r)   = σ_max · r^m
κ(r)   = 1 + (κ_max − 1) · r^m
α(r)   = α_max · (1 − r)
σ_max  = −(m+1)·ln(R₀) / (2·η₀·Δ·L)
```

其中 `m = 3`（多项式阶数）、`κ_max = 7`、`α_max = 0.05`、`η₀ = √(μ₀/ε₀) ≈ 377 Ω`、`R₀ = pml_reflection`、`L = pml_thickness`。

**调参建议**（CPML 的常用工程取值区间）：

| 参数 | 常用区间 | 本程序默认 |
| --- | --- | --- |
| `m`（阶数） | 3 ~ 4 | 3 |
| `κ_max` | 3 ~ 15 | 7 |
| `α_max` | 0.005 ~ 0.1（常取 `σ_max/κ_max`） | 0.05 |
| `R₀`（`pml_reflection`） | 1e-6 ~ 1e-8 | 1e-3 |
| `pml_thickness` | 10 ~ 20 格 | 10 |

> 注意到默认 `pml_reflection = 1e-3` 比推荐值宽松得多；对反射敏感的问题（例如需要干净的近场相位）建议显式设为 `1e-6` 或更小。

场更新使用辅助变量 `ψ` 的递推形式（每个有 PML 的坐标方向、每个 H/D 分量各一个 `ψ`）：

```
b = exp(−(σ/(κε₀) + α/ε₀)·Δt)
c = σ/(κ²ε₀) · (b − 1) / (σ/(κε₀) + α/ε₀)
ψ^{n+1} = b·ψ^n + c·(空间导数)
更新式  = …/κ + ψ
```

### 9.3 由 D 求 E（各向异性 + 损耗）

对每个格点（`mat` 为该点材料），记 `x = σΔt/(2ε₀)`：

```
P_ion = Σ_atom Z*ᵀ·r                          # 离子位移贡献的极化强度（晶体坐标系）
D'    = D − P_ion
若 use_ADE = 0:  E = ε⁻¹·D' / (ε₀·(1+x)²)     # 各向异性本构 + 电导损耗
若 use_ADE = 1:  E = (D' − P_ADE) / (ε₀·EPS_INF·(1+x))
```

即：先把 `D`、`E` 旋转到材料晶体坐标系，用材料介电张量的逆矩阵求 `E`，再旋转回仿真坐标系；`σ` 的损耗以 `1/(1+x)` 的因子施加（`use_ADE=0` 时该因子在代码中出现两次，故等效为 `1/(1+x)²`，一阶精度上等价于 `exp(−σΔt/ε₀)`）。`σ_m` 为磁损耗参数，当前版本未参与更新。

### 9.4 各向异性本构关系与坐标旋转

- **绕 z 轴**（`anglez = θ`）：`E_crystal = R_z(θ)·E`（`R_z` 为标准旋转矩阵），更新后再用 `R_z(−θ)` 变换回仿真系。
- **绕 x 轴**（`anglex = φ`）：同理在 y-z 平面内旋转。
- **`switchxz = 1`**：在读入阶段直接交换张量的 x/z 分量（`ε_xx ↔ ε_zz`、`ε_xy ↔ ε_zy`、`ε_yx ↔ ε_yz`、`ε_xz ↔ ε_zx`，以及对应的虚部，`born`/`QeZV`/`QeZM`/`DMM` 一并处理）。这比旋转更便宜，适合"晶轴与仿真轴只差一次置换"的情形。
- 若 `anglez` 非零则忽略 `anglex`。

### 9.5 负介电常数与 ADE 方法

当材料在某频率下介电函数实部为负（金属、极性晶体剩余射线带等），直接对本构关系求逆会得到非物理结果，此时应设 `use_ADE=1`。程序把色散介电函数写成"高频背景 ε∞ + 张量形式振子"的 Drude / Drude-Lorentz 形式：

```
ε(ω) = ε∞·I − ω_p² / (ω² + i·γ·ω)        （各向异性时 γ、ω_p² 为 3×3 张量）
```

由此得到极化强度的时域运动方程（辅助微分方程，ADE）：

```
d²P/dt² + γ·dP/dt = ε₀·ω_p²·E
```

在程序中离散为（`P_ADE` 为当前步、`P_ADEo` 为上一步）：

```
(1 + γΔt/2)·P^{n+1} = 2P^n − (1 − γΔt/2)·P^{n−1} + ε₀Δt²·ω_p²·E^n
E = (D − P^{n+1}) / (ε₀·EPS_INF)
```

其中张量 `γ`、`ω_p²` 由**第一个源的频率** ω 以及材料的 `eps_*`/`eps_*i` 反推（`εr1 = ε_real − EPS_INF·I` 须可逆；`γ = −ω·ε_imag·εr1⁻¹`；`ω_p² = −ω²·εr1 + ω·γ·ε_imag`），使得在 ω 处的介电函数恰好还原为输入值。

> **重要**：`EPS_INF` 是**编译期常量**（`#define EPS_INF 20.0`）。它只影响 `use_ADE=1` 的材料；如果实际材料的 ε∞ 与之不符，请修改后**重新编译**（程序会在 `εr1` 不可逆时提示 `EPS_INF is too small`）。
>
> **重要**：`γ`、`ω_p²` 只对**第一个源的频率**精确；多频或宽带源下 ADE 只是在单频点上的近似。此外 ADE 假定色散是"单一振子 + 常数背景"，宽频带模拟前请先用 `Lorentz_dielectric/` 里的脚本核对介电函数表。

### 9.6 离子运动方程（PhP 耦合）

对 `natom > 0` 的材料，每个格点额外维护 `3·natom` 个位移自由度（`r`）与速度（`v`），在材料晶体坐标系中演化：

```
a_i = Σ_j Z*_ij · E_j − Σ_{j,β} DMM_{iα,jβ} · r_{jβ}     # 加速度
r ← r + v·Δt + ½·a·Δt²                                    # 速度 Verlet
v ← v + a·Δt
P_ion,i = Σ_j Z*_ij · r_j                                  # 反馈到 Maxwell 方程
```

- `Z*` 由 `QeZM`（加速度耦合）与 `QeZV`（极化耦合）给出，`DMM` 为质量加权动力学矩阵。
- 这是一个**辛（symplectic）式**的速度 Verlet 步，与 Maxwell 部分交替推进，因此总能量在稳定条件下守恒良好。
- `r`、`v` 数组**只对 `natom > 0` 的格点分配**，因此内存与"含原子材料的体积"成正比。
- 稳定性要求 `Δt < 2/√λ_max(DMM)`（最高声子频率决定）；若模拟发散，请先检查 `Δt` 与 `DMM` 的单位是否自洽。

> ⚠️ **发散保护**：程序每隔 `output_interval` 步检查一次全局最大位移，一旦 `r_max > 1e-10 m`（即 1 Å）就认为离子子系统已经发散，打印诊断信息后 **`MPI_Abort` 直接终止**。诊断内容包括：超出阈值的步数、最大位移位置 `(i, j, k)`、原子序号与方向、该点的速度 `v` 与加速度 `a`（已分解为 `QeZM·E` 驱动力与 `DMM·r` 恢复力两部分），可据此判断是场太强、数据单位不自洽还是 `dt` 过大：
>
> ```
> ERROR: r field too large at step 4000
> Max r value: 3.2e-09
> Location: i=40, j=40, k=20 (global k=20)
> Atom: 0, Direction: 0
> v value: ...
> a value: ...
> ```
>
> 注意 1 Å 的阈值同时也意味着该模型只在**线性（简谐）近似**成立的小位移区间内有效——超出这一区间时，即使没有触发终止，结果也已不可信。

### 9.7 介电函数虚部与电导的等价性

材料的介电函数虚部 `eps_*i` 与电导率 `σ` 是同一物理量的两种描述方式。采用本程序的时谐约定 `D ∝ e^{−iωt}` 时，对**相对**介电常数的虚部有

```
σ = ε₀ · ω · ε_imag        （因此 ε_imag ≥ 0 表示损耗，必须为正）
```

所以由 `dielectric=` 文件给出的介电函数表只需填写（正的）虚部，不需要再额外换算成 `sigma`；反之，若某材料的损耗是用 `sigma=` 给出的，也无需再填虚部。当材料设了 `use_ADE = 1` 时，虚部还会参与反推 ADE 的阻尼张量 `γ`（见 [9.5](#95-负介电常数与-ade-方法)）。

---

## 10. 后处理工作流

```
simulation.cfg ──► generate_medium5.py ──► medium_map.bin
      │                                          │
      ├──► visualize_input3.m（几何/源检查）      │
      │                                          ▼
      └──────────────────────────────────►  fdtd.x（MPI）──► field_slice_*.bin
                                                    │        monitor_point.dat
                                                    │        full_monitor.dat
                                                    │        source.txt
                                                    ▼
   visualize_medium.m（介质分布检查）          ┌───┴────────────────────┐
                                               ▼                        ▼
                                visualize_results2*.py        frequency_analysis（FFT）
                                （场分布快照 / 动画）                     │
                                                                        ▼
                                                          frequency_slice_*.txt
                                                          （用 Origin 等画近场图）
```

### 10.1 `generate_medium5.py` — 生成介质分布

```bash
python3 generate_medium5.py simulation.cfg [medium_map.bin]
```

- 第 1 个参数：配置文件；第 2 个参数：输出文件名（默认为 `medium_map.bin`）。
- 从 `[medium]` 段解析 `cubes` / `hexahedra` / `cylinders`，生成 `int32` 数组并写入二进制。
- 依赖 NumPy；会打印每种材料占据的体素数量与百分比，便于核对几何是否正确。

### 10.2 `visualize_input3.m` / `visualize_medium.m` — 输入检查（MATLAB）

- `visualize_input3.m`：读取 `simulation.cfg`（默认文件名写在脚本第 4 行），3D 绘制立方体/六面体/圆柱体与源位置。
- `visualize_medium.m`：读取 `medium_map.bin`（默认文件名写在脚本第 5 行），逐切片读取以节省内存，绘制介质分布。

### 10.3 `visualize_results2.py` / `visualize_results2_switchxy.py` — 场分布快照

```bash
python3 visualize_results2.py
```

- 自动 `glob` 当前目录下的 `field_slice_*.bin`，打印每帧 `max|field|`，并把前 30 帧画成子图。
- 输出文件：`field_evolution.png`（前 30 帧网格图）、`last_frame.png`（最后一帧）、`field_evolution.gif`（全部帧的动画，200 ms/帧），同时弹出交互窗口。
- `switchxy` 版本在绘图时交换 x/y 轴，用于 x/y 颠倒的切片。
- 需要 NumPy + Matplotlib。**注意**：脚本按字典序读取文件，不同切片位置/轴的文件会混在一起，建议在专用目录中运行。

### 10.4 `frequency_analysis` — 近场频率切片（s-SNOM 核心工具）

```bash
./frequency_analysis simulation.cfg
```

- 自动扫描**当前目录**下所有**文件名中同时含 `field_slice_` 与 `.bin`** 的文件，从文件名最后一个 `_` 之后的数字解析出时间步并排序构成时间序列；
- 由配置文件读取 `dt`、`output_interval` 与**第一个源**的频率（`tsteps`/`nx`/`ny`/`nz` 会被读取但**不使用**，网格尺寸以二进制文件头为准），据此确定采样率与目标频率；
- 对每个空间点沿时间做 FFT（FFTW3，**不加窗、不去趋势**，结果除以 `N` 归一化），输出目标频率**及其前后各 4 个**频率共最多 9 个切片，文件名为
  `frequency_slice_<index>_<f>Hz_<k>cm-1.txt`；
- 每个文件含 `Re`、`Im`、`|E|`、相位（弧度/度），可直接用 Origin / Python 画近场振幅与相位图；
- 为控制文件体积，结果文件**只包含每个维度 25%–75% 之间的中心区域**；若需要完整切片，请修改 `save_frequency_result()` 中的判断条件。

> **使用要点**：
> 1. 运行前请**删除源仍在振荡期间的场切片**（例如 `t < t0 + 6·tau` 的那些），只保留源已衰减到零之后的"自由振荡/近场响应"阶段，否则源自身的频谱会污染结果。文件名中的步数即 `t/dt`，可据此筛选。
> 2. 该方法假定场切片在时间上**等间隔**，因此不要混入不同 `output_interval` 的残留文件，也不要在同一目录中混放多次运行的结果。
> 3. FFT 频率分辨率 = `1/(N·Δt)`，其中 `N` 为切片数；想要更细的频率分辨率就需要更多切片（更小的 `output_interval` 或更多 `tsteps`）。

### 10.5 `Lorentz_dielectric/*.m` — 介电函数表生成（MATLAB）

`Lorentz_Ag.m`、`Lorentz_Au.m`、`Lorentz_ARP.m`、`Lorentz_MoO3.m` 分别用 Drude-Lorentz（金属）或各向异性 Lorentz（极性晶体）模型**解析地生成**（不是拟合）15 列的介电函数表，供 `dielectric=` 使用。修改脚本顶部的 ε∞、ω_LO、ω_TO、γ 参数即可得到新材料的数据。

| 脚本 | 模型 |
| --- | --- |
| `Lorentz_Ag.m` | Drude 项 + 5 个 Lorentz 振子 |
| `Lorentz_Au.m` | 单一自由电子 Drude 项 |
| `Lorentz_ARP.m` | 双 Lorentz 振子（s²/(ω²−ω²−iγω) 形式） |
| `Lorentz_MoO3.m` | 各向异性 LO–TO 单振子模型，x/y/z 各一组参数 |

> 注意：这些脚本运行时会**覆盖**同目录下的同名 `.txt`（且要求当前目录就是 `Lorentz_dielectric/`，因为输出用的是裸文件名），请勿在未备份的情况下重跑。

### 10.6 `v_source.m` — 源波形预览（MATLAB）

几行脚本，按 `v_source.m` 顶部的 `amplitude`/`wavenumber`/`t0`/`tau` 画出源的时域波形，用于在正式运行前确认源的频谱与持续时间是否合适。

---

## 11. 并行与性能

### 11.1 域分解

- 仅沿 **z 方向**做一维分解：`base = nz/size`，前 `nz % size` 个进程各多分一层。
- 每个进程的局部数组尺寸为 `nx × ny × (local_nz + 2)`，额外 2 层用于 z 方向的 halo（含全局边界外的死单元）。
- 每步做 2 次 halo 交换（更新 H 前交换 E，更新 D 前交换 H），使用**打包 + 预分配缓冲区**的矢量通信（先 `MPI_Sendrecv` 交换边界，再 `MPI_Bcast` 广播 halo 值）。
- 要求 `size ≤ nz`（否则某进程分不到格点，程序提示 `Process N has no grid points!`）。
- 由于只有 z 方向并行，x/y 方向的网格数不宜过大而 z 过小，否则并行效率与内存分担都会变差。

### 11.2 内存估算

设 `nz_local = nz/size + 2`，`V = nx·ny·nz_local`：

| 数组 | 大小 |
| --- | --- |
| 9 个场分量（Ex…Hz） | `9 × V × 8 B` |
| 电场模 `E`（仅 `output_component = E` 时分配） | `V × 8 B` |
| 介质分布 `medium` | `V × 4 B` |
| 离子位移/速度 `r`,`v` | `2 × 3·natom × V_atom × 8 B`（**仅** `natom>0` 的格点 `V_atom`） |
| ADE 极化 `P_ADE`,`P_ADEo` | `2 × 3 × V_ADE × 8 B`（仅 `use_ADE=1` 的格点，每格点固定 3 个分量） |
| CPML 辅助场 `ψ`（x/y 法向的 8 个） | `8 × 2·pml_thickness × (ny 或 nx) × nz_local × 8 B`（每个进程都有） |
| CPML 辅助场 `ψ`（z 法向的 4 个） | `4 × V × 8 B`（**仅** z 区间与 z 方向 PML 重叠的进程分配） |

其中 `V = nx·ny·nz_local`，`nz_local = local_nz + 2·halo`。

程序在启动时和运行中会把实际占用打印到 stderr（`[MEM]` 行，只统计 ≥ 0.1 GB 的分配），例如：

```
[MEM] Before time loop: per-rank = 1.234567 GB, total (28 ranks) = 34.567876 GB
```

> **内存热点**：`r`/`v` 与 `natom` 成正比。`natom=16` 的材料每个格点需要 `2×48×8 = 768 B`，百万级格点就会占用数百 GB。做 PhP 模拟时请优先缩小含原子材料的区域，或增加 MPI 进程数（内存按 z 层分担）。

### 11.3 运行建议

- 编译使用 `-Ofast -flto -funroll-loops`（作者推荐），可显著加速热点三重循环。
- 输出频率（`output_interval`）对性能影响很大：每次输出要写整个切片并做集合通信，`output_interval` 太小会成为 I/O 瓶颈。
- `full_monitor` 与 `monitor_point` 的开销可以忽略（只写单点），但 `monitor_point` 每步都写、行数 = `tsteps`，长时间模拟会产生较大文本文件。
- 进度打印只在 0 号进程输出；每次打印要做 2 次 `MPI_Allreduce`，`output_interval` 很小时也会带来同步开销。

---

## 12. 已知限制与注意事项

1. **只支持 CPML 吸收边界**，周期性边界（`bc_x/y/z`）未实现。
2. **并行只沿 z 方向**分解，x/y 方向不分解。
3. **时间是显式推进**，`dt` 受三维 CFL 条件限制（`dt ≤ 0.99·Δ_min/(c√3)`）；**未实现** ADI-FDTD 等无条件稳定格式，因此细网格 + 长时间模拟的代价较高。
4. **`EPS_INF` 是编译期常量**（默认 20.0），只影响 `use_ADE=1` 的材料；换材料需要重新编译。
5. **ADE 系数与介电函数表插值都只针对第一个源的频率**，多频/宽带问题需要分频段多次运行。
6. **`source` / `material` 行必须单行书写**，且必须从第 0 列开始；值中不能有空格或逗号。
7. **源参数请全部显式给出**：`x/y/z/t0/tau` 的默认值是在读入 `nx`、`dt` 之前计算的，因此省略时会与预期不符，建议逐项写全。
8. **`dt` 被 CFL 钳制后不会自动调整 `tsteps`**，模拟的总物理时长可能与你以为的不同。
9. **不解析 `dt` 之外的稳定性参数**，也不会检查 `DMM` 是否满足离子子系统稳定性条件；发散时请自行减小 `dt`。
10. **`born=<OUTCAR>` 会用静态介电张量覆盖 `eps_*`**，而 Born 电荷数组本身未参与更新（详见 [5.3](#53-材料-material)）。
11. **`spheres` / `slabs` / `points` 几何未实现**（`generate_medium5.py` 只支持 `cubes`、`hexahedra`、`cylinders`）。
12. **材料数据文件名大小写不一致**（`QeZM_clean.txt` vs `QeZm_clean.txt`），在区分大小写的系统上需注意。
13. **场切片输出无目录隔离**：所有输出都写在当前工作目录，多次运行的 `field_slice_*` 会混在一起并影响 `frequency_analysis`（它按文件名扫描整个目录）。建议每次运行使用独立目录。
14. **输入不做自动纠错**：材料 id 越界、`natom` 与数据文件长度不符、`medium_map.bin` 尺寸不符等都会直接 `MPI_Abort`，请对照本文档仔细核对输入。
15. `-Ofast` 会启用非 IEEE 严格语义的浮点优化；如需严格复现，请改用 `-O3`。
16. 代码注释与运行日志以**中文**为主。

---

## 13. 常见问题排查

| 现象 | 可能原因与处理 |
| --- | --- |
| `Error opening medium file` | `medium_file` 路径错误，或没有先运行 `generate_medium5.py` |
| `Medium file dimensions (...) do not match simulation (...)` | 配置文件里的 `nx/ny/nz` 与生成 `medium_map.bin` 时不一致，重新生成 |
| `Process N has no grid points!` | MPI 进程数 > `nz`，减少 `-np` 或增大 `nz` |
| `Error: QeZM is not set!` / `Error: DMM is not set!` | 材料设了 `natom>0` 但没有给出 `QeZM`/`DMM`（或文件名拼写/大小写不对） |
| `Error: Only N values read (expected M)` | `QeZV`/`QeZM`/`DMM` 文件元素个数与 `natom` 不匹配（应为 `9N`、`9N`、`9N²`） |
| `Error: epsr1 can not be inversed` | `use_ADE=1` 但 `epsr1 = ε_real − EPS_INF` 奇异；检查 `EPS_INF` 与介电张量 |
| `EPS_INF is too small for material N` | `EPS_INF` 小于等于 ε 实部，请修改 `#define EPS_INF` 后重新编译 |
| `Warning: No exact frequency match in ...` | 第一个源的频率与介电函数表的频率网格差距 > 10%，检查 `wavenumber` 换算或表格频率范围 |
| `No slice files found!` | `frequency_analysis` 在错误的目录运行，或还没有产生场切片 |
| 模拟发散（场值指数增长） | ① `dt` 超过 CFL；② `use_ADE` 设置不当或 `EPS_INF` 不符；③ `DMM`/`QeZM` 单位不自洽导致离子子系统不稳定；④ CPML 参数过强/过弱 |
| `ERROR: r field too large at step N` 后进程被杀 | 离子位移超过 1 Å 的发散保护（见 [9.6](#96-离子运动方程php-耦合)）。按提示的 `i,j,k`/原子/方向定位问题点：通常是源幅度过大、`QeZV`/`QeZM`/`DMM` 三者单位或坐标约定不一致，或 `dt` 过大 |
| 场值不衰减、边界反射明显 | `pml_thickness` 太小（建议 ≥ 10–20）、`pml_reflection` 太大，或源/结构离 PML 太近 |
| `frequency_analysis` 结果有奇怪的频率成分 | 场切片目录中混入了源振荡期间的切片或多次运行的文件；清理后重跑 |
| 近场图相位混乱 | 未剔除源未衰减完的时间段；应只用 `t > t0 + 6·tau` 之后的切片 |
| OpenMPI 4 在程序结束时打印错误信息 | 这是 OpenMPI 4 自身的已知问题，不影响结果；可换用 OpenMPI 5 |

---

## 14. 参考与致谢

程序中的 CPML 实现与参数选取参考了：

- J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media," *Microwave and Optical Technology Letters* **27**(5), 334–339 (2000).
- S. D. Gedney, *Introduction to the Finite-Difference Time-Domain (FDTD) Method for Electromagnetics*, Morgan & Claypool (2011).
- K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.* **14**(3), 302–307 (1966).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).

材料数据来自第一性原理计算：Born 有效电荷张量与介电张量取自 VASP 静态计算，动力学矩阵取自 Γ 点声子计算（有限位移法）。数据文件的坐标约定见各目录内的 `README.txt`。感谢 VASP、phonopy 等开源/商用第一性原理工具，以及 FFTW、NumPy、Matplotlib 等基础库。

---

## 15. 许可证

本项目采用 **MIT License**，详见 [LICENSE](LICENSE)。

---

## 贡献

欢迎提交 Issue 与 Pull Request。提交代码前请尽量保证：

1. 不引入新的编译警告；
2. 新增的输出文件格式在 README 中同步说明；
3. 涉及物理公式的改动请附推导或参考资料。
