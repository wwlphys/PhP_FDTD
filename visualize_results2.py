import numpy as np
import matplotlib.pyplot as plt
import glob

# 读取场切片文件
def read_field_slice(filename):
    with open(filename, 'rb') as f:
        nx = np.frombuffer(f.read(4), dtype=np.int32)[0]
        ny = np.frombuffer(f.read(4), dtype=np.int32)[0]
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((nx, ny))

files = sorted(glob.glob("field_slice_*.bin"))

# 打印每个切片 |field| 的最大值
for i, file in enumerate(files):          # 处理所有切片
    field = read_field_slice(file)
    max_abs = np.abs(field).max()              # 计算模的最大值
    print(f"Time step {i:6d}: max|field| = {max_abs:.6g}")

# 可视化所有场切片并输出每个切片上的最大值
plt.figure(figsize=(10, 8))

for i, file in enumerate(files[:30]):  # 显示前30个切片
    field = read_field_slice(file)
    vmax = field.max()                        # 计算最大值
    
    plt.subplot(6, 5, i+1)
    plt.imshow(field.T, cmap='jet', origin='lower')
    plt.title(f"t={i}, max={vmax:.3g}")   # 把最大值显示在标题里
    plt.colorbar()

plt.tight_layout()
plt.savefig("field_evolution.png")
plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------- 1. 读入所有切片 ----------
fields = []
for file in files:
    fields.append(read_field_slice(file))          # list of 2-D array
fields = np.asarray(fields)                        # shape = (nT, nx, ny)
nT, nx, ny = fields.shape

# ---------- 2. 统一 colorbar 范围 ----------
vmin, vmax = fields.min(), fields.max()

# ---------- 新增功能：输出最后一帧为文件 ----------
plt.figure(figsize=(8, 6))
last_field = fields[-1]  # 获取最后一帧数据
plt.imshow(last_field.T, cmap='jet', origin='lower',
               vmin=vmin/10000000000, vmax=vmax/10000000000)
plt.colorbar(label='|field|')
plt.title(f'Last Frame (t = {(nT-1)})')
plt.tight_layout()
plt.savefig('last_frame.png', dpi=150, bbox_inches='tight')
plt.close()  # 关闭图形，释放内存
print("最后一帧已保存为 'last_frame.png'")

# ---------- 3. 画布与图像对象 ----------
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(fields[0].T,
               cmap='jet', origin='lower',
               vmin=vmin/10000000000, vmax=vmax/10000000000)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('|field|')
ttl = ax.set_title('')

# ---------- 4. 更新函数 ----------
def update(frame):
    im.set_array(fields[frame].T)
    ttl.set_text(f't = {frame}')   # 假设 20 步一个切片，可自行修改
    return [im, ttl]

# ---------- 5. 生成动画 ----------
ani = animation.FuncAnimation(fig,
                              update,
                              frames=nT,
                              interval=200,   # 每帧 200 ms
                              blit=True)

# ---------- 6. 保存 ----------
# 6-a) gif（需 imagemagick 或 pillow）
ani.save('field_evolution.gif', writer='pillow', fps=5)

# 6-b) mp4（需 ffmpeg）
#ani.save('field_evolution.mp4', writer='ffmpeg', fps=10)

plt.show()

