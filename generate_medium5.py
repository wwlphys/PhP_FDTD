import numpy as np
import struct
import re
import sys
import os
import ast

def read_config(config_file):
    """从配置文件读取参数，支持立方体(cubes)、平行六面体(hexahedra)、圆柱体(cylinders)"""
    nx, ny, nz = 50, 50, 50  # 默认值
    dx, dy, dz = 1.0, 1.0, 1.0  # 默认网格间距（物理单位）
    
    with open(config_file, 'r') as f:
        content = f.read()
        
        # 提取网格参数
        nx_match = re.search(r'nx\s*=\s*(\d+)', content)
        ny_match = re.search(r'ny\s*=\s*(\d+)', content)
        nz_match = re.search(r'nz\s*=\s*(\d+)', content)
        
        if nx_match: nx = int(nx_match.group(1))
        if ny_match: ny = int(ny_match.group(1))
        if nz_match: nz = int(nz_match.group(1))
        
        # 提取网格间距（物理单位，如米）
        dx_match = re.search(r'delta_x\s*=\s*([\d.eE+-]+)', content)
        dy_match = re.search(r'delta_y\s*=\s*([\d.eE+-]+)', content)
        dz_match = re.search(r'delta_z\s*=\s*([\d.eE+-]+)', content)
        
        if dx_match: dx = float(dx_match.group(1))
        if dy_match: dy = float(dy_match.group(1))
        if dz_match: dz = float(dz_match.group(1))
    
    print(f"[DEBUG] Grid size: nx={nx}, ny={ny}, nz={nz}")
    print(f"[DEBUG] Grid spacing: dx={dx:.3e}, dy={dy:.3e}, dz={dz:.3e}")
    print(f"[DEBUG] Valid indices: x:0-{nx-1}, y:0-{ny-1}, z:0-{nz-1}")
    
    medium = np.zeros((nx, ny, nz), dtype=np.int32)
    
    # 查找 [medium] 部分
    medium_match = re.search(r'\[medium\](.*?)(?=\n\[|\Z)', content, re.DOTALL | re.IGNORECASE)
    if not medium_match:
        print("[DEBUG] No [medium] section found")
        return medium
    
    medium_content = medium_match.group(1).strip()
    print(f"[DEBUG] Medium section content: {medium_content[:200]}...")
    
    # ---------- 处理立方体 (cubes) ----------
    cubes_match = re.search(r'cubes\s*=\s*\[(.*?)\]', medium_content, re.DOTALL)
    if cubes_match:
        cubes_str = cubes_match.group(1).strip()
        cubes_str = re.sub(r'#.*', '', cubes_str)          # 移除注释
        cubes_str = cubes_str.replace('\n', ' ')
        cubes_str = re.sub(r'\s+', ' ', cubes_str)
        
        dict_pattern = r'\{\s*"xrange"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*"yrange"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*"zrange"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\),\s*"material"\s*:\s*(\d+)\s*\}'
        for match in re.finditer(dict_pattern, cubes_str):
            xmin, xmax, ymin, ymax, zmin, zmax, material = map(int, match.groups())
            cubes = [{
                'xrange': (xmin, xmax),
                'yrange': (ymin, ymax),
                'zrange': (zmin, zmax),
                'material': material
            }]
            for cb in cubes:
                x_start = max(cb['xrange'][0], 0)
                x_end   = min(cb['xrange'][1] + 1, nx)
                y_start = max(cb['yrange'][0], 0)
                y_end   = min(cb['yrange'][1] + 1, ny)
                z_start = max(cb['zrange'][0], 0)
                z_end   = min(cb['zrange'][1] + 1, nz)
                if x_start < x_end and y_start < y_end and z_start < z_end:
                    medium[x_start:x_end, y_start:y_end, z_start:z_end] = cb['material']
    
    # ---------- 处理平行六面体 (hexahedra) ----------
    hexahedra_list = extract_hexahedra(medium_content)
    for hex_item in hexahedra_list:
        apply_hexahedron(medium, hex_item, nx, ny, nz)
    
    # ---------- 处理圆柱体 (cylinders) ----------
    cylinders_list = extract_cylinders(medium_content)
    for cyl_item in cylinders_list:
        apply_cylinder(medium, cyl_item, nx, ny, nz, dx, dy, dz)
    
    return medium

def extract_hexahedra(content):
    """从介质配置内容中提取六面体列表，返回 [{'vertices': list, 'material': int}]"""
    # 移除所有注释（行内 # 及之后内容）
    content = re.sub(r'#.*', '', content)
    
    # 查找 hexahedra = 关键字
    match = re.search(r'hexahedra\s*=', content, re.IGNORECASE)
    if not match:
        return []
    
    start_pos = match.end()
    slice_str = content[start_pos:]
    
    # 找到列表的起始 '['
    list_start = slice_str.find('[')
    if list_start == -1:
        return []
    
    # 括号匹配，提取完整的列表字符串
    balance = 0
    in_string = False
    list_str = ''
    i = list_start
    while i < len(slice_str):
        ch = slice_str[i]
        list_str += ch
        if ch == '[' and not in_string:
            balance += 1
        elif ch == ']' and not in_string:
            balance -= 1
            if balance == 0:
                break
        i += 1
    else:
        print("[ERROR] Unmatched brackets in hexahedra definition")
        return []
    
    # 使用 ast.literal_eval 安全解析 Python 字面量
    try:
        raw_list = ast.literal_eval(list_str)
    except Exception as e:
        print(f"[ERROR] Failed to parse hexahedra list: {e}")
        return []
    
    validated = []
    for item in raw_list:
        if not isinstance(item, dict):
            print(f"[WARNING] Hexahedron definition is not a dict: {item}")
            continue
        if 'vertices' not in item or 'material' not in item:
            print(f"[WARNING] Missing 'vertices' or 'material' in hexahedron: {item}")
            continue
        vertices = item['vertices']
        material = item['material']
        if len(vertices) != 8:
            print(f"[WARNING] Hexahedron has {len(vertices)} vertices, expected 8. Skipping.")
            continue
        # 转换所有顶点坐标为浮点数
        try:
            verts = [tuple(float(coord) for coord in v) for v in vertices]
        except Exception as e:
            print(f"[WARNING] Invalid vertex coordinates: {e}")
            continue
        validated.append({'vertices': verts, 'material': material})
    
    return validated

def apply_hexahedron(medium, hex_item, nx, ny, nz):
    """
    将平行六面体填充到介质数组中（向量化版本）。
    顶点顺序约定：底面逆时针四个点 v0,v1,v2,v3，顶面对应点 v4,v5,v6,v7。
    内部判断基于仿射变换 p = v0 + α*u + β*v + γ*w，其中 u=v1-v0, v=v3-v0, w=v4-v0。
    """
    vertices = hex_item['vertices']
    material = hex_item['material']

    # 计算包围盒
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    xmin = max(0, int(np.floor(min(xs))))
    xmax = min(nx-1, int(np.ceil(max(xs))))
    ymin = max(0, int(np.floor(min(ys))))
    ymax = min(ny-1, int(np.ceil(max(ys))))
    zmin = max(0, int(np.floor(min(zs))))
    zmax = min(nz-1, int(np.ceil(max(zs))))

    # 提取基点及三个边向量
    v0 = np.array(vertices[0])
    u  = np.array(vertices[1]) - v0
    v  = np.array(vertices[3]) - v0
    w  = np.array(vertices[4]) - v0

    A = np.column_stack((u, v, w))
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print(f"[ERROR] Degenerate hexahedron (non-invertible matrix), skipping.")
        return

    # 向量化：一次生成包围盒内所有格点
    xi = np.arange(xmin, xmax + 1)
    yi = np.arange(ymin, ymax + 1)
    zi = np.arange(zmin, zmax + 1)
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')
    shape_3d = X.shape
    N = X.size

    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))  # (N, 3)
    abc = invA @ (points - v0).T  # (3, N)

    tol = 1e-9
    mask_flat = (
        (abc[0] >= -tol) & (abc[0] <= 1+tol) &
        (abc[1] >= -tol) & (abc[1] <= 1+tol) &
        (abc[2] >= -tol) & (abc[2] <= 1+tol)
    )

    mask_3d = mask_flat.reshape(shape_3d)
    medium[X[mask_3d], Y[mask_3d], Z[mask_3d]] = material
    count = np.sum(mask_flat)
    print(f"[DEBUG] Hexahedron material {material} filled {count} voxels")

def extract_cylinders(content):
    """从介质配置内容中提取圆柱体列表(两端坐标+半径)，返回 [{'p1': tuple, 'p2': tuple, 'radius': float, 'material': int}]"""
    # 移除所有注释（行内 # 及之后内容）
    content = re.sub(r'#.*', '', content)

    # 查找 cylinders = 关键字
    match = re.search(r'cylinders\s*=', content, re.IGNORECASE)
    if not match:
        return []

    start_pos = match.end()
    slice_str = content[start_pos:]

    # 找到列表的起始 '['
    list_start = slice_str.find('[')
    if list_start == -1:
        return []

    # 括号匹配，提取完整的列表字符串
    balance = 0
    in_string = False
    list_str = ''
    i = list_start
    while i < len(slice_str):
        ch = slice_str[i]
        list_str += ch
        if ch == '[' and not in_string:
            balance += 1
        elif ch == ']' and not in_string:
            balance -= 1
            if balance == 0:
                break
        i += 1
    else:
        print("[ERROR] Unmatched brackets in cylinders definition")
        return []

    # 使用 ast.literal_eval 安全解析 Python 字面量
    try:
        raw_list = ast.literal_eval(list_str)
    except Exception as e:
        print(f"[ERROR] Failed to parse cylinders list: {e}")
        return []

    validated = []
    for item in raw_list:
        if not isinstance(item, dict):
            print(f"[WARNING] Cylinder definition is not a dict: {item}")
            continue
        if 'p1' not in item or 'p2' not in item or 'radius' not in item or 'material' not in item:
            print(f"[WARNING] Missing required fields (p1, p2, radius, material) in cylinder: {item}")
            continue
        p1 = tuple(float(c) for c in item['p1'])
        p2 = tuple(float(c) for c in item['p2'])
        radius = float(item['radius'])
        material = int(item['material'])
        validated.append({'p1': p1, 'p2': p2, 'radius': radius, 'material': material})

    return validated


def apply_cylinder(medium, cyl_item, nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    """
    将任意方向圆柱体填充到介质数组中（向量化版本）。
    p1, p2: 圆柱体两端坐标（格子索引）
    radius: 半径（物理单位，与 delta_x/delta_y/delta_z 一致）
    delta_x/y/z: 网格间距（物理单位），用于处理各向异性网格
    """
    p1 = np.array(cyl_item['p1'])
    p2 = np.array(cyl_item['p2'])
    radius = cyl_item['radius']
    material = cyl_item['material']

    # 缩放因子：将格子索引转为物理坐标
    scale = np.array([dx, dy, dz])

    # 转到物理空间计算
    p1_phys = p1 * scale
    p2_phys = p2 * scale

    v = p2_phys - p1_phys            # 轴线方向向量（物理空间）
    length = np.linalg.norm(v)
    if length < 1e-12:
        print(f"[WARNING] Cylinder p1==p2 (zero length), skipping.")
        return
    v_hat = v / length               # 单位方向向量

    r2 = radius * radius

    # 包围盒（格子索引空间）
    r_cells_x = radius / dx
    r_cells_y = radius / dy
    r_cells_z = radius / dz
    all_x = [p1[0], p2[0]]
    all_y = [p1[1], p2[1]]
    all_z = [p1[2], p2[2]]
    xmin = max(0, int(np.floor(min(all_x) - r_cells_x)))
    xmax = min(nx - 1, int(np.ceil(max(all_x) + r_cells_x)))
    ymin = max(0, int(np.floor(min(all_y) - r_cells_y)))
    ymax = min(ny - 1, int(np.ceil(max(all_y) + r_cells_y)))
    zmin = max(0, int(np.floor(min(all_z) - r_cells_z)))
    zmax = min(nz - 1, int(np.ceil(max(all_z) + r_cells_z)))

    # 向量化：一次生成包围盒内所有格点
    xi = np.arange(xmin, xmax + 1)
    yi = np.arange(ymin, ymax + 1)
    zi = np.arange(zmin, zmax + 1)
    X, Y, Z = np.meshgrid(xi, yi, zi, indexing='ij')
    shape_3d = X.shape

    # 转为物理坐标
    px = X * dx
    py = Y * dy
    pz = Z * dz

    # 向量 u = P - p1_phys
    ux = px - p1_phys[0]
    uy = py - p1_phys[1]
    uz = pz - p1_phys[2]

    # 投影到轴上的标量长度
    proj = ux * v_hat[0] + uy * v_hat[1] + uz * v_hat[2]

    # 垂直分量
    perpx = ux - proj * v_hat[0]
    perpy = uy - proj * v_hat[1]
    perpz = uz - proj * v_hat[2]
    perp2 = perpx**2 + perpy**2 + perpz**2

    # 综合 mask
    mask = (proj >= 0.0) & (proj <= length) & (perp2 <= r2)

    medium[X[mask], Y[mask], Z[mask]] = material
    count = np.sum(mask)

    print(f"[DEBUG] Cylinder (r={radius:.3e} m, |v|={length:.3e} m) material {material} filled {count} voxels")

def save_medium_file(medium, output_file):
    """保存介质分布到二进制文件（与原来相同）"""
    nx, ny, nz = medium.shape
    
    print(f"\n[DEBUG] Saving medium file with shape {nx}x{ny}x{nz}")
    print(f"[DEBUG] Material distribution summary:")
    unique, counts = np.unique(medium, return_counts=True)
    for mat_id, count in zip(unique, counts):
        print(f"  Material {mat_id}: {count} voxels ({count/(nx*ny*nz)*100:.2f}%)")
    
    vacuum_count = np.sum(medium == 0)
    print(f"  Vacuum (0): {vacuum_count} voxels ({vacuum_count/(nx*ny*nz)*100:.2f}%)")
    
    with open(output_file, 'wb') as f:
        f.write(struct.pack('iii', nx, ny, nz))
        # 按 x 切片一次性写入（Fortran-order: x最慢, y次之, z最快）
        # medium[i, :, :] 在 C-order 内存中恰好是按 y 最慢、z 最快的连续块
        for i in range(nx):
            f.write(medium[i, :, :].tobytes())
            if (i+1) % 100 == 0:
                print(f"[DEBUG]   Written {i+1}/{nx} x-slices...")
    
    print(f"\nMedium file '{output_file}' created successfully")
    print(f"Dimensions: {nx}x{ny}x{nz}")
    print(f"Total size: {os.path.getsize(output_file):,} bytes")

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_medium2.py <config_file> [output_file]")
        print("Example: python generate_medium2.py simulation.cfg medium_map.bin")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "medium_map.bin"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    
    print(f"[DEBUG] Reading config from: {config_file}")
    medium = read_config(config_file)
    save_medium_file(medium, output_file)

if __name__ == "__main__":
    main()