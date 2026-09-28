%% 读取 simulation.cfg 并可视化几何形状与源点
clear; clc; close all;

filename = 'simulation.cfg';
fid = fopen(filename, 'r');
if fid == -1
    error('无法打开文件: %s', filename);
end

% 初始化存储结构
sources = [];
cubes = [];
hexahedra = [];
cylinders = [];
material_ids = [];

% 网格间距（默认 1.0，向后兼容旧配置）
dx = 1.0; dy = 1.0; dz = 1.0;

line = fgetl(fid);
while ischar(line)
    % 去除首尾空格
    line = strtrim(line);
    
    % 跳过空行和注释行
    if isempty(line) || startsWith(line, '#')
        line = fgetl(fid);
        continue;
    end
    
    % 读取网格间距（物理单位，如米）
    tokens = regexp(line, 'delta_x\s*=\s*([\d.eE+-]+)', 'tokens');
    if ~isempty(tokens), dx = str2double(tokens{1}{1}); end
    tokens = regexp(line, 'delta_y\s*=\s*([\d.eE+-]+)', 'tokens');
    if ~isempty(tokens), dy = str2double(tokens{1}{1}); end
    tokens = regexp(line, 'delta_z\s*=\s*([\d.eE+-]+)', 'tokens');
    if ~isempty(tokens), dz = str2double(tokens{1}{1}); end
    
    % 处理 source 行
    if startsWith(line, 'source')
        % 提取 x,y,z
        tokens = regexp(line, 'x=(\d+),y=(\d+),z=(\d+)', 'tokens');
        if ~isempty(tokens)
            src = str2double(tokens{1});
            sources = [sources; src];
        end
    end
    
    % 处理 cubes 定义（可能跨多行）
    if contains(line, 'cubes = [')
        cubes_str = '';
        % 累积直到找到闭合的 ']'（简单处理：遇到单独的 ']' 行或包含 ']' 的行）
        while ~contains(line, ']') || ~endsWith(strtrim(line), ']')
            cubes_str = [cubes_str, line]; %#ok<AGROW>
            line = fgetl(fid);
            if ~ischar(line), break; end
            line = strtrim(line);
        end
        cubes_str = [cubes_str, line]; % 加上最后一行
        % 移除注释（#后面的内容）
        cubes_str = regexprep(cubes_str, '#.*', '');
        % 提取每个立方体的花括号内容
        pattern = '\{[^{}]*\}';
        matches = regexp(cubes_str, pattern, 'match');
        for i = 1:length(matches)
            cubestr = matches{i};
            % 提取 xrange, yrange, zrange, material
            xr = regexp(cubestr, 'xrange":\s*\((\d+),\s*(\d+)\)', 'tokens');
            yr = regexp(cubestr, 'yrange":\s*\((\d+),\s*(\d+)\)', 'tokens');
            zr = regexp(cubestr, 'zrange":\s*\((\d+),\s*(\d+)\)', 'tokens');
            mat = regexp(cubestr, 'material":\s*(\d+)', 'tokens');
            if ~isempty(xr) && ~isempty(yr) && ~isempty(zr) && ~isempty(mat)
                xmin = str2double(xr{1}{1}); xmax = str2double(xr{1}{2});
                ymin = str2double(yr{1}{1}); ymax = str2double(yr{1}{2});
                zmin = str2double(zr{1}{1}); zmax = str2double(zr{1}{2});
                matid = str2double(mat{1}{1});
                cubes = [cubes; struct('xmin',xmin,'xmax',xmax,...
                    'ymin',ymin,'ymax',ymax,...
                    'zmin',zmin,'zmax',zmax,'material',matid)]; %#ok<AGROW>
                material_ids = [material_ids, matid]; %#ok<AGROW>
            end
        end
    end
    
    % 处理 hexahedra 定义
if contains(line, 'hexahedra = [')
    hex_str = '';
    while true
        % 移除当前行中的注释（# 及其后面的内容）
        line_clean = regexprep(line, '#.*', '');
        hex_str = [hex_str, line_clean, newline];   % 保留换行便于后续处理
        
        if contains(line, ']')
            break;
        end
        line = fgetl(fid);
        if ~ischar(line), break; end
        line = strtrim(line);
    end
    hex_str = strtrim(hex_str);
    
    % ---------- 新的解析方式 ----------
    % 定位 "vertices": 后面的数组起始 '[' 和结束 ']'
    start_idx = strfind(hex_str, '"vertices":');
    if ~isempty(start_idx)
        % 从 "vertices": 之后找到第一个 '['
        bracket_start = strfind(hex_str(start_idx(1):end), '[');
        if ~isempty(bracket_start)
            actual_start = start_idx(1) + bracket_start(1) - 1;
            % 找到与之配对的 ']'（简单计数，假设括号正确匹配）
            depth = 1;
            pos = actual_start;
            while depth > 0 && pos < length(hex_str)
                pos = pos + 1;
                if hex_str(pos) == '['
                    depth = depth + 1;
                elseif hex_str(pos) == ']'
                    depth = depth - 1;
                end
            end
            if depth == 0
                % 提取 vertices 数组内容（包括外层的 []）
                verts_str = hex_str(actual_start:pos);
                % 用正则提取所有括号内的三元组
                triples = regexp(verts_str, '\((\d+),\s*(\d+),\s*(\d+)\)', 'tokens');
                if length(triples) >= 8
                    vertices = zeros(8,3);
                    for j = 1:8
                        vertices(j,:) = str2double(triples{j});
                    end
                    % 提取 material
                    mat_match = regexp(hex_str, 'material":\s*(\d+)', 'tokens');
                    if ~isempty(mat_match)
                        matid = str2double(mat_match{1}{1});
                        hexahedra = [hexahedra; struct('vertices',vertices,'material',matid)];
                        material_ids = [material_ids, matid];
                    end
                else
                    warning('顶点数量不足8个，实际找到 %d 个', length(triples));
                end
            end
        end
    end
    % ---------------------------------
end

    % 处理 cylinders 定义
if contains(line, 'cylinders = [')
    cyl_str = '';
    while true
        line_clean = regexprep(line, '#.*', '');
        cyl_str = [cyl_str, line_clean, newline];
        if contains(line, ']')
            break;
        end
        line = fgetl(fid);
        if ~ischar(line), break; end
        line = strtrim(line);
    end
    cyl_str = strtrim(cyl_str);

    % 提取所有 {...} 花括号块
    blocks = regexp(cyl_str, '\{[^{}]*\}', 'match');
    for b = 1:length(blocks)
        blk = blocks{b};
        % 提取 p1
        p1_match = regexp(blk, '"p1":\s*\((\d+),\s*(\d+),\s*(\d+)\)', 'tokens');
        % 提取 p2
        p2_match = regexp(blk, '"p2":\s*\((\d+),\s*(\d+),\s*(\d+)\)', 'tokens');
        % 提取 radius（物理单位，支持科学计数法如 250e-9）
        r_match = regexp(blk, '"radius":\s*([\d.eE+-]+)', 'tokens');
        % 提取 material
        mat_match = regexp(blk, '"material":\s*(\d+)', 'tokens');
        if ~isempty(p1_match) && ~isempty(p2_match) && ~isempty(r_match) && ~isempty(mat_match)
            p1 = str2double(p1_match{1});
            p2 = str2double(p2_match{1});
            radius = str2double(r_match{1}{1});
            matid = str2double(mat_match{1}{1});
            cylinders = [cylinders; struct('p1',p1,'p2',p2,'radius',radius,'material',matid)];
            material_ids = [material_ids, matid];
        end
    end
end

% 处理 material 行（只提取 id）
    if startsWith(line, 'material')
        tokens = regexp(line, 'id=(\d+)', 'tokens');
        if ~isempty(tokens)
            material_ids = [material_ids, str2double(tokens{1}{1})]; %#ok<AGROW>
        end
    end
    
    line = fgetl(fid);
end
fclose(fid);

% 去除重复的 material id
material_ids = unique(material_ids);
% 为每个 material id 分配颜色
cmap = lines(length(material_ids));  % 使用 MATLAB 的 lines 颜色映射
% 创建映射容器
colorMap = containers.Map(material_ids, num2cell(cmap,2));

%% 绘制图形
fprintf('网格间距: dx=%.3e, dy=%.3e, dz=%.3e\n', dx, dy, dz);
figure;
hold on; grid on; box on;
xlabel('x (grid index)'); ylabel('y (grid index)'); zlabel('z (grid index)');
title(sprintf('3D Visualization (dx=%.1e, dy=%.1e, dz=%.1e; 圆柱体截面在物理空间为正圆)', dx, dy, dz));
view(3);
axis equal;

% 绘制立方体
for i = 1:length(cubes)
    c = cubes(i);
    % 立方体的8个顶点
    x = [c.xmin c.xmax];
    y = [c.ymin c.ymax];
    z = [c.zmin c.zmax];
    % 生成顶点矩阵（8x3）
    [X, Y, Z] = meshgrid(x, y, z);
    vertices = [X(:) Y(:) Z(:)];
    % 定义每个面的顶点顺序（确保法向向外）
    % 顶点索引顺序：
    % 1: (xmin,ymin,zmin)
    % 2: (xmax,ymin,zmin)
    % 3: (xmin,ymax,zmin)
    % 4: (xmax,ymax,zmin)
    % 5: (xmin,ymin,zmax)
    % 6: (xmax,ymin,zmax)
    % 7: (xmin,ymax,zmax)
    % 8: (xmax,ymax,zmax)
    faces = [
        1 3 4 2;   % z = zmin 面
        5 6 8 7;   % z = zmax 面
        1 2 6 5;   % y = ymin 面
        2 4 8 6;   % x = xmax 面
        4 3 7 8;   % y = ymax 面
        3 1 5 7    % x = xmin 面
        ];
    patch('Faces', faces, 'Vertices', vertices, ...
        'FaceColor', colorMap(c.material), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    patch('Faces', faces, 'Vertices', vertices, ...
        'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 1);
end

% 绘制六面体
for i = 1:length(hexahedra)
    h = hexahedra(i);
    verts = h.vertices;  % 8x3 矩阵，顺序按照定义
    % 假设顺序为：底面：1-2-3-4 (v0,v1,v2,v3)；顶面：5-6-7-8 (v4,v5,v6,v7)
    % 侧面：1-2-6-5, 2-3-7-6, 3-4-8-7, 4-1-5-8
    faces = [
        1 2 3 4;   % 底面
        5 8 7 6;   % 顶面 (注意顺序以保持法向向外)
        1 2 6 5;   % 侧面
        2 3 7 6;
        3 4 8 7;
        4 1 5 8
        ];
    patch('Faces', faces, 'Vertices', verts, ...
        'FaceColor', colorMap(h.material), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
end

% 绘制圆柱体（含两端圆盘，闭合曲面） —— 各向异性网格：物理空间计算，再缩放回格子索引
for i = 1:length(cylinders)
    c = cylinders(i);
    p1 = c.p1(:); p2 = c.p2(:);        % 格子索引
    scale = [dx; dy; dz];
    
    % 转换到物理空间
    p1_phys = p1 .* scale;
    p2_phys = p2 .* scale;
    v_phys = p2_phys - p1_phys;
    height_phys = norm(v_phys);
    if height_phys < 1e-12, continue; end
    v_hat = v_phys / height_phys;

    % 生成单位圆柱侧面（沿z轴, 半径=c.radius 物理单位, 高度1）
    n_div = 30;
    [X, Y, Z] = cylinder(c.radius, n_div);
    Z = Z * height_phys;

    % 旋转: 将 [0;0;1] 对齐到 v_hat
    z_axis = [0; 0; 1];
    if norm(cross(z_axis, v_hat)) < 1e-12
        if v_hat(3) < 0, Z = -Z; end
        R = eye(3);
    else
        rot_axis = cross(z_axis, v_hat);
        rot_axis = rot_axis / norm(rot_axis);
        theta = acos(dot(z_axis, v_hat));
        K = [0, -rot_axis(3), rot_axis(2);
             rot_axis(3), 0, -rot_axis(1);
             -rot_axis(2), rot_axis(1), 0];
        R = eye(3) + sin(theta)*K + (1-cos(theta))*(K*K);
    end

    % 旋转+平移所有侧面顶点（物理空间），然后缩放回格子空间
    [nr, nc] = size(X);          % nr=2（顶底两圈）, nc=n_div+1（闭合圈）
    N_side = nr * nc;
    verts = zeros(N_side + 2, 3);  % +2：两个圆盘中心点
    for r = 1:nr
        for col = 1:nc
            pt_phys = R * [X(r,col); Y(r,col); Z(r,col)] + p1_phys;
            pt_grid = pt_phys ./ scale;        % 物理→格子索引
            verts((r-1)*nc + col, :) = pt_grid';
        end
    end
    % 两端圆心（格子索引）
    verts(N_side + 1, :) = p1';
    verts(N_side + 2, :) = p2';

    % --- 构建面 ---
    % 侧面四边形条带
    n_side_faces = (nr-1) * (nc-1);
    faces = zeros(n_side_faces + 2*nc, 3);  % 三角形面（侧面可拆为2个△/格，但直接用quad也可；这里统一用tri方便合并）
    fi = 1;
    % 侧面：每个四边形拆成两个三角形
    for r = 1:nr-1
        for col = 1:nc-1
            v1 = (r-1)*nc + col;
            v2 = (r-1)*nc + col + 1;
            v3 = r*nc + col;
            v4 = r*nc + col + 1;
            faces(fi,:) = [v1, v2, v4]; fi = fi + 1;
            faces(fi,:) = [v1, v4, v3]; fi = fi + 1;
        end
    end
    % p1端圆盘（triangular fan，顶点1:nc → 中心 N_side+1）
    center1 = N_side + 1;
    for k = 1:nc-1
        faces(fi,:) = [k, k+1, center1]; fi = fi + 1;
    end
    % p2端圆盘（triangular fan，顶点 (nr-1)*nc+1 : N_side → 中心 N_side+2）
    center2 = N_side + 2;
    base2 = (nr-1)*nc;
    for k = 1:nc-1
        faces(fi,:) = [base2 + k + 1, base2 + k, center2]; fi = fi + 1;
    end

    patch('Faces', faces(1:fi-1,:), 'Vertices', verts, ...
        'FaceColor', colorMap(c.material), 'EdgeColor', 'none', 'FaceAlpha', 0.7);

    % 两端底面边缘加黑线（用 patch 的 XData/YData/ZData 模式）
    patch(verts(1:nc, 1), verts(1:nc, 2), verts(1:nc, 3), 'k', ...
          'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 1);
    idx2 = base2+1 : base2+nc;
    patch(verts(idx2, 1), verts(idx2, 2), verts(idx2, 3), 'k', ...
          'FaceColor', 'none', 'EdgeColor', 'k', 'LineWidth', 1);
end

% 绘制源点
if ~isempty(sources)
    scatter3(sources(:,1), sources(:,2), sources(:,3), 100, 'r', 'filled', 'Marker', 'p');
end

% 添加图例（手动构造）
% 获取所有用到的 material id（从绘制的对象中）
if ~isempty(cubes)
    cube_materials = [cubes.material];
else
    cube_materials = [];
end
if ~isempty(hexahedra)
    hex_materials = [hexahedra.material];
else
    hex_materials = [];
end
if ~isempty(cylinders)
    cyl_materials = [cylinders.material];
else
    cyl_materials = [];
end
used_materials = unique([cube_materials, hex_materials, cyl_materials]);

leg_entries = {};
leg_handles = [];
% 为每个使用的材料创建图例句柄
for id = used_materials
    % 创建一个不可见的 patch 用于图例
    h = patch(NaN, NaN, NaN, 'FaceColor', colorMap(id), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    leg_handles = [leg_handles, h];
    leg_entries{end+1} = sprintf('Material %d', id);
end
% 添加源点图例（如果存在）
if ~isempty(sources)
    h = scatter3(NaN, NaN, NaN, 100, 'r', 'filled', 'Marker', 'p');
    leg_handles = [leg_handles, h];
    leg_entries{end+1} = 'Source';
end
if ~isempty(leg_handles)
    legend(leg_handles, leg_entries, 'Location', 'best');
end

hold off;