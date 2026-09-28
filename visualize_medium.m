%% 读取 medium_map.bin 并可视化介质分布 (内存优化版：逐切片处理)
clear; clc; close all;

filename = 'medium_map.bin';
fid = fopen(filename, 'r');
if fid == -1
    error('无法打开文件: %s', filename);
end

% 读取维度（前12字节：3个int32）
nx = fread(fid, 1, 'int32');
ny = fread(fid, 1, 'int32');
nz = fread(fid, 1, 'int32');

fprintf('网格尺寸: nx=%d, ny=%d, nz=%d (总计 %d 格点)\n', nx, ny, nz, nx*ny*nz);

% 读取完整介质数据（480M 体素 ≈ 1.92GB，单份存储不复制）
% Python 写入顺序：for i in 0:nx-1, for j in 0:ny-1, write medium[i,j,:]
% 即 x 最慢, y 次之, z 最快（Fortran-like order）
medium = zeros(nx, ny, nz, 'int32');
for i = 1:nx
    for j = 1:ny
        medium(i, j, :) = fread(fid, nz, 'int32');
    end
end
fclose(fid);

%% 统计信息（逐切片累加，避免 medium(:) 复制导致 OOM）
fprintf('统计材料分布...\n');
mat_set = [];           % 出现的材料编号集合
material_counts = [];   % 对应的总计数

for i = 1:nx
    slice = medium(i, :, :);
    [uv, ~, ic] = unique(slice(:));
    cnt = accumarray(ic, 1);
    for u = 1:length(uv)
        mat = uv(u);
        idx = find(mat_set == mat, 1);
        if isempty(idx)
            mat_set(end+1) = mat;
            material_counts(end+1) = cnt(u);
        else
            material_counts(idx) = material_counts(idx) + cnt(u);
        end
    end
end

unique_materials = mat_set;
fprintf('材料分布:\n');
for m = unique_materials
    count = material_counts(mat_set == m);
    if m == 0
        fprintf('  Vacuum (0):    %d voxels (%.2f%%)\n', count, count/(nx*ny*nz)*100);
    else
        fprintf('  Material %d:     %d voxels (%.2f%%)\n', m, count, count/(nx*ny*nz)*100);
    end
end

%% 读取网格间距（从 simulation.cfg）
config_file = 'simulation.cfg';
dx = 5e-9; dy = 10e-9; dz = 10e-9;  % 默认值
if exist(config_file, 'file')
    fid_cfg = fopen(config_file, 'r');
    if fid_cfg ~= -1
        while ~feof(fid_cfg)
            line = fgetl(fid_cfg);
            if ischar(line)
                tokens = regexp(line, 'delta_x\s*=\s*([\d.eE+-]+)', 'tokens');
                if ~isempty(tokens), dx = str2double(tokens{1}{1}); end
                tokens = regexp(line, 'delta_y\s*=\s*([\d.eE+-]+)', 'tokens');
                if ~isempty(tokens), dy = str2double(tokens{1}{1}); end
                tokens = regexp(line, 'delta_z\s*=\s*([\d.eE+-]+)', 'tokens');
                if ~isempty(tokens), dz = str2double(tokens{1}{1}); end
            end
        end
        fclose(fid_cfg);
    end
end
fprintf('网格间距: dx=%.3e m, dy=%.3e m, dz=%.3e m\n', dx, dy, dz);

% 转换为微米/格点
x_um_per_cell = dx * 1e6;  % μm per grid cell in x
y_um_per_cell = dy * 1e6;  % μm per grid cell in y
z_um_per_cell = dz * 1e6;  % μm per grid cell in z

%% 可视化
% 获取需要显示的材料编号（排除 material 0 即真空）
display_materials = unique_materials(unique_materials > 0);  % 只跳过真空（材料0），材料1正常显示

if isempty(display_materials)
    fprintf('没有需要显示的材料（排除真空 material 0 后无数据）\n');
    return;
end

fprintf('\n将显示的材料: %s\n', mat2str(display_materials'));

% 颜色映射（为每种材料分配不同颜色）
cmap = lines(length(display_materials));

% 根据网格大小自动计算合适的点大小
max_dim = max([nx, ny, nz]);
marker_size = max(8, min(50, round(600 / max_dim)));
fprintf('自动点大小: %d (最大维度=%d)\n', marker_size, max_dim);

%% 降采样设置
target_points = 200000;  % 默认降采样到 20 万点

% 统计所有需要显示的材料总格点数（使用逐切片累加的结果，不再调用 sum(medium(:)==m)）
total_display = 0;
for idx = 1:length(display_materials)
    mat_id = display_materials(idx);
    total_display = total_display + material_counts(mat_set == mat_id);
end

% 计算全局采样步长（均匀采样，保证空间均匀性）
if total_display > target_points
    sample_step = max(1, round(total_display / target_points));
    fprintf('总显示格点 %d > 目标 %d，采样步长: %d (约 %.0f 万点)\n', ...
        total_display, target_points, sample_step, total_display/sample_step/10000);
else
    sample_step = 1;
    fprintf('总格点 %d ≤ 目标 %d，不降采样\n', total_display, target_points);
end

%% 准备绘图数据（逐切片采样，避免 find(medium==mat_id) 复制整个数组）
fprintf('采样绘图数据...\n');
plot_data = cell(length(display_materials), 5);  % {xi, yi, zi, mat_id, n_plotted}

% 为每种材料维护一个计数器，用于均匀采样
mat_counters = zeros(1, length(display_materials));
mat_xi = cell(1, length(display_materials));  % 预分配，但动态增长
mat_yi = cell(1, length(display_materials));
mat_zi = cell(1, length(display_materials));

for i = 1:nx
    slice = medium(i, :, :);  % [1, ny, nz]
    for idx = 1:length(display_materials)
        mat_id = display_materials(idx);
        n_total = material_counts(mat_set == mat_id);
        
        % 在切片中找该材料的格点索引
        [yi_slice, zi_slice] = find(squeeze(slice) == mat_id);
        
        if ~isempty(yi_slice)
            xi_slice = repmat(i, length(yi_slice), 1);
            
            % 均匀采样：通过全局采样步长
            if sample_step > 1
                % 计算这些点在全局中的位置
                for p = 1:length(yi_slice)
                    mat_counters(idx) = mat_counters(idx) + 1;
                    if mod(mat_counters(idx) - 1, sample_step) == 0
                        mat_xi{idx}(end+1) = xi_slice(p);
                        mat_yi{idx}(end+1) = yi_slice(p);
                        mat_zi{idx}(end+1) = zi_slice(p);
                    end
                end
            else
                % 不降采样，全部保留
                mat_xi{idx} = [mat_xi{idx}; xi_slice];
                mat_yi{idx} = [mat_yi{idx}; yi_slice];
                mat_zi{idx} = [mat_zi{idx}; zi_slice];
            end
        end
    end
    if mod(i, 50) == 0
        fprintf('  采样进度: %d/%d x-slices (%.0f%%)\n', i, nx, 100*i/nx);
    end
end

for idx = 1:length(display_materials)
    mat_id = display_materials(idx);
    n_total = material_counts(mat_set == mat_id);
    plot_data{idx, 1} = mat_xi{idx};
    plot_data{idx, 2} = mat_yi{idx};
    plot_data{idx, 3} = mat_zi{idx};
    plot_data{idx, 4} = mat_id;
    plot_data{idx, 5} = length(mat_xi{idx});
    
    fprintf('  Material %d: %d 个格点待绘制 (%.1f%%)\n', mat_id, length(mat_xi{idx}), ...
        length(mat_xi{idx})/n_total*100);
end

% 清理大数组，释放内存
clear medium slice mat_xi mat_yi mat_zi mat_counters;

% ======================== 图1：格子索引坐标 ========================
figure('Position', [50 100 900 700]);
hold on; grid on; box on;
xlabel('x (grid index)'); ylabel('y (grid index)'); zlabel('z (grid index)');
title(sprintf('Medium Map — Grid Index  (%d × %d × %d)', nx, ny, nz));
view(3);
axis equal;
xlim([0 nx+1]); ylim([0 ny+1]); zlim([0 nz+1]);

leg_handles = gobjects(0);
leg_entries = {};

for idx = 1:length(display_materials)
    xi = plot_data{idx, 1};
    yi = plot_data{idx, 2};
    zi = plot_data{idx, 3};
    mat_id = plot_data{idx, 4};
    n_plotted = plot_data{idx, 5};
    
    h = scatter3(xi, yi, zi, marker_size, ...
        'filled', ...
        'MarkerFaceColor', cmap(idx,:), ...
        'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.7);
    
    leg_handles(idx) = h;
    leg_entries{idx} = sprintf('Material %d  (%d voxels)', mat_id, n_plotted);
end

if ~isempty(leg_handles)
    legend(leg_handles, leg_entries, 'Location', 'best');
end
hold off;
fprintf('图1（格子索引）绘制完成。\n');

% ======================== 图2：微米坐标 ========================
figure('Position', [1000 100 900 700]);
hold on; grid on; box on;
xlabel('x (μm)'); ylabel('y (μm)'); zlabel('z (μm)');
title(sprintf('Medium Map — Physical Units  (%d × %d × %d, dx=%.1f nm, dy=%.1f nm, dz=%.1f nm)', ...
    nx, ny, nz, dx*1e9, dy*1e9, dz*1e9));
view(3);
axis equal;
xlim([0 nx*x_um_per_cell]); ylim([0 ny*y_um_per_cell]); zlim([0 nz*z_um_per_cell]);

leg_handles2 = gobjects(0);
leg_entries2 = {};

for idx = 1:length(display_materials)
    xi_um = plot_data{idx, 1} * x_um_per_cell;
    yi_um = plot_data{idx, 2} * y_um_per_cell;
    zi_um = plot_data{idx, 3} * z_um_per_cell;
    mat_id = plot_data{idx, 4};
    n_plotted = plot_data{idx, 5};
    
    h = scatter3(xi_um, yi_um, zi_um, marker_size, ...
        'filled', ...
        'MarkerFaceColor', cmap(idx,:), ...
        'MarkerEdgeColor', 'none', ...
        'MarkerFaceAlpha', 0.7);
    
    leg_handles2(idx) = h;
    leg_entries2{idx} = sprintf('Material %d  (%d voxels)', mat_id, n_plotted);
end

if ~isempty(leg_handles2)
    legend(leg_handles2, leg_entries2, 'Location', 'best');
end
hold off;
fprintf('图2（微米坐标）绘制完成。\n');

fprintf('\n可视化完成。\n');
