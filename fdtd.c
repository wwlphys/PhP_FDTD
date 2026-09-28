#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <mpi.h>
#include <float.h>


/* === Memory tracking === */
static size_t g_total_allocated = 0;
static int g_mpi_rank = 0;          /* set in main(); print only on rank 0 */
static int g_mpi_size = 1;          /* set in main() */
static int g_suppress_print = 0;    /* suppress during groups */
static size_t g_group_saved = 0;    /* total at group start */
#define TRACK_MIN_PRINT  100000000  /* 0.1 GB */

static void track_begin_group(void) {
    g_suppress_print = 1;
    g_group_saved = g_total_allocated;
}

static void track_end_group(const char* label) {
    g_suppress_print = 0;
    size_t delta = g_total_allocated - g_group_saved;
    if (g_mpi_rank == 0 && delta >= TRACK_MIN_PRINT) {
        fprintf(stderr, "[MEM] +%.6f GB (%s), total=%.6f GB (est. all: %.6f GB)\n",
                delta / 1e9, label, g_total_allocated / 1e9,
                g_total_allocated * g_mpi_size / 1e9);
    }
}

static void* tracked_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr) g_total_allocated += size;
    return ptr;
}

static void* tracked_calloc(size_t nmemb, size_t size) {
    void* ptr = calloc(nmemb, size);
    if (ptr) g_total_allocated += nmemb * size;
    return ptr;
}

static void track_realloc_delta(size_t old_bytes, size_t new_bytes) {
    g_total_allocated += (new_bytes - old_bytes);
}

static void print_alloc(size_t bytes, const char* name) {
    if (!g_suppress_print && g_mpi_rank == 0 && bytes >= TRACK_MIN_PRINT) {
        fprintf(stderr, "[MEM] +%.6f GB (%s), total=%.6f GB (est. all: %.6f GB)\n",
                bytes / 1e9, name, g_total_allocated / 1e9,
                g_total_allocated * g_mpi_size / 1e9);
    }
}

static void print_mem_total(const char* label) {
    size_t local = g_total_allocated;
    size_t global = 0;
    MPI_Reduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (g_mpi_rank == 0) {
        fprintf(stderr, "[MEM] %s: per-rank = %.6f GB, total (%d ranks) = %.6f GB\n",
                label, local / 1e9, g_mpi_size, global / 1e9);
    }
}
/* === End memory tracking === */


// 定义介电张量结构体
typedef struct {
    double xx, xy, xz;
    double yx, yy, yz;
    double zx, zy, zz;
	double xxi, xyi, xzi;
	double yxi, yyi, yzi;
	double zxi, zyi, zzi;
} DielectricTensor;

// 材料属性结构体
typedef struct {
    DielectricTensor eps;      // 介电张量
    DielectricTensor eps_inv;  // 介电张量逆矩阵
    double sigma;              // 电导率 (S/m)
    double sigma_m;            // 磁损耗 (Ω/m)
    char name[64];             // 材料名称
	int natom;                 // 原子数量（新增）
	double* born;              // Born有效电荷数组[3*natom][3]
	char born_filename[1024];   // Born有效电荷文件名
	int use_born_file_for_eps; // 新增：1=从born文件读取介电张量
	double* QeZV;
	double* QeZM;
	double* DMM;
	char QeZV_filename[1024];
	char QeZM_filename[1024];
	char DMM_filename[1024];
	int QeZV_filename_set;
	int QeZM_filename_set;
	int DMM_filename_set;
	char dielectric_filename[1024];   // 介电函数文件名
	int use_dielectric_file;         // 是否从文件读取介电函数
	double anglez;				// 材料绕-z轴转过的角度(rad)
	double anglex;				// 材料绕x轴转过的角度(rad)
	int switchxz;				// 0:不动；1：把这个材料的参数中XZ轴调换
	//  ADE 相关
	int use_ADE;				// 介电函数有负数，需要使用辅助微分方程方法
	double epsr1[3][3];			// eps_r - 1
	double epsr1_inv[3][3];		// (eps_r-1)的逆矩阵
	double gama[3][3];
	double wp2[3][3];
	double gdt2p1[3][3], gdt2p1_inv[3][3], gdt2m1[3][3], gdt2m1_inv[3][3];
} Material;

// CPML参数结构体 (Roden & Gedney, 2000)
typedef struct {
    double *sigma;   // 电导率分布 σ
    double *kappa;   // 实坐标伸缩 κ ≥ 1
    double *alpha;   // CFS频移参数 α ≥ 0
} PML_Params;

// 激励源结构体
typedef struct {
    int type;           // 0:高斯脉冲, 1:正弦波, 2:Ricker小波
    int position[3];    // 源位置 (x,y,z)
    double amplitude;   // 幅度
    double frequency;   // 频率 (Hz)
    double t0;          // 脉冲中心时间 (s)
    double tau;         // 脉冲宽度 (s)
    char component;     // 场分量 ('x','y','z')
    double start_time;  // 起始时间 (s)
    double end_time;    // 结束时间 (s)
} Source;

// 仿真配置结构体
typedef struct {
    int nx, ny, nz;          // 网格尺寸
    int tsteps;               // 时间步数
    double delta_x;           // x方向空间步长 (m)
    double delta_y;           // y方向空间步长 (m)
    double delta_z;           // z方向空间步长 (m)
    double dt;                // 时间步长 (s)
    double c;                 // 光速 (m/s)
    double epsilon0;          // 真空介电常数
    double mu0;               // 真空磁导率
    int pml_thickness;        // PML层厚度
	double pml_reflection;    // PML反射率 
	int output_interval;      // 输出间隔步数
    char medium_file[1024];    // 介质分布文件路径
    Source *sources;          // 激励源数组
    int num_sources;          // 激励源数量
	Material* materials;       // 材料数组
	int num_materials;         // 材料数量
	char output_component[10];   // 输出场分量："Ex", "Ey", "Ez", "E"
	// 输出切片配置 - 新增：支持不同方向的切片
	char output_slice_axis;     // 切片轴：'x', 'y', 'z'
	int output_slice_position;  // 切片位置
	// 监测点配置
	int monitor_enabled;      // 是否启用监测点
	int monitor_x;            // 监测点x坐标
	int monitor_y;            // 监测点y坐标
	int monitor_z;            // 监测点z坐标
	int monitor_atom;         // 要监测的原子序号
	int monitor_dir;          // 要监测的方向 (0=x,1=y,2=z)
	char monitor_output_file[1024]; // 输出文件名
	int full_monitor_enabled;      // 是否启用全量monitor（所有场分量+所有原子）
	char full_monitor_output_file[1024]; // 全量monitor输出文件名
	double unique_source_frequency; // 唯一频率
} Config;

typedef struct {
    double*** ptr_array;
    double* data;
} Array3D;

// 四维数组结构
typedef struct {
	double**** ptr_array;
	double* data;
} Array4D;

void trim(char* str) {
	int i = 0;
	int j = 0;
	// 跳过首部空格
	while (isspace((unsigned char)str[i])) {
		i++;
	}
	// 复制到开头
	while (str[i] != '\0') {
		str[j++] = str[i++];
	}
	str[j] = '\0';
	// 去除尾部空格
	j = strlen(str) - 1;
	while (j >= 0 && isspace((unsigned char)str[j])) {
		str[j--] = '\0';
	}
}

#define N 3
#define EPS_INF 20.0   // Drude 模型中的高频介电函数

// 计算矩阵的行列式
double determinant(double matrix[N][N]) {
	double det = 0;

	det = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
		- matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
		+ matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

	return det;
}

// 计算伴随矩阵（余子式矩阵的转置）
void adjoint(double matrix[N][N], double adj[N][N]) {
	// 计算每个元素的余子式
	adj[0][0] = (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]);
	adj[0][1] = -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]);
	adj[0][2] = (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

	adj[1][0] = -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]);
	adj[1][1] = (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]);
	adj[1][2] = -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]);

	adj[2][0] = (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]);
	adj[2][1] = -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]);
	adj[2][2] = (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
}

// 求矩阵的逆
// 返回值：0表示成功，-1表示矩阵不可逆
int inverse(double matrix[N][N], double inv[N][N]) {
	double det = determinant(matrix);
	/*for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("matrix[%d][%d]=%g\n", i, j, matrix[i][j]);
		}
	}
	printf("det=%g\n", det);*/
	// 检查矩阵是否可逆
	if (fabs(det) < 1e-12) {
		printf("矩阵不可逆（行列式为0）%g\n",det);
		return -1;
	}

	double adj[N][N];
	adjoint(matrix, adj);

	// 计算逆矩阵：伴随矩阵除以行列式
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			inv[i][j] = adj[j][i] / det;  // 注意这里转置了伴随矩阵
		}
	}

	return 0;
}

// 三维数组内存分配函数 (带halo) - 修复版本
Array3D malloc_3d_halo(int x, int y, int z, int halo, const char* name) {
    Array3D arr;
    size_t sz_ptr = x * sizeof(double**);
    size_t sz_data = (size_t)x * y * (z + 2*halo) * sizeof(double);
    size_t sz_inner = (size_t)x * y * sizeof(double*);
    size_t total = sz_ptr + sz_data + sz_inner;
    
    arr.ptr_array = (double***)tracked_malloc(sz_ptr);
    arr.data = (double*)tracked_calloc((size_t)x * y * (z + 2*halo), sizeof(double));
    
    for (int i = 0; i < x; i++) {
        arr.ptr_array[i] = (double**)tracked_malloc(y * sizeof(double*));
        for (int j = 0; j < y; j++) {
            arr.ptr_array[i][j] = arr.data + (i * y * (z + 2*halo)) + j * (z + 2*halo) + halo;
        }
    }
    
    if (!g_suppress_print && g_mpi_rank == 0 && total >= TRACK_MIN_PRINT) {
        fprintf(stderr, "[MEM] +%.6f GB (%s), total=%.6f GB (est. all: %.6f GB)\n",
                total / 1e9, name ? name : "?", g_total_allocated / 1e9,
                g_total_allocated * g_mpi_size / 1e9);
    }
    return arr;
}

int*** malloc_3d_int_halo(int x, int y, int z, int halo, const char* name) {
    size_t total = x * sizeof(int**) + (size_t)x * y * sizeof(int*)
                 + (size_t)x * y * (z + 2*halo) * sizeof(int);
    int ***array = (int***)tracked_malloc(x * sizeof(int**));
    if (!array) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < x; i++) {
        array[i] = (int**)tracked_malloc(y * sizeof(int*));
        if (!array[i]) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        
        for (int j = 0; j < y; j++) {
            array[i][j] = (int*)tracked_calloc(z + 2*halo, sizeof(int));
            if (!array[i][j]) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    
    if (!g_suppress_print && g_mpi_rank == 0 && total >= TRACK_MIN_PRINT) {
        fprintf(stderr, "[MEM] +%.6f GB (%s), total=%.6f GB (est. all: %.6f GB)\n",
                total / 1e9, name ? name : "?", g_total_allocated / 1e9,
                g_total_allocated * g_mpi_size / 1e9);
    }
    return array;
}

// 三维数组内存释放函数
void free_3d(Array3D arr, int x, int y) {
    if (arr.ptr_array) {
        for (int i = 0; i < x; i++) {
            if (arr.ptr_array[i]) free(arr.ptr_array[i]);
        }
        free(arr.ptr_array);
    }
    if (arr.data) free(arr.data);
}

void free_3d_int(int*** array, int x, int y) {
    if (array) {
        for (int i = 0; i < x; i++) {
            if (array[i]) {
                for (int j = 0; j < y; j++) {
                    free(array[i][j]); // 释放第三维
                }
                free(array[i]); // 释放第二维
            }
        }
        free(array); // 释放第一维
    }
}

// 从文本文件读数据
// 添加函数来读取介电函数文件并找到最接近频率的数据
int read_dielectric_from_file(const char* filename, double frequency, DielectricTensor* eps) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Error opening dielectric file: %s\n", filename);
		return 0;
	}

	double best_freq_diff = DBL_MAX;
	DielectricTensor best_eps;
	char line[256];
	int found = 0;

	while (fgets(line, sizeof(line), file)) {
		// 跳过注释行和空行
		if (line[0] == '#' || line[0] == '\n') continue;

		double file_freq, file_wavelength, file_wavenumber, eps_xx, eps_xy, eps_xz, eps_yy, eps_yz, eps_zz,
			eps_xxi, eps_xyi, eps_xzi, eps_yyi, eps_yzi, eps_zzi;

		if (sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
			&file_freq, &file_wavelength, &file_wavenumber, &eps_xx, &eps_xy, &eps_xz, &eps_yy, &eps_yz, &eps_zz,
			&eps_xxi, &eps_xyi, &eps_xzi, &eps_yyi, &eps_yzi, &eps_zzi) == 15) {

			double freq_diff = fabs(file_freq - frequency);
			if (freq_diff < best_freq_diff) {
				best_freq_diff = freq_diff;

				// 设置介电张量（注意对称性）
				best_eps.xx = eps_xx; best_eps.xy = eps_xy; best_eps.xz = eps_xz;
				best_eps.yx = eps_xy; best_eps.yy = eps_yy; best_eps.yz = eps_yz; // 对称性: yx = xy
				best_eps.zx = eps_xz; best_eps.zy = eps_yz; best_eps.zz = eps_zz; // 对称性: zx = xz, zy = yz

				best_eps.xxi = eps_xxi; best_eps.xyi = eps_xyi; best_eps.xzi = eps_xzi;
				best_eps.yxi = eps_xyi; best_eps.yyi = eps_yyi; best_eps.yzi = eps_yzi; // 对称性
				best_eps.zxi = eps_xzi; best_eps.zyi = eps_yzi; best_eps.zzi = eps_zzi; // 对称性

				found = 1;
			}
		}
	}

	fclose(file);

	if (found) {
		*eps = best_eps;
		if (best_freq_diff > 0.1 * frequency) {
			fprintf(stderr, "Warning: No exact frequency match in %s. Using closest frequency (diff: %.2e Hz)\n",
				filename, best_freq_diff);
		}
		return 1;
	}
	else {
		fprintf(stderr, "Error: No valid data found in dielectric file: %s\n", filename);
		return 0;
	}
}

int read_data(const char* filename, double* data, int n) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		perror("文件打开失败");
		exit(EXIT_FAILURE);
	}
	printf("%s %d\n", filename,n);
	for (int i = 0; i < n; i++) {
		if (fscanf(file, "%lf", &data[i]) != 1) {
			fprintf(stderr, "错误：读取数据失败或数据不足\n");
			fclose(file);
			exit(EXIT_FAILURE);
		}
	}

	fclose(file);
	return 1;
}

// 从文本文件读取Born有效电荷
int read_born_from_file(const char* filename, double* born, int natom,
	DielectricTensor* eps, int use_born_file_for_eps) {
	FILE* file = fopen(filename, "r");
	if (file == NULL) {
		fprintf(stderr, "Error opening Born file: %s\n", filename);
		return 0;
	}

	char line[512];
	int values_read = 0;
	double values[1000]; // 足够大的缓冲区
	int total_required = (use_born_file_for_eps ? 9 : 0) + 9 * natom;

	// 状态标志
	int in_dielectric = 0;
	int in_born = 0;
	int current_ion = -1;
	int direction = 0;
	int dielectric_values_read = 0;
	int born_values_read = 0;

	// 读取文件
	while (fgets(line, sizeof(line), file)) {
		// 移除换行符
		line[strcspn(line, "\n")] = 0;

		// 跳过空行
		if (strlen(line) == 0) continue;

		// 检测介电张量节开始
		if (strstr(line, "MACROSCOPIC STATIC DIELECTRIC TENSOR")) {
			in_dielectric = 1;
			dielectric_values_read = 0;
			continue;
		}

		// 检测Born有效电荷节开始
		if (strstr(line, "BORN EFFECTIVE CHARGES")) {
			in_born = 1;
			born_values_read = 0;
			continue;
		}

		// 检测节结束（任何包含多个'-'的行）
		if ((in_dielectric || in_born) && strstr(line, "---")) {
			if (in_dielectric) {
				//in_dielectric = 0;
			}
			if (in_born) {
				//in_born = 0;
				current_ion = -1;
			}
			continue;
		}

		// 解析介电张量数据
		if (in_dielectric && dielectric_values_read < 9) {
			double a, b, c;
			// 处理可选的引导空格和行号
			if (sscanf(line, "%*s %lf %lf %lf", &a, &b, &c) == 3 ||
				sscanf(line, "%lf %lf %lf", &a, &b, &c) == 3) {
				values[values_read++] = a;
				values[values_read++] = b;
				values[values_read++] = c;
				dielectric_values_read += 3;
			}
			continue;
		}

		// 解析Born有效电荷数据
		if (in_born) {
			// 检测离子行: "ion X" (处理可变空格)
			if (strstr(line, "ion")) {
				char* num_start = strstr(line, "ion");
				if (num_start) {
					num_start += 3; // 跳过"ion"
					// 跳过任何空格
					while (*num_start && isspace((unsigned char)* num_start)) num_start++;
					current_ion = atoi(num_start);
					direction = 0;
				}
				continue;
			}

			// 解析数据行 (格式: "1 6.51576 -0.00000 -0.28531")
			int idx;
			double a, b, c;
			if (sscanf(line, "%d %lf %lf %lf", &idx, &a, &b, &c) == 4) {
				if (current_ion >= 1 && current_ion <= natom && direction < 3) {
					// 存储为 [atom][direction][field component]
					values[values_read++] = a;
					values[values_read++] = b;
					values[values_read++] = c;
					born_values_read += 3;
					direction++;
				}
			}
			continue;
		}
	}
	fclose(file);

	// 检查读取数量
	if (values_read < total_required) {
		fprintf(stderr, "Error: Only %d values read (expected %d) from %s\n",
			values_read, total_required, filename);
		fprintf(stderr, "Dielectric values read: %d, Born values read: %d\n",
			dielectric_values_read, born_values_read);
		return 0;
	}

	// 解析介电张量
	if (use_born_file_for_eps) {
		int start_idx = 0;
		eps->xx = values[start_idx++]; eps->xy = values[start_idx++]; eps->xz = values[start_idx++];
		eps->yx = values[start_idx++]; eps->yy = values[start_idx++]; eps->yz = values[start_idx++];
		eps->zx = values[start_idx++]; eps->zy = values[start_idx++]; eps->zz = values[start_idx++];
		eps->xxi = 0.0; eps->xyi = 0.0; eps->xzi = 0.0;
		eps->yxi = 0.0; eps->yyi = 0.0; eps->yzi = 0.0;
		eps->zxi = 0.0; eps->zyi = 0.0; eps->zzi = 0.0;
	}

	// 解析Born有效电荷
	int start_idx = use_born_file_for_eps ? 9 : 0;
	for (int i = 0; i < 9 * natom; i++) {
		born[i] = values[start_idx + i];
	}

	return 1;
}

// 计算介电张量的逆矩阵（考虑实部和虚部）
DielectricTensor inverse_dielectric_tensor(DielectricTensor eps) {
	// 计算实部矩阵的行列式
	double det_real = eps.xx * (eps.yy * eps.zz - eps.yz * eps.zy)
		- eps.xy * (eps.yx * eps.zz - eps.yz * eps.zx)
		+ eps.xz * (eps.yx * eps.zy - eps.yy * eps.zx);

	// 计算虚部矩阵的行列式
	double det_imag = eps.xxi * (eps.yyi * eps.zzi - eps.yzi * eps.zyi)
		- eps.xyi * (eps.yxi * eps.zzi - eps.yzi * eps.zxi)
		+ eps.xzi * (eps.yxi * eps.zyi - eps.yyi * eps.zxi);

	// 计算复行列式的模
	double det_magnitude = sqrt(det_real * det_real + det_imag * det_imag);

	if (fabs(det_magnitude) < 1e-12) {
		printf("Error: Singular dielectric tensor!\n");
		exit(1);
	}

	// 计算复行列式的倒数
	double inv_det_real = det_real / (det_real * det_real + det_imag * det_imag);
	double inv_det_imag = -det_imag / (det_real * det_real + det_imag * det_imag);

	DielectricTensor inv;

	// 计算逆矩阵的实部
	inv.xx = inv_det_real * (eps.yy * eps.zz - eps.yz * eps.zy)
		- inv_det_imag * (eps.yyi * eps.zzi - eps.yzi * eps.zyi);
	inv.xy = inv_det_real * (eps.xz * eps.zy - eps.xy * eps.zz)
		- inv_det_imag * (eps.xzi * eps.zyi - eps.xyi * eps.zzi);
	inv.xz = inv_det_real * (eps.xy * eps.yz - eps.xz * eps.yy)
		- inv_det_imag * (eps.xyi * eps.yzi - eps.xzi * eps.yyi);

	inv.yx = inv_det_real * (eps.yz * eps.zx - eps.yx * eps.zz)
		- inv_det_imag * (eps.yzi * eps.zxi - eps.yxi * eps.zzi);
	inv.yy = inv_det_real * (eps.xx * eps.zz - eps.xz * eps.zx)
		- inv_det_imag * (eps.xxi * eps.zzi - eps.xzi * eps.zxi);
	inv.yz = inv_det_real * (eps.xz * eps.yx - eps.xx * eps.yz)
		- inv_det_imag * (eps.xzi * eps.yxi - eps.xxi * eps.yzi);

	inv.zx = inv_det_real * (eps.yx * eps.zy - eps.yy * eps.zx)
		- inv_det_imag * (eps.yxi * eps.zyi - eps.yyi * eps.zxi);
	inv.zy = inv_det_real * (eps.xy * eps.zx - eps.xx * eps.zy)
		- inv_det_imag * (eps.xyi * eps.zxi - eps.xxi * eps.zyi);
	inv.zz = inv_det_real * (eps.xx * eps.yy - eps.xy * eps.yx)
		- inv_det_imag * (eps.xxi * eps.yyi - eps.xyi * eps.yxi);

	// 计算逆矩阵的虚部
	inv.xxi = inv_det_real * (eps.yyi * eps.zzi - eps.yzi * eps.zyi)
		+ inv_det_imag * (eps.yy * eps.zz - eps.yz * eps.zy);
	inv.xyi = inv_det_real * (eps.xzi * eps.zyi - eps.xyi * eps.zzi)
		+ inv_det_imag * (eps.xz * eps.zy - eps.xy * eps.zz);
	inv.xzi = inv_det_real * (eps.xyi * eps.yzi - eps.xzi * eps.yyi)
		+ inv_det_imag * (eps.xy * eps.yz - eps.xz * eps.yy);

	inv.yxi = inv_det_real * (eps.yzi * eps.zxi - eps.yxi * eps.zzi)
		+ inv_det_imag * (eps.yz * eps.zx - eps.yx * eps.zz);
	inv.yyi = inv_det_real * (eps.xxi * eps.zzi - eps.xzi * eps.zxi)
		+ inv_det_imag * (eps.xx * eps.zz - eps.xz * eps.zx);
	inv.yzi = inv_det_real * (eps.xzi * eps.yxi - eps.xxi * eps.yzi)
		+ inv_det_imag * (eps.xz * eps.yx - eps.xx * eps.yz);

	inv.zxi = inv_det_real * (eps.yxi * eps.zyi - eps.yyi * eps.zxi)
		+ inv_det_imag * (eps.yx * eps.zy - eps.yy * eps.zx);
	inv.zyi = inv_det_real * (eps.xyi * eps.zxi - eps.xxi * eps.zyi)
		+ inv_det_imag * (eps.xy * eps.zx - eps.xx * eps.zy);
	inv.zzi = inv_det_real * (eps.xxi * eps.yyi - eps.xyi * eps.yxi)
		+ inv_det_imag * (eps.xx * eps.yy - eps.xy * eps.yx);

	return inv;
}

// 创建材料库
Material* create_material_library(int *num_materials, double frequency) {
    // 定义材料数量
    //*num_materials = 4;
    Material *materials = (Material*)tracked_malloc(*num_materials * sizeof(Material));
    if (!materials) {
        fprintf(stderr, "Error528\n");
        exit(EXIT_FAILURE);
    }
	double omega = frequency / 0.03e12; // wavenumber (cm-1)
    
	// 为每个材料添加默认的natom和born
	for (int i = 0; i < *num_materials; i++) {
		materials[i].natom = 0;
		materials[i].born = NULL;
		materials[i].born_filename[0] = '\0';
		materials[i].use_born_file_for_eps = 0; // 默认不从born文件读取介电张量
		materials[i].use_ADE = 0; // 默认不使用ADE方法
		materials[i].anglez = 0.0;  // 默认不转动晶体
		materials[i].anglex = 0.0;
		materials[i].switchxz = 0; // 默认不调换XZ
		materials[i].use_dielectric_file = 0; // 默认不从dielectric文件读介电张量
		// 初始化 QeZV/QeZM/DMM 指针和文件名标志，避免读取未初始化的malloc垃圾内存
		materials[i].QeZV = NULL;
		materials[i].QeZM = NULL;
		materials[i].DMM = NULL;
		materials[i].QeZV_filename_set = 0;
		materials[i].QeZM_filename_set = 0;
		materials[i].DMM_filename_set = 0;
	}
	
	// 材料0: 真空
    strncpy(materials[0].name, "Vacuum", sizeof(materials[0].name));
    materials[0].eps.xx = 1.0; materials[0].eps.xy = 0.0; materials[0].eps.xz = 0.0;
    materials[0].eps.yx = 0.0; materials[0].eps.yy = 1.0; materials[0].eps.yz = 0.0;
    materials[0].eps.zx = 0.0; materials[0].eps.zy = 0.0; materials[0].eps.zz = 1.0;
	materials[0].eps.xxi = 0.0; materials[0].eps.xyi = 0.0; materials[0].eps.xzi = 0.0;
	materials[0].eps.yxi = 0.0; materials[0].eps.yyi = 0.0; materials[0].eps.yzi = 0.0;
	materials[0].eps.zxi = 0.0; materials[0].eps.zyi = 0.0; materials[0].eps.zzi = 0.0;
	materials[0].eps_inv = inverse_dielectric_tensor(materials[0].eps);
    materials[0].sigma = 0.0;
    materials[0].sigma_m = 0.0;
    
    // 材料1: SiO2 (Drude-Lonrentz Model)
    strncpy(materials[1].name, "SiO2", sizeof(materials[1].name));
	double e0 = 1.5, s1 = 231, g1 = 69, w1 = 806, s2 = 866, g2 = 75, w2 = 1063;
	double temp1 = w1 * w1 - omega * omega, temp2 = w2 * w2 - omega * omega;
	double epsr = e0 + s1 * s1 * temp1 / (temp1 * temp1 + omega * omega * g1 * g1) + s2 * s2 * temp2 / (temp2 * temp2 + omega * omega * g2 * g2);
	double epsi = s1 * s1 * omega * g1 / (temp1 * temp1 + omega * omega * g1 * g1) + s2 * s2 * omega * g2 / (temp2 * temp2 + omega * omega * g2 * g2);
    // 介电函数实部
	materials[1].eps.xx = epsr; materials[1].eps.xy = 0.0; materials[1].eps.xz = 0.0;
    materials[1].eps.yx = 0.0; materials[1].eps.yy = epsr; materials[1].eps.yz = 0.0;
    materials[1].eps.zx = 0.0; materials[1].eps.zy = 0.0; materials[1].eps.zz = epsr;
	// 介电函数虚部
	materials[1].eps.xxi = epsi; materials[1].eps.xyi = 0.0;  materials[1].eps.xzi = 0.0;
	materials[1].eps.yxi = 0.0;  materials[1].eps.yyi = epsi; materials[1].eps.yzi = 0.0;
	materials[1].eps.zxi = 0.0;  materials[1].eps.zyi = 0.0;  materials[1].eps.zzi = epsi;
	materials[1].eps_inv = inverse_dielectric_tensor(materials[1].eps);
    materials[1].sigma = 0.0;
    materials[1].sigma_m = 0.0;
	if (epsr <= 0)	materials[1].use_ADE = 1;
    
    // 材料2: 双轴晶体 (ε_xx = 8, ε_yy = 7, ε_zz = 5)
    strncpy(materials[2].name, "Uniaxial Crystal", sizeof(materials[2].name));
	materials[2].eps.xx = 8.0; materials[2].eps.xy = 0.0; materials[2].eps.xz = 0.0;
	materials[2].eps.yx = 0.0; materials[2].eps.yy = 7.0; materials[2].eps.yz = 0.0;
	materials[2].eps.zx = 0.0; materials[2].eps.zy = 0.0; materials[2].eps.zz = 5.0;
	materials[2].eps.xxi = 0.0; materials[2].eps.xyi = 0.0; materials[2].eps.xzi = 0.0;
	materials[2].eps.yxi = 0.0; materials[2].eps.yyi = 0.0; materials[2].eps.yzi = 0.0;
	materials[2].eps.zxi = 0.0; materials[2].eps.zyi = 0.0; materials[2].eps.zzi = 0.0;
	materials[2].eps_inv = inverse_dielectric_tensor(materials[2].eps);
    materials[2].sigma = 0.0;
    materials[2].sigma_m = 0.0;
    
    // 材料3: 有耗介质 (ε_r = 9, σ = 0.1)
    strncpy(materials[3].name, "Lossy Medium", sizeof(materials[3].name));
	materials[3].eps.xx = 9.0; materials[3].eps.xy = 0.0; materials[3].eps.xz = 0.0;
	materials[3].eps.yx = 0.0; materials[3].eps.yy = 9.0; materials[3].eps.yz = 0.0;
	materials[3].eps.zx = 0.0; materials[3].eps.zy = 0.0; materials[3].eps.zz = 9.0;
	materials[3].eps.xxi = 0.0; materials[3].eps.xyi = 0.0; materials[3].eps.xzi = 0.0;
	materials[3].eps.yxi = 0.0; materials[3].eps.yyi = 0.0; materials[3].eps.yzi = 0.0;
	materials[3].eps.zxi = 0.0; materials[3].eps.zyi = 0.0; materials[3].eps.zzi = 0.0;
	materials[3].eps_inv = inverse_dielectric_tensor(materials[3].eps);
    materials[3].sigma = 0.1;
    materials[3].sigma_m = 0.0;

    
    return materials;
}

// 从二进制文件读取介质分布 (并行版本)
int*** read_medium_map_parallel(const char* filename, int nx, int ny, int nz,
	int local_z_start, int local_nz, int halo, int rank, int size) {
	FILE* file = NULL;
	int*** medium = NULL;

	// 主进程读取文件头
	if (rank == 0) {
		file = fopen(filename, "rb");
		if (file == NULL) {
			perror("Error opening medium file");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		// 读取文件头信息
		int file_nx, file_ny, file_nz;
		if (fread(&file_nx, sizeof(int), 1, file) != 1 ||
			fread(&file_ny, sizeof(int), 1, file) != 1 ||
			fread(&file_nz, sizeof(int), 1, file) != 1) {
			perror("Error reading medium file header");
			fclose(file);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		// 检查尺寸匹配
		if (file_nx != nx || file_ny != ny || file_nz != nz) {
			fprintf(stderr, "Medium file dimensions (%d,%d,%d) do not match simulation (%d,%d,%d)\n",
				file_nx, file_ny, file_nz, nx, ny, nz);
			fclose(file);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	// 广播确认文件头正确
	int header_valid = 1;
	MPI_Bcast(&header_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (!header_valid) {
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	// 计算每个进程需要读取的数据量 - FIXED
	int* recvcounts = NULL;
	int* displs = NULL;

	if (rank == 0) {
		recvcounts = (int*)tracked_malloc(size * sizeof(int));
		displs = (int*)tracked_malloc(size * sizeof(int));
		if (!recvcounts || !displs) {
			fprintf(stderr, "Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// 使用与main函数相同的分解逻辑
		int base_nz = nz / size;
		int remainder = nz % size;
		int offset = 0;
		for (int i = 0; i < size; i++) {
			int chunk_size = base_nz;
			if (i < remainder) {
				chunk_size++;
			}

			recvcounts[i] = chunk_size * ny * nx;
			displs[i] = offset;
			offset += recvcounts[i];
			//printf("displs[%d]=%d   %d  %d\n", i, displs[i],chunk_size,local_nz);
		}
	}

	// 分配本地内存 (包含halo)
	medium = malloc_3d_int_halo(nx, ny, local_nz, halo, "medium");

	// 准备本地缓冲区 (不含halo)
	int* local_buffer = (int*)tracked_malloc(nx * ny * local_nz * sizeof(int));
	print_alloc((size_t)nx * ny * local_nz * sizeof(int), "medium local");
	if (!local_buffer) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	// 主进程读取整个数据
	if (rank == 0) {
		int* global_buffer = (int*)tracked_malloc(nx * ny * nz * sizeof(int));
		print_alloc((size_t)nx * ny * nz * sizeof(int), "medium global");
		if (!global_buffer) {
			fprintf(stderr, "Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}

		// 读取数据
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				if (fread(&global_buffer[i * ny * nz + j * nz], sizeof(int), nz, file) != nz) {
					perror("Error reading medium data");
					fclose(file);
					free(global_buffer);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
		}
		fclose(file);
		/*for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++){
			for (int k = 0; k < nz; k++){
			if (i % 50 == 0 && j % 50 == 0 && k == 240) {
				printf("medium i=%d,j=%d, %d\n",i,j, global_buffer[i* ny* nz + j * nz + k]);
				}
			}
			}
		}*/
		// reshape the data
		int* temp = (int*)tracked_malloc(nx * ny * nz * sizeof(int));
		print_alloc((size_t)nx * ny * nz * sizeof(int), "medium temp");
		if (!temp) {
			fprintf(stderr, "Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++){
				for (int k = 0; k < nz; k++){
					temp[k * nx * ny + i * ny + j] = global_buffer[i * ny * nz + j * nz + k];
				}
			}
		}
		//printf("copy data\n");
		//memcpy(&global_buffer[0], &temp[0], nx* ny* nz * sizeof(int));
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				for (int k = 0; k < nz; k++) {
					global_buffer[k * nx * ny + i * ny + j] = temp[k * nx * ny + i * ny + j];
				}
			}
		}

        // 分发数据到各进程
        MPI_Scatterv(global_buffer, recvcounts, displs, MPI_INT,
                     local_buffer, nx*ny*local_nz, MPI_INT,
                     0, MPI_COMM_WORLD);
        
        free(global_buffer);
		free(temp);
        free(recvcounts);
        free(displs);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
                     local_buffer, nx*ny*local_nz, MPI_INT,
                     0, MPI_COMM_WORLD);
    }
    
    // 将本地缓冲区复制到medium数组 (不含halo)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				medium[i][j][halo + k] = local_buffer[k * nx * ny + i * ny + j];
			}
        }
    }
	/*for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++){
		for (int k = 0; k < local_nz; k++){
			int global_k = k + local_z_start;
			if (i % 10 == 0 && j % 10 == 0 && global_k== 240 && medium[i][j][halo + k] !=0) {
				printf("medium i=%d,j=%d,globak_k=%d, %d\n", i, j, global_k, medium[i][j][halo + k]);
//				printf("medium i=%d,j=%d,globak_k=%d, %d\n", i, j, global_k, local_buffer[k * nx * ny + i * ny + j]);
			}
		}
		}
	}*/
    free(local_buffer);
    
    // 交换halo区域 - 使用批量通信
    MPI_Status status;
    int tag_lower = 0, tag_upper = 1;
    
    // 发送下边界，接收上边界
    if (rank > 0) {
        int *send_buf_lower = (int*)tracked_malloc(nx * ny * sizeof(int));
        if (!send_buf_lower) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                send_buf_lower[i*ny+j] = medium[i][j][halo];
            }
        }
        MPI_Send(send_buf_lower, nx*ny, MPI_INT, rank-1, tag_lower, MPI_COMM_WORLD);
        free(send_buf_lower);
    }
    
    if (rank < size-1) {
        int *recv_buf_upper = (int*)tracked_malloc(nx * ny * sizeof(int));
        if (!recv_buf_upper) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        MPI_Recv(recv_buf_upper, nx*ny, MPI_INT, rank+1, tag_lower, MPI_COMM_WORLD, &status);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                medium[i][j][halo+local_nz] = recv_buf_upper[i*ny+j];
            }
        }
        free(recv_buf_upper);
    }
    
    // 发送上边界，接收下边界
    if (rank < size-1) {
        int *send_buf_upper = (int*)tracked_malloc(nx * ny * sizeof(int));
        if (!send_buf_upper) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                send_buf_upper[i*ny+j] = medium[i][j][halo+local_nz-1];
            }
        }
        MPI_Send(send_buf_upper, nx*ny, MPI_INT, rank+1, tag_upper, MPI_COMM_WORLD);
        free(send_buf_upper);
    }
    
    if (rank > 0) {
        int *recv_buf_lower = (int*)tracked_malloc(nx * ny * sizeof(int));
        if (!recv_buf_lower) {
            fprintf(stderr, "Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        MPI_Recv(recv_buf_lower, nx*ny, MPI_INT, rank-1, tag_upper, MPI_COMM_WORLD, &status);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                medium[i][j][0] = recv_buf_lower[i*ny+j];
            }
        }
        free(recv_buf_lower);
    }
    
    if (rank == 0) {
        printf("Successfully loaded medium map from %s\n", filename);
    }
    
    return medium;
}

// 初始化PML参数
PML_Params init_pml(int size, double delta, double dt, double epsilon0, double mu0, double reflection) {
    PML_Params pml;
    
    pml.sigma = (double*)tracked_calloc(size, sizeof(double));
    pml.kappa = (double*)tracked_calloc(size, sizeof(double));
    pml.alpha = (double*)tracked_calloc(size, sizeof(double));
    print_alloc(size * sizeof(double) * 3, "PML sigma+kappa+alpha");
    if (!pml.sigma || !pml.kappa || !pml.alpha) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    // CPML参数设置 (Roden & Gedney, 2000; Gedney, 2011)
    double m = 3.0;                     // 多项式阶数
    double kappa_max = 7.0;             // 实坐标伸缩最大值
    double alpha_max = 0.05;            // CFS频移最大值
    double eta0 = sqrt(mu0 / epsilon0); // 真空波阻抗 ≈ 377 Ω
    double sigma_max = -(m + 1) * log(reflection) / (2.0 * eta0 * delta * size);
    
    for (int i = 0; i < size; i++) {
        double r = (double)(size - i) / size; // ρ/L, 界面=0, 外部=1
        pml.sigma[i] = sigma_max * pow(r, m);
        pml.kappa[i] = 1.0 + (kappa_max - 1.0) * pow(r, m);
        pml.alpha[i] = alpha_max * (1.0 - r); // 界面最大, 向外递减
    }
    
    return pml;
}

// 释放PML内存
void free_pml(PML_Params pml) {
    free(pml.sigma);
    free(pml.kappa);
    free(pml.alpha);
}

// 保存场分布到文件（并行版本）- 修改：支持不同方向的切片
void save_field_slice_parallel(const char* filename, double ***field, int nx, int ny, int nz, 
                              int halo, int local_z_start, int local_nz, char axis, int position,
                              int rank, int size) {
	if (0) {
		double max_val = -1e100;
		double min_val = 1e100;
		int nan_count = 0;

		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				double val = field[position][j][k];
				if (isnan(val)) nan_count++;
				if (val > max_val) max_val = val;
				if (val < min_val) min_val = val;
			}
		}


		printf("rank %d Slice stats: min=%e, max=%e, NaN count=%d\n", rank,
			min_val, max_val, nan_count);
	}

    // 根据切片轴处理不同的切片方向
    if (axis == 'z') {
        // z方向切片 - 原始逻辑
        // 检查position是否在当前进程
        int has_slice = 0;
        int local_z = -1;
        if (position >= local_z_start && position < local_z_start + local_nz) {
            has_slice = 1;
            local_z = position - local_z_start + halo;
        }
        
        // 广播哪个进程包含切片
        int owner_rank = -1;
        if (has_slice) {
            owner_rank = rank;
        }
        
        // 找到包含切片的进程
        int global_owner;
        MPI_Allreduce(&owner_rank, &global_owner, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        
        if (global_owner == -1) {
            if (rank == 0) {
                fprintf(stderr, "Error: No process contains slice %c=%d\n", axis, position);
            }
            return;
        }
        
        // 准备数据缓冲区
        float *slice_data = NULL;
        if (rank == global_owner) {
            slice_data = (float*)tracked_malloc(nx * ny * sizeof(float));
            if (!slice_data) {
                fprintf(stderr, "Memory allocation failed\n");
                return;
            }
            
            for (int i = 0; i < nx; i++) {
                for (int j = 0; j < ny; j++) {
                    slice_data[i*ny+j] = (float)field[i][j][local_z];
                }
            }
        }
        
        // 主进程接收数据
        if (rank == 0) {
            if (global_owner != 0) {
                // 如果不是主进程包含切片，则接收数据
                slice_data = (float*)tracked_malloc(nx * ny * sizeof(float));
                if (!slice_data) {
                    fprintf(stderr, "Memory allocation failed\n");
                    return;
                }
                
                MPI_Status status;
                MPI_Recv(slice_data, nx*ny, MPI_FLOAT, global_owner, 0, MPI_COMM_WORLD, &status);
            }
            
            // 保存到文件
            FILE *file = fopen(filename, "wb");
            if (file == NULL) {
                perror("Error opening field file");
                free(slice_data);
                return;
            }
            
            // 写入尺寸信息
            fwrite(&nx, sizeof(int), 1, file);
            fwrite(&ny, sizeof(int), 1, file);
            
            // 写入切片数据
            fwrite(slice_data, sizeof(float), nx*ny, file);
            fclose(file);
            free(slice_data);
        } 
        else if (rank == global_owner && global_owner != 0) {
            // 如果当前进程包含切片但不是主进程，则发送给主进程
            MPI_Send(slice_data, nx*ny, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            free(slice_data);
        }
    }
	else if (axis == 'x') {
		// x方向切片 - 每个进程都有部分数据（在z方向）
		// 检查position是否在有效范围内
		if (position < 0 || position >= nx) {
			if (rank == 0) {
				fprintf(stderr, "Error: Invalid x slice position %d (range: 0-%d)\n", position, nx - 1);
			}
			return;
		}

		// 每个进程准备自己的数据
		// x方向切片是y-z平面，每个进程都有完整的y方向数据，但z方向是部分的
		int local_slice_size = ny * local_nz;
		float* local_slice = (float*)tracked_malloc(local_slice_size * sizeof(float));
		if (!local_slice) {
			fprintf(stderr, "Memory allocation failed\n");
			return;
		}

		// 提取本地数据（不包括halo区域）
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				local_slice[j * local_nz + k] = (float)field[position][j][k];
			}
		}

		// 主进程收集所有数据
		if (rank == 0) {
			// 分配完整切片数据
			float* full_slice = (float*)tracked_malloc(ny * nz * sizeof(float));
			if (!full_slice) {
				fprintf(stderr, "Memory allocation failed\n");
				free(local_slice);
				return;
			}

			// 收集每个进程的数据
			// 首先复制主进程自己的数据
			for (int j = 0; j < ny; j++) {
				memcpy(&full_slice[j * nz],
					&local_slice[j * local_nz],
					local_nz * sizeof(float));
			}
			//memcpy(full_slice, local_slice, local_slice_size * sizeof(float));
			if (0) {
				char test_filename[256];
				sprintf(test_filename, "local_slice0.txt");
				FILE* test = fopen(test_filename, "w");

				if (test == NULL) {
					fprintf(stderr, "Rank %d: Failed to open file %s\n", rank, test_filename);
				}
				else {
					//fprintf(test, "# Time step %d, Rank %d: local_z_start=%d, local_nz=%d, halo=%d\n",
					//	t, rank, local_z_start, local_nz, halo);
					//fprintf(test, "# j  global_k  Ex_value\n");

					for (int j = 0; j < ny; j++) {
						for (int k = 0; k < local_nz; k++) {
							int global_k = k + local_z_start;
							fprintf(test, "%d  %d  %e  %e\n", j, global_k, local_slice[j * local_nz + k], full_slice[j * nz + k]);
						}
					}
					fclose(test);
				}
			}

			// 接收其他进程的数据
			for (int src = 1; src < size; src++) {
				// 计算源进程的local_z_start和local_nz
				int src_base_nz = nz / size;
				int src_remainder = nz % size;
				int src_local_nz = (src < src_remainder) ? (src_base_nz + 1) : src_base_nz;
				int src_local_z_start = 0;

				for (int r = 0; r < src; r++) {
					src_local_z_start += (r < src_remainder) ? (src_base_nz + 1) : src_base_nz;
				}

				int src_slice_size = ny * src_local_nz;
				float* src_slice = (float*)tracked_malloc(src_slice_size * sizeof(float));
				if (!src_slice) {
					fprintf(stderr, "Memory allocation failed\n");
					free(full_slice);
					free(local_slice);
					return;
				}

				MPI_Status status;
				MPI_Recv(src_slice, src_slice_size, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
				if (0) {
					char test_filename[256];
					sprintf(test_filename, "src_slice%d.txt",src);
					FILE* test = fopen(test_filename, "w");

					if (test == NULL) {
						fprintf(stderr, "Rank %d: Failed to open file %s\n", rank, test_filename);
					}
					else {
						//fprintf(test, "# Time step %d, Rank %d: local_z_start=%d, local_nz=%d, halo=%d\n",
						//	t, rank, local_z_start, local_nz, halo);
						//fprintf(test, "# j  global_k  Ex_value\n");

						for (int j = 0; j < ny; j++) {
							for (int k = 0; k < src_local_nz; k++) {
								int global_k = k + src_local_z_start;
								fprintf(test, "%d  %d  %e\n", j, global_k, src_slice[j * src_local_nz + k]);
							}
						}
						fclose(test);
					}
				}

				// 将数据复制到正确位置
				for (int j = 0; j < ny; j++) {
					memcpy(&full_slice[j * nz + src_local_z_start],
						&src_slice[j * src_local_nz],
						src_local_nz * sizeof(float));
				}
				free(src_slice);
			}

			// 保存到文件
			FILE* file = fopen(filename, "wb");
			if (file == NULL) {
				perror("Error opening field file");
				free(full_slice);
				free(local_slice);
				return;
			}

			// 写入尺寸信息
			fwrite(&ny, sizeof(int), 1, file);
			fwrite(&nz, sizeof(int), 1, file);
			if (0) {
				char test_filename[256];
				sprintf(test_filename, "full_slice_in.txt");
				FILE* test = fopen(test_filename, "w");

				if (test == NULL) {
					fprintf(stderr, "Rank %d: Failed to open file %s\n", rank, test_filename);
				}
				else {
					//fprintf(test, "# Time step %d, Rank %d: local_z_start=%d, local_nz=%d, halo=%d\n",
					//	t, rank, local_z_start, local_nz, halo);
					//fprintf(test, "# j  global_k  Ex_value\n");

					for (int j = 0; j < ny; j++) {
						for (int k = 0; k < nz; k++) {
							fprintf(test, "%d  %d  %e\n", j, k, full_slice[j * nz + k]);
						}
					}
					fclose(test);
				}
			}
			


			// 写入切片数据
			fwrite(full_slice, sizeof(float), ny * nz, file);
			fclose(file);
			free(full_slice);
		}
		else {
			// 从进程发送数据给主进程
			MPI_Send(local_slice, local_slice_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
		if (0) {
			char test_filename[256];
			sprintf(test_filename, "test_slice_in_rank%d.txt", rank);
			FILE* test = fopen(test_filename, "w");

			if (test == NULL) {
				fprintf(stderr, "Rank %d: Failed to open file %s\n", rank, test_filename);
			}
			else {
				//fprintf(test, "# Time step %d, Rank %d: local_z_start=%d, local_nz=%d, halo=%d\n",
				//	t, rank, local_z_start, local_nz, halo);
				//fprintf(test, "# j  global_k  Ex_value\n");

				for (int j = 0; j < ny; j++) {
					for (int k = 0; k < local_nz; k++) {
						int global_k = k + local_z_start;
						fprintf(test, "%d  %d  %e\n", j, global_k, local_slice[j * local_nz + k]);
					}
				}
				fclose(test);
			}
		}
		

		free(local_slice);
    }
    else if (axis == 'y') {
        // y方向切片 - 每个进程都有部分数据（在z方向）
        // 检查position是否在有效范围内
        if (position < 0 || position >= ny) {
            if (rank == 0) {
                fprintf(stderr, "Error: Invalid y slice position %d (range: 0-%d)\n", position, ny-1);
            }
            return;
        }
        
        // 每个进程准备自己的数据
        // y方向切片是x-z平面，每个进程都有完整的x方向数据，但z方向是部分的
        int local_slice_size = nx * local_nz;
        float *local_slice = (float*)tracked_malloc(local_slice_size * sizeof(float));
        if (!local_slice) {
            fprintf(stderr, "Memory allocation failed\n");
            return;
        }
        
        // 提取本地数据（不包括halo区域）
        for (int i = 0; i < nx; i++) {
            for (int k = 0; k < local_nz; k++) {
                local_slice[i * local_nz + k] = (float)field[i][position][k];
            }
        }
        
        // 主进程收集所有数据
        if (rank == 0) {
            // 分配完整切片数据
            float *full_slice = (float*)tracked_malloc(nx * nz * sizeof(float));
            if (!full_slice) {
                fprintf(stderr, "Memory allocation failed\n");
                free(local_slice);
                return;
            }
            
            // 收集每个进程的数据
            // 首先复制主进程自己的数据
            memcpy(full_slice, local_slice, local_slice_size * sizeof(float));
            
            // 接收其他进程的数据
            for (int src = 1; src < size; src++) {
                // 计算源进程的local_z_start和local_nz
                int src_base_nz = nz / size;
                int src_remainder = nz % size;
                int src_local_nz = (src < src_remainder) ? (src_base_nz + 1) : src_base_nz;
                int src_local_z_start = 0;
                
                for (int r = 0; r < src; r++) {
                    src_local_z_start += (r < src_remainder) ? (src_base_nz + 1) : src_base_nz;
                }
                
                int src_slice_size = nx * src_local_nz;
                float *src_slice = (float*)tracked_malloc(src_slice_size * sizeof(float));
                if (!src_slice) {
                    fprintf(stderr, "Memory allocation failed\n");
                    free(full_slice);
                    free(local_slice);
                    return;
                }
                
                MPI_Status status;
                MPI_Recv(src_slice, src_slice_size, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
                
                // 将数据复制到正确位置
                for (int i = 0; i < nx; i++) {
                    memcpy(&full_slice[i * nz + src_local_z_start], 
                           &src_slice[i * src_local_nz], 
                           src_local_nz * sizeof(float));
                }
                
                free(src_slice);
            }
            
            // 保存到文件
            FILE *file = fopen(filename, "wb");
            if (file == NULL) {
                perror("Error opening field file");
                free(full_slice);
                free(local_slice);
                return;
            }
            
            // 写入尺寸信息
            fwrite(&nx, sizeof(int), 1, file);
            fwrite(&nz, sizeof(int), 1, file);
            
            // 写入切片数据
            fwrite(full_slice, sizeof(float), nx*nz, file);
            fclose(file);
            free(full_slice);
            free(local_slice);
        }
        else {
            // 从进程发送数据给主进程
            MPI_Send(local_slice, local_slice_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            free(local_slice);
        }
    }
    else {
        if (rank == 0) {
            fprintf(stderr, "Error: Invalid slice axis '%c'. Must be 'x', 'y', or 'z'.\n", axis);
        }
        return;
    }
}

// 从配置文件读取激励源
void parse_source(Config *config, char *line) {
    static int source_count = 0;
    
    // 重新分配内存以容纳新源
    size_t _realloc_old = source_count * sizeof(Source);
    Source *temp = realloc(config->sources, (source_count + 1) * sizeof(Source));
    if (temp) track_realloc_delta(_realloc_old, (source_count + 1) * sizeof(Source));
    if (!temp) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    config->sources = temp;
    Source *src = &config->sources[source_count];
    
    // 初始化默认值
    memset(src, 0, sizeof(Source)); // 确保所有字段初始化为0
    src->type = 0; // 高斯脉冲
    src->position[0] = config->nx / 2;
    src->position[1] = config->ny / 2;
    src->position[2] = config->nz / 3;
    src->amplitude = 1.0;
    src->frequency = 1e9;
    src->t0 = 40 * config->dt;
    src->tau = 15 * config->dt;
    src->component = 'z';
    src->start_time = 0.0;
    src->end_time = config->tsteps * config->dt;

	// 添加wavenumber标志
	int has_frequency = 0;
	int has_wavenumber = 0;
	double wavenumber = 0.0;
    
    // 使用安全的字符串解析
    char *token, *saveptr;
    token = strtok_r(line, ",", &saveptr);
    while (token != NULL) {
        char key[32], value[1024];
        if (sscanf(token, " %31[^=] = %1023s", key, value) == 2) {
            trim(key);
            trim(value);
            
            if (strcmp(key, "type") == 0) {
				if (strcmp(value, "gaussian") == 0) src->type = 0;
				else if (strcmp(value, "sinusoidal") == 0) src->type = 1;
				else if (strcmp(value, "ricker") == 0) src->type = 2;
				else if (strcmp(value, "dipole") == 0) src->type = 3;
            }
            else if (strcmp(key, "x") == 0) {
                src->position[0] = atoi(value);
            }
            else if (strcmp(key, "y") == 0) {
                src->position[1] = atoi(value);
            }
            else if (strcmp(key, "z") == 0) {
                src->position[2] = atoi(value);
            }
            else if (strcmp(key, "amplitude") == 0) {
                src->amplitude = strtod(value, NULL);
            }
			else if (strcmp(key, "frequency") == 0) {
				src->frequency = strtod(value, NULL);
				has_frequency = 1;
			}
			else if (strcmp(key, "wavenumber") == 0) {
				wavenumber = strtod(value, NULL);
				has_wavenumber = 1;
			}
            else if (strcmp(key, "t0") == 0) {
                src->t0 = strtod(value, NULL);
            }
            else if (strcmp(key, "tau") == 0) {
                src->tau = strtod(value, NULL);
            }
            else if (strcmp(key, "component") == 0) {
                src->component = value[0];
            }
            else if (strcmp(key, "start_time") == 0) {
                src->start_time =  strtod(value, NULL);
            }
            else if (strcmp(key, "end_time") == 0) {
                src->end_time =  strtod(value, NULL);
            }
        }
        token = strtok_r(NULL, ",", &saveptr);
    }

	// 检查dipole源的参数设置
	if (src->type == 3) { // dipole类型
		if (has_frequency && has_wavenumber) {
			fprintf(stderr, "Error: Cannot specify both frequency and wavenumber for dipole source\n");
			exit(EXIT_FAILURE);
		}

		if (!has_frequency && !has_wavenumber) {
			fprintf(stderr, "Error: Must specify either frequency or wavenumber for dipole source\n");
			exit(EXIT_FAILURE);
		}

		if (has_wavenumber) {
			// 转换wavenumber到frequency: frequency = wavenumber * 0.03e12
			src->frequency = wavenumber * 0.03e12;
		}
	}
    
    source_count++;
    config->num_sources = source_count;
}

void parse_material(Config* config, char* line) {
	int mat_id = -1;
	char* token, * saveptr;
	Material* mat = NULL;

	// 创建行的副本用于第一次解析
	char line_copy[4096];
	strncpy(line_copy, line, sizeof(line_copy));
	line_copy[sizeof(line_copy) - 1] = '\0';

	// 首先从副本中解析材料ID
	token = strtok_r(line_copy, ",", &saveptr);
	while (token != NULL) {
		char key[32], value[1024];
		if (sscanf(token, " %31[^=] = %1023s", key, value) == 2) {
			trim(key);
			trim(value);

			if (strcmp(key, "id") == 0) {
				mat_id = atoi(value);
				break;
			}
		}
		token = strtok_r(NULL, ",", &saveptr);
	}

	// 检查材料ID是否有效
	if (mat_id < 0) {
		fprintf(stderr, "Error: Material must have a valid ID\n");
		return;
	}

	// 如果材料ID大于当前材料数量，扩展材料数组
	if (mat_id >= config->num_materials) {
		int new_num = mat_id + 1;
		size_t _realloc_old = config->num_materials * sizeof(Material);
		Material* temp = realloc(config->materials, new_num * sizeof(Material));
		if (temp) track_realloc_delta(_realloc_old, new_num * sizeof(Material));
		if (!temp) {
			fprintf(stderr, "Memory allocation failed\n");
			exit(EXIT_FAILURE);
		}
		config->materials = temp;

		// 初始化新添加的材料为真空
		for (int i = config->num_materials; i < new_num; i++) {
			strncpy(config->materials[i].name, "Vacuum",sizeof(config->materials[i].name));
			config->materials[i].eps.xx = 1.0; config->materials[i].eps.xy = 0.0; config->materials[i].eps.xz = 0.0;
			config->materials[i].eps.yx = 0.0; config->materials[i].eps.yy = 1.0; config->materials[i].eps.yz = 0.0;
			config->materials[i].eps.zx = 0.0; config->materials[i].eps.zy = 0.0; config->materials[i].eps.zz = 1.0;
			config->materials[i].eps.xxi = 0.0; config->materials[i].eps.xyi = 0.0; config->materials[i].eps.xzi = 0.0;
			config->materials[i].eps.yxi = 0.0; config->materials[i].eps.yyi = 0.0; config->materials[i].eps.yzi = 0.0;
			config->materials[i].eps.zxi = 0.0; config->materials[i].eps.zyi = 0.0; config->materials[i].eps.zzi = 0.0;
			config->materials[i].eps_inv = inverse_dielectric_tensor(config->materials[i].eps);
			config->materials[i].sigma = 0.0;
			config->materials[i].sigma_m = 0.0;
			config->materials[i].natom = 0;
			config->materials[i].born = NULL;
			config->materials[i].born_filename[0] = '\0'; // 初始化born_filename
			config->materials[i].use_born_file_for_eps = 0; // 默认不从born文件读取介电张量
			config->materials[i].use_ADE = 0; // 默认不使用ADE方法
			config->materials[i].anglez = 0.0; // 默认不转动晶体
			config->materials[i].anglex = 0.0;
			config->materials[i].switchxz = 0; // 默认不调换XZ
			config->materials[i].use_dielectric_file = 0; // 默认不从dielectric文件读介电张量
			// 确保所有指针初始化为NULL
			config->materials[i].QeZV = NULL;
			config->materials[i].QeZM = NULL;
			config->materials[i].DMM = NULL;
			config->materials[i].QeZV_filename_set = 0;
			config->materials[i].QeZM_filename_set = 0;
			config->materials[i].DMM_filename_set = 0;
		}
		config->num_materials = new_num;
	}

	// 获取当前材料指针
	mat = &config->materials[mat_id];

	// 重新解析原始行，设置材料属性
	saveptr = NULL;
	token = strtok_r(line, ",", &saveptr);
	while (token != NULL) {
		char key[32], value[1024];
		if (sscanf(token, " %31[^=] = %1023s", key, value) == 2) {
			trim(key);
			trim(value);

			if (strcmp(key, "name") == 0) {
				strncpy(mat->name, value, sizeof(mat->name) - 1);
				mat->name[sizeof(mat->name) - 1] = '\0';  // Ensure null termination
			}
			else if (strcmp(key, "eps_xx") == 0) mat->eps.xx = atof(value);
			else if (strcmp(key, "eps_xy") == 0) mat->eps.xy = atof(value);
			else if (strcmp(key, "eps_xz") == 0) mat->eps.xz = atof(value);
			else if (strcmp(key, "eps_yx") == 0) mat->eps.yx = atof(value);
			else if (strcmp(key, "eps_yy") == 0) mat->eps.yy = atof(value);
			else if (strcmp(key, "eps_yz") == 0) mat->eps.yz = atof(value);
			else if (strcmp(key, "eps_zx") == 0) mat->eps.zx = atof(value);
			else if (strcmp(key, "eps_zy") == 0) mat->eps.zy = atof(value);
			else if (strcmp(key, "eps_zz") == 0) mat->eps.zz = atof(value);
			else if (strcmp(key, "eps_xxi") == 0) mat->eps.xxi = atof(value);
			else if (strcmp(key, "eps_xyi") == 0) mat->eps.xyi = atof(value);
			else if (strcmp(key, "eps_xzi") == 0) mat->eps.xzi = atof(value);
			else if (strcmp(key, "eps_yxi") == 0) mat->eps.yxi = atof(value);
			else if (strcmp(key, "eps_yyi") == 0) mat->eps.yyi = atof(value);
			else if (strcmp(key, "eps_yzi") == 0) mat->eps.yzi = atof(value);
			else if (strcmp(key, "eps_zxi") == 0) mat->eps.zxi = atof(value);
			else if (strcmp(key, "eps_zyi") == 0) mat->eps.zyi = atof(value);
			else if (strcmp(key, "eps_zzi") == 0) mat->eps.zzi = atof(value); 
			else if (strcmp(key, "sigma") == 0) mat->sigma = atof(value);
			else if (strcmp(key, "sigma_m") == 0) mat->sigma_m = atof(value);
			else if (strcmp(key, "dielectric") == 0) {
				strncpy(mat->dielectric_filename, value, sizeof(mat->dielectric_filename) - 1);
				mat->dielectric_filename[sizeof(mat->dielectric_filename) - 1] = '\0';
				mat->use_dielectric_file = 1;
			}
			else if (strcmp(key, "switchxz") == 0) mat->switchxz = atoi(value);
			else if (strcmp(key, "natom") == 0) {
				mat->natom = atoi(value);
				// 根据natom分配born数组内存
				if (mat->natom > 0) {
					int born_size = 3 * mat->natom * 3;
					mat->born = (double*)tracked_malloc(born_size * sizeof(double));
					memset(mat->born, 0, born_size * sizeof(double));
				}
			}
			else if (strcmp(key, "QeZV") == 0) {
				strncpy(mat->QeZV_filename, value, sizeof(mat->QeZV_filename) - 1);
				mat->QeZV_filename[sizeof(mat->QeZV_filename) - 1] = '\0';
				mat->QeZV_filename_set = 1; // 设置标志
			}
			else if (strcmp(key, "QeZM") == 0) {
				strncpy(mat->QeZM_filename, value, sizeof(mat->QeZM_filename) - 1);
				mat->QeZM_filename[sizeof(mat->QeZM_filename) - 1] = '\0';
				mat->QeZM_filename_set = 1; // 设置标志
			}
			else if (strcmp(key, "DMM") == 0) {
				strncpy(mat->DMM_filename, value, sizeof(mat->DMM_filename) - 1);
				mat->DMM_filename[sizeof(mat->DMM_filename) - 1] = '\0';
				mat->DMM_filename_set = 1; // 设置标志
			}
			else if (strcmp(key, "use_ADE") == 0) {
				mat->use_ADE = atoi(value);
				if (mat->use_ADE != 0 && mat->use_ADE != 1) {
					fprintf(stderr, "Error: use_ADE must be 0 or 1\n");
					return;
				}
			}
			else if (strcmp(key, "anglez") == 0) mat->anglez = atof(value);
			else if (strcmp(key, "anglex") == 0) mat->anglex = atof(value);
			else if (strcmp(key, "born") == 0) {
				// 检查是否是文件名格式
				if (strchr(value, '[') == NULL && strchr(value, ']') == NULL) {
					// 不是数组格式，假定是文件名
					strncpy(mat->born_filename, value, sizeof(mat->born_filename) - 1);
					mat->born_filename[sizeof(mat->born_filename) - 1] = '\0';
					mat->use_born_file_for_eps = 1; // 设置标志
				}
				else {
					// 原有的数组格式处理
					if (mat->natom <= 0) {
						fprintf(stderr, "Error: natom must be defined before born array\n");
						return;
					}

					// 解析数组格式: born = [v1, v2, v3, ...]
					char* array_start = strchr(value, '[');
					char* array_end = strchr(value, ']');

					if (!array_start || !array_end) {
						fprintf(stderr, "Error: born array must be enclosed in square brackets\n");
						return;
					}

					// 提取数组内容
					*array_end = '\0';
					char* array_content = array_start + 1;

					// 计算需要的元素数量
					int required_elements = 3 * mat->natom * 3;

					// 分配内存
					mat->born = (double*)tracked_malloc(required_elements * sizeof(double));
					if (!mat->born) {
						fprintf(stderr, "Memory allocation failed for born array\n");
						return;
					}

					// 解析逗号分隔的值
					char* token = strtok(array_content, ",");
					int index = 0;

					while (token != NULL && index < required_elements) {
						trim(token);
						mat->born[index++] = atof(token);
						token = strtok(NULL, ",");
					}

					// 检查是否解析了足够的值
					if (index != required_elements) {
						fprintf(stderr, "Error: born array has %d elements but %d are required\n",
							index, required_elements);
						free(mat->born);
						mat->born = NULL;
						return;
					}
				}
			}
		}
		token = strtok_r(NULL, ",", &saveptr);
	}

	// 更新逆矩阵
	mat->eps_inv = inverse_dielectric_tensor(mat->eps);
}

// 从配置文件读取参数
Config read_config(const char* filename) {
    Config config = {0};
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening config file");
        exit(EXIT_FAILURE);
    }
    
    // 默认值
    config.nx = 50;
    config.ny = 50;
    config.nz = 50;
    config.tsteps = 500;
    config.delta_x = 0.01;
    config.delta_y = 0.01;
    config.delta_z = 0.01;
    config.pml_thickness = 10;
	config.pml_reflection = 0.001;  // 默认反射率 0.1%
    config.output_interval = 20;
    strcpy(config.medium_file, "medium_map.bin");
    config.c = 3e8;
    config.epsilon0 = 8.854e-12;
    config.mu0 = M_PI * 4e-7;
    config.sources = NULL;
    config.num_sources = 0;
	config.dt = -1.0;
	strcpy(config.output_component, "Ex");
	// 输出切片默认配置
	config.output_slice_axis = 'z';  // 默认z轴切片
	config.output_slice_position = config.nz / 2;  // 默认中间位置
	config.full_monitor_enabled = 0;
	strncpy(config.full_monitor_output_file, "full_monitor.dat", sizeof(config.full_monitor_output_file));
	config.unique_source_frequency = -1.0;

    char line[4096];
	while (fgets(line, sizeof(line), file)) {
		// 去除换行符
		line[strcspn(line, "\n")] = 0;

		// 跳过注释行和空行
		if (line[0] == '#' || line[0] == '\n') continue;
		// 处理源定义行
		if (strstr(line, "source") == line) {
			parse_source(&config, line + 6); // 跳过"source"
			config.unique_source_frequency = config.sources[0].frequency;
			continue;
		}
	}
	fclose(file);

	// 初始化默认材料库
	config.num_materials = 4;
	config.materials = create_material_library(&config.num_materials, config.unique_source_frequency);

	file = fopen(filename, "r");
	if (file == NULL) {
		perror("Error opening config file");
		exit(EXIT_FAILURE);
	}
	while (fgets(line, sizeof(line), file)) {
        // 去除换行符
        line[strcspn(line, "\n")] = 0;
        
        // 跳过注释行和空行
        if (line[0] == '#' || line[0] == '\n') continue;
        

		if (strstr(line, "material") == line) {
			parse_material(&config, line + 8); // 跳过"material"
			continue;
		}
       
        // 解析其他参数
        char key[64], value[1024];
        if (sscanf(line, "%63[^=]=%1023[^\n]", key, value) == 2) {
            trim(key);
            trim(value);
            
			if (strcmp(key, "nx") == 0) config.nx = atoi(value);
			else if (strcmp(key, "ny") == 0) config.ny = atoi(value);
			else if (strcmp(key, "nz") == 0) config.nz = atoi(value);
			else if (strcmp(key, "tsteps") == 0) config.tsteps = atoi(value);
            else if (strcmp(key, "delta_x") == 0) config.delta_x = atof(value);
            else if (strcmp(key, "delta_y") == 0) config.delta_y = atof(value);
            else if (strcmp(key, "delta_z") == 0) config.delta_z = atof(value);
            else if (strcmp(key, "pml_thickness") == 0) config.pml_thickness = atoi(value);
			else if (strcmp(key, "pml_reflection") == 0) config.pml_reflection = atof(value);  // 读取PML反射率
            else if (strcmp(key, "output_interval") == 0) config.output_interval = atoi(value);
            else if (strcmp(key, "medium_file") == 0) strncpy(config.medium_file, value, sizeof(config.medium_file));
			else if (strcmp(key, "dt") == 0) config.dt = atof(value);
			else if (strcmp(key, "output_component") == 0) {
				strncpy(config.output_component, value, sizeof(config.output_component));
				config.output_component[sizeof(config.output_component) - 1] = '\0'; // 确保以空字符结尾
				// 添加验证逻辑
				if (strcmp(config.output_component, "Ex") != 0 &&
					strcmp(config.output_component, "Ey") != 0 &&
					strcmp(config.output_component, "Ez") != 0 &&
					strcmp(config.output_component, "E") != 0 &&
					strcmp(config.output_component, "Dx") != 0 &&
					strcmp(config.output_component, "Dy") != 0 &&
					strcmp(config.output_component, "Dz") != 0 ) {
					fprintf(stderr, "错误: output_component 必须是Ex, Ey, Ez, Dx, Dy, Dz或E之一，当前值为 '%s'\n",
						config.output_component);
					fprintf(stderr, "将使用默认值 'Ex'\n");
					strcpy(config.output_component, "Ex"); // 使用默认值
				}
			}
			else if (strcmp(key, "output_slice_axis") == 0) {
				// 解析切片轴
				if (strlen(value) > 0) {
					config.output_slice_axis = value[0];
					// 验证轴的有效性
					if (config.output_slice_axis != 'x' && 
						config.output_slice_axis != 'y' && 
						config.output_slice_axis != 'z') {
						fprintf(stderr, "警告: output_slice_axis 必须是 'x', 'y' 或 'z'，当前值为 '%c'，使用默认值 'z'\n",
							config.output_slice_axis);
						config.output_slice_axis = 'z';
					}
				}
			}
			else if (strcmp(key, "output_slice_position") == 0) {
				config.output_slice_position = atoi(value);
			}
			// 解析监测点配置
			else if (strcmp(key, "monitor_point") == 0) {
				config.monitor_enabled = 1;

				// 创建value的副本进行解析，避免修改原始字符串
				char value_copy[1024];
				strncpy(value_copy, value, sizeof(value_copy) - 1);
				value_copy[sizeof(value_copy) - 1] = '\0';

				// 解析x,y,z,atom,dir
				char* token = strtok(value_copy, ",");
				while (token != NULL) {
					char subkey[32], subvalue[1024];
					if (sscanf(token, "%31[^=]=%1023s", subkey, subvalue) == 2) {
						trim(subkey);
						trim(subvalue);
						if (strcmp(subkey, "x") == 0) {
							config.monitor_x = atoi(subvalue);
						}
						else if (strcmp(subkey, "y") == 0) {
							config.monitor_y = atoi(subvalue);
						}
						else if (strcmp(subkey, "z") == 0) config.monitor_z = atoi(subvalue);
						else if (strcmp(subkey, "atom") == 0) config.monitor_atom = atoi(subvalue);
						else if (strcmp(subkey, "dir") == 0) {
							if (strcmp(subvalue, "x") == 0) config.monitor_dir = 0;
							else if (strcmp(subvalue, "y") == 0) config.monitor_dir = 1;
							else if (strcmp(subvalue, "z") == 0) config.monitor_dir = 2;
							else {
								fprintf(stderr, "Warning: Invalid direction '%s' for monitor_point. Using 'x'.\n", subvalue);
								config.monitor_dir = 0;
							}
						}
					}
					token = strtok(NULL, ",");
				}
			}
			else if (strcmp(key, "monitor_output_file") == 0) {
				strncpy(config.monitor_output_file, value, sizeof(config.monitor_output_file) - 1);
				config.monitor_output_file[sizeof(config.monitor_output_file) - 1] = '\0';
			}
			else if (strcmp(key, "full_monitor") == 0) config.full_monitor_enabled = atoi(value);
			else if (strcmp(key, "full_monitor_output_file") == 0) {
				strncpy(config.full_monitor_output_file, value, sizeof(config.full_monitor_output_file) - 1);
				config.full_monitor_output_file[sizeof(config.full_monitor_output_file) - 1] = '\0';
			}
        }
    }
    
    fclose(file);
    
    // 计算时间步长 (CFL条件)
    double min_delta = config.delta_x;
    if (config.delta_y < min_delta) min_delta = config.delta_y;
    if (config.delta_z < min_delta) min_delta = config.delta_z;
	double temp = min_delta / (config.c * sqrt(3.0)) * 0.99; // 安全系数0.99
	if (config.dt < 0 || config.dt > temp) config.dt = temp;
    
    // 为源设置默认时间参数
    for (int i = 0; i < config.num_sources; i++) {
        if (config.sources[i].end_time <= 0) {
            config.sources[i].end_time = config.tsteps * config.dt;
        }
    }

    return config;
}

// 计算激励源值
double calculate_source_value(Source *src, double time) {
    if (time < src->start_time || time > src->end_time) {
        return 0.0;
    }
    
    switch (src->type) {
        case 0: // 高斯脉冲
            return src->amplitude * exp(-pow((time - src->t0) / src->tau, 2));
        
        case 1: // 正弦波
            return src->amplitude * sin(2 * M_PI * src->frequency * time);
        
        case 2: // Ricker小波(二阶高斯导数)
            {
                double t_shift = time - src->t0;
                double arg = M_PI * M_PI * src->frequency * src->frequency * t_shift * t_shift;
                return src->amplitude * (1.0 - 2.0 * arg) * exp(-arg);
            }
		case 3: // 偶极辐射
			return src->amplitude* sin(2 * M_PI * src->frequency * time) * exp(-pow((time - src->t0) / src->tau, 2));

        default:
            return 0.0;
    }
}

void exchange_halo_z(double ***field, int nx, int ny, int local_nz, int halo, int rank, int size) {
    MPI_Status status;
    int total_points = nx * ny;
    double *send_buf_lower = NULL;
    double *recv_buf_lower = NULL;
    double *send_buf_upper = NULL;
    double *recv_buf_upper = NULL;

    // 发送下边界内部点给下邻居，并接收下邻居的上边界数据
    if (rank > 0) {
        send_buf_lower = (double*)tracked_malloc(total_points * sizeof(double));
        recv_buf_lower = (double*)tracked_malloc(total_points * sizeof(double));
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                send_buf_lower[i*ny+j] = field[i][j][halo];  // 下边界内部点
            }
        }
        // 与下邻居交换数据
        MPI_Sendrecv(send_buf_lower, total_points, MPI_DOUBLE, rank-1, 0,
                     recv_buf_lower, total_points, MPI_DOUBLE, rank-1, 1,
                     MPI_COMM_WORLD, &status);
        // 填充下边界halo
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                field[i][j][0] = recv_buf_lower[i*ny+j];
            }
        }
        free(send_buf_lower);
        free(recv_buf_lower);
    }

    // 发送上边界内部点给上邻居，并接收上邻居的下边界数据
    if (rank < size-1) {
        send_buf_upper = (double*)tracked_malloc(total_points * sizeof(double));
        recv_buf_upper = (double*)tracked_malloc(total_points * sizeof(double));
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                send_buf_upper[i*ny+j] = field[i][j][halo+local_nz-1];  // 上边界内部点
            }
        }
        // 与上邻居交换数据
        MPI_Sendrecv(send_buf_upper, total_points, MPI_DOUBLE, rank+1, 1,
                     recv_buf_upper, total_points, MPI_DOUBLE, rank+1, 0,
                     MPI_COMM_WORLD, &status);
        // 填充上边界halo
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                field[i][j][halo+local_nz] = recv_buf_upper[i*ny+j];
            }
        }
        free(send_buf_upper);
        free(recv_buf_upper);
    }
}

// 优化13：将3个分量打包成一条MPI消息
// 优化17：使用预分配缓冲区的halo交换
void exchange_halo_z_vec3_with_buffers(double*** vx, double*** vy, double*** vz, int nx, int ny, int local_nz, int halo, int rank, int size, double* send_lower, double* recv_lower, double* send_upper, double* recv_upper) {
	MPI_Status status;
	int total_points = nx * ny;
	int buf_size = total_points * 3;  // 3个分量

	// 发送下边界
	if (rank > 0) {
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				int idx = (i * ny + j) * 3;
				send_lower[idx] = vx[i][j][0];
				send_lower[idx + 1] = vy[i][j][0];
				send_lower[idx + 2] = vz[i][j][0];
			}
		}
		MPI_Sendrecv(send_lower, buf_size, MPI_DOUBLE, rank - 1, 0,
			recv_lower, buf_size, MPI_DOUBLE, rank - 1, 1,
			MPI_COMM_WORLD, &status);
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				int idx = (i * ny + j) * 3;
				vx[i][j][-1] = recv_lower[idx];
				vy[i][j][-1] = recv_lower[idx + 1];
				vz[i][j][-1] = recv_lower[idx + 2];
			}
		}
	}

	// 发送上边界
	if (rank < size - 1) {
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				int idx = (i * ny + j) * 3;
				send_upper[idx] = vx[i][j][local_nz - 1];
				send_upper[idx + 1] = vy[i][j][local_nz - 1];
				send_upper[idx + 2] = vz[i][j][local_nz - 1];
			}
		}
		MPI_Sendrecv(send_upper, buf_size, MPI_DOUBLE, rank + 1, 1,
			recv_upper, buf_size, MPI_DOUBLE, rank + 1, 0,
			MPI_COMM_WORLD, &status);
		for (int i = 0; i < nx; i++) {
			for (int j = 0; j < ny; j++) {
				int idx = (i * ny + j) * 3;
				vx[i][j][local_nz] = recv_upper[idx];
				vy[i][j][local_nz] = recv_upper[idx + 1];
				vz[i][j][local_nz] = recv_upper[idx + 2];
			}
		}
	}
}

// 优化17：旧的函数保留兼容（预分配版本在循环外调用）
/*void exchange_halo_z_vec3(double*** vx, double*** vy, double*** vz, int nx, int ny, int local_nz, int halo, int rank, int size) {
	int total_points = nx * ny;
	int buf_size = total_points * 3;
	double* send_lower = (double*)tracked_malloc(buf_size * sizeof(double));
	double* recv_lower = (double*)tracked_malloc(buf_size * sizeof(double));
	double* send_upper = (double*)tracked_malloc(buf_size * sizeof(double));
	double* recv_upper = (double*)tracked_malloc(buf_size * sizeof(double));

	exchange_halo_z_vec3_with_buffers(vx, vy, vz, nx, ny, local_nz, halo, rank, size, send_lower, recv_lower, send_upper, recv_upper);

	free(send_lower);
	free(recv_lower);
	free(send_upper);
	free(recv_upper);
}*/

int main(int argc, char *argv[]) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    g_mpi_rank = rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    g_mpi_size = size;
	FILE* monitor_file = NULL;
	FILE* full_monitor_file = NULL;
	FILE* source_file = NULL;

	double**** r = NULL;   // r[i][j][k] 指向长度为 (natom*3) 的数组
	double**** v = NULL;   // v[i][j][k] 指向长度为 (natom*3) 的数组
	double**** P_ADE = NULL; //P_ADE[i][j][k] 指向长度为 (use_ADE*3) 的数组
	double**** P_ADEo = NULL; //P_ADEo[i][j][k] 指向长度为 (use_ADE*3) 的数组

    // 检查命令行参数
	if (rank == 0) printf("FDTD v2026.09.27 running on %d processors\nInitializing",size);
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <config_file.cfg>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // 初始化配置结构体
	if (rank == 0) printf(" .");
    Config config = {0};
    memset(&config, 0, sizeof(Config));

    // 读取配置文件 (仅主进程读取)
	if (rank == 0) printf(" .");
    if (rank == 0) {
        config = read_config(argv[1]);
    }
    
    // ==== 安全广播配置 ====
	if (rank == 0) printf(" .");

	struct {
        int nx, ny, nz, tsteps, pml_thickness, output_interval, num_sources;
        double delta_x, delta_y, delta_z, dt, c, epsilon0, mu0, pml_reflection;
		int num_materials; // 添加材料数量字段
		char output_slice_axis; // 切片轴
		int output_slice_position; // 切片位置
		// 监测点配置
		int monitor_enabled;
		int monitor_x, monitor_y, monitor_z;
		int monitor_atom, monitor_dir;
		int full_monitor_enabled;
		double unique_source_frequency;
		} config_bcast;

	if (rank == 0) printf(" .");
	if (rank == 0) {
        config_bcast.nx = config.nx;
        config_bcast.ny = config.ny;
        config_bcast.nz = config.nz;
        config_bcast.tsteps = config.tsteps;
        config_bcast.pml_thickness = config.pml_thickness;
		config_bcast.pml_reflection = config.pml_reflection;
        config_bcast.output_interval = config.output_interval;
        config_bcast.delta_x = config.delta_x;
        config_bcast.delta_y = config.delta_y;
        config_bcast.delta_z = config.delta_z;
        config_bcast.dt = config.dt;
        config_bcast.c = config.c;
        config_bcast.epsilon0 = config.epsilon0;
        config_bcast.mu0 = config.mu0;
        config_bcast.num_sources = config.num_sources;
		config_bcast.num_materials = config.num_materials; // 添加材料数量
		config_bcast.output_slice_axis = config.output_slice_axis; // 切片轴
		config_bcast.output_slice_position = config.output_slice_position; // 切片位置
		// 监测点配置
		config_bcast.monitor_enabled = config.monitor_enabled;
		config_bcast.monitor_x = config.monitor_x;
		config_bcast.monitor_y = config.monitor_y;
		config_bcast.monitor_z = config.monitor_z;
		config_bcast.monitor_atom = config.monitor_atom;
		config_bcast.monitor_dir = config.monitor_dir;
		config_bcast.full_monitor_enabled = config.full_monitor_enabled;
		config_bcast.unique_source_frequency = config.unique_source_frequency;
	}
    
	if (rank == 0) printf(" .");
	MPI_Bcast(&config_bcast, sizeof(config_bcast), MPI_BYTE, 0, MPI_COMM_WORLD);
    
	if (rank == 0) printf(" .");
	if (rank != 0) {
        config.nx = config_bcast.nx;
        config.ny = config_bcast.ny;
        config.nz = config_bcast.nz;
        config.tsteps = config_bcast.tsteps;
        config.pml_thickness = config_bcast.pml_thickness;
		config.pml_reflection = config_bcast.pml_reflection;
		config.output_interval = config_bcast.output_interval;
        config.delta_x = config_bcast.delta_x;
        config.delta_y = config_bcast.delta_y;
        config.delta_z = config_bcast.delta_z;
        config.dt = config_bcast.dt;
        config.c = config_bcast.c;
        config.epsilon0 = config_bcast.epsilon0;
        config.mu0 = config_bcast.mu0;
        config.num_sources = config_bcast.num_sources;
		config.num_materials = config_bcast.num_materials; // 添加材料数量
		config.output_slice_axis = config_bcast.output_slice_axis; // 切片轴
		config.output_slice_position = config_bcast.output_slice_position; // 切片位置
		config.sources = NULL;
		config.materials = NULL; // 初始化材料指针
		// 监测点配置
		config.monitor_enabled = config_bcast.monitor_enabled;
		config.monitor_x = config_bcast.monitor_x;
		config.monitor_y = config_bcast.monitor_y;
		config.monitor_z = config_bcast.monitor_z;
		config.monitor_atom = config_bcast.monitor_atom;
		config.monitor_dir = config_bcast.monitor_dir;
		config.full_monitor_enabled = config_bcast.full_monitor_enabled;
		strncpy(config.monitor_output_file, "monitor_point.dat", sizeof(config.monitor_output_file));
		strncpy(config.full_monitor_output_file, "full_monitor.dat", sizeof(config.full_monitor_output_file));
		config.unique_source_frequency = config_bcast.unique_source_frequency;
	}

	// ==== 广播 output_component ====
	int comp_len = 0;
	if (rank == 0) {
		comp_len = strlen(config.output_component) + 1; // 包含结束符
	}
	MPI_Bcast(&comp_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0) {
		// 确保缓冲区足够大
		if (comp_len > sizeof(config.output_component)) {
			fprintf(stderr, "Error: output_component buffer too small on rank %d\n", rank);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	MPI_Bcast(config.output_component, comp_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // 广播介质文件名
	if (rank == 0) printf(" .");
	int file_len = 0;
    if (rank == 0) {
        file_len = strlen(config.medium_file) + 1;
    }
	if (rank == 0) printf(" .");
	MPI_Bcast(&file_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank == 0) printf(" .");

    if (rank != 0) {
        strncpy(config.medium_file, "medium_map.bin", sizeof(config.medium_file));
    }
	if (rank == 0) printf(" .");
	MPI_Bcast(config.medium_file, file_len, MPI_CHAR, 0, MPI_COMM_WORLD);
	if (rank == 0) printf(" .");

    // ==== 广播源信息 ====
    if (config.num_sources > 0) {
        if (rank != 0) {
            config.sources = tracked_malloc(config.num_sources * sizeof(Source));
            if (!config.sources) {
                fprintf(stderr, "Memory allocation failed\n");
                exit(EXIT_FAILURE);
            }
        }
        MPI_Bcast(config.sources, config.num_sources * sizeof(Source), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) printf(" .\n");
	MPI_Barrier(MPI_COMM_WORLD);

	// ===== 新增：0号进程先读取Born文件 =====
	if (rank == 0) {
		for (int i = 0; i < config.num_materials; i++) {
			int natom = config.materials[i].natom;
			if (natom > 0 &&
				config.materials[i].born_filename[0] != '\0') {
				printf("read BORN charges\n");

				int born_size = 3 * natom * 3;
				config.materials[i].born = (double*)tracked_malloc(born_size * sizeof(double));

				// 新增：处理介电张量读取
				DielectricTensor new_eps;
				if (!read_born_from_file(config.materials[i].born_filename,
					config.materials[i].born,
					natom,
					&new_eps,
					config.materials[i].use_born_file_for_eps)) {
					fprintf(stderr, "Failed to read Born charges from %s\n",
						config.materials[i].born_filename);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				if (config.materials[i].switchxz == 1) {	// 调换XZ轴
					for (int iatom = 0; iatom < natom; iatom++) {
						double temp = config.materials[i].born[iatom * 9 + 0 * 3 + 0];
						config.materials[i].born[iatom * 9 + 0 * 3 + 0] = config.materials[i].born[iatom * 9 + 2 * 3 + 2];
						config.materials[i].born[iatom * 9 + 2 * 3 + 2] = temp;
						temp = config.materials[i].born[iatom * 9 + 0 * 3 + 1];
						config.materials[i].born[iatom * 9 + 0 * 3 + 1] = config.materials[i].born[iatom * 9 + 2 * 3 + 1];
						config.materials[i].born[iatom * 9 + 2 * 3 + 1] = temp;
						temp = config.materials[i].born[iatom * 9 + 0 * 3 + 2];
						config.materials[i].born[iatom * 9 + 0 * 3 + 2] = config.materials[i].born[iatom * 9 + 2 * 3 + 0];
						config.materials[i].born[iatom * 9 + 2 * 3 + 0] = temp;
						temp = config.materials[i].born[iatom * 9 + 1 * 3 + 0];
						config.materials[i].born[iatom * 9 + 1 * 3 + 0] = config.materials[i].born[iatom * 9 + 1 * 3 + 2];
						config.materials[i].born[iatom * 9 + 1 * 3 + 2] = temp;
					}
					// born[iatom*9+idir*3+jdir]
				}

				// 如果需要从born文件读取介电张量
				if (config.materials[i].use_born_file_for_eps) {
					config.materials[i].eps = new_eps;
					if (config.materials[i].switchxz == 1) {	// 调换XZ轴
						double temp = config.materials[i].eps.xx;
						config.materials[i].eps.xx = config.materials[i].eps.zz;
						config.materials[i].eps.zz = temp;
						temp = config.materials[i].eps.xy;
						config.materials[i].eps.xy = config.materials[i].eps.zy;
						config.materials[i].eps.zy = temp;
						temp = config.materials[i].eps.xz;
						config.materials[i].eps.xz = config.materials[i].eps.zx;
						config.materials[i].eps.zx = temp;
						temp = config.materials[i].eps.yx;
						config.materials[i].eps.yx = config.materials[i].eps.yz;
						config.materials[i].eps.yz = temp;
						temp = config.materials[i].eps.xxi;
						config.materials[i].eps.xxi = config.materials[i].eps.zzi;
						config.materials[i].eps.zzi = temp;
						temp = config.materials[i].eps.xyi;
						config.materials[i].eps.xyi = config.materials[i].eps.zyi;
						config.materials[i].eps.zyi = temp;
						temp = config.materials[i].eps.yxi;
						config.materials[i].eps.yxi = config.materials[i].eps.yzi;
						config.materials[i].eps.yzi = temp;
						temp = config.materials[i].eps.xzi;
						config.materials[i].eps.xzi = config.materials[i].eps.zxi;
						config.materials[i].eps.zxi = temp;
					}
					config.materials[i].eps_inv = inverse_dielectric_tensor(config.materials[i].eps);
				}
			}
			if (natom > 0) {
				printf("read QeZV\n");
				if (config.materials[i].QeZV_filename_set) {
					int size = natom * 9;
					config.materials[i].QeZV = (double*)tracked_malloc(size * sizeof(double));
					if (!config.materials[i].QeZV) {
						fprintf(stderr, "Memory allocation failed for QeZV\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (!read_data(config.materials[i].QeZV_filename, config.materials[i].QeZV, natom * 9)) {
						printf("Error: Failed to read QeZV from %s\n", config.materials[i].QeZV_filename);
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (config.materials[i].switchxz == 1) {	// 调换XZ轴
						for (int iatom = 0; iatom < natom; iatom++) {
							double temp = config.materials[i].QeZV[iatom * 9 + 0 * 3 + 0];
							config.materials[i].QeZV[iatom * 9 + 0 * 3 + 0] = config.materials[i].QeZV[iatom * 9 + 2 * 3 + 2];
							config.materials[i].QeZV[iatom * 9 + 2 * 3 + 2] = temp;
							temp = config.materials[i].QeZV[iatom * 9 + 0 * 3 + 1];
							config.materials[i].QeZV[iatom * 9 + 0 * 3 + 1] = config.materials[i].QeZV[iatom * 9 + 2 * 3 + 1];
							config.materials[i].QeZV[iatom * 9 + 2 * 3 + 1] = temp;
							temp = config.materials[i].QeZV[iatom * 9 + 0 * 3 + 2];
							config.materials[i].QeZV[iatom * 9 + 0 * 3 + 2] = config.materials[i].QeZV[iatom * 9 + 2 * 3 + 0];
							config.materials[i].QeZV[iatom * 9 + 2 * 3 + 0] = temp;
							temp = config.materials[i].QeZV[iatom * 9 + 1 * 3 + 0];
							config.materials[i].QeZV[iatom * 9 + 1 * 3 + 0] = config.materials[i].QeZV[iatom * 9 + 1 * 3 + 2];
							config.materials[i].QeZV[iatom * 9 + 1 * 3 + 2] = temp;
						}
						//QeZV[iatom * 9 + iE * 3 + idir]
					}
				}
				else {
					printf("Error: QeZV is not set!\n");
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
			if (natom > 0) {
				printf("Read QeZM\n");
				if (config.materials[i].QeZM_filename_set) {
					int size = natom * 9;
					config.materials[i].QeZM = (double*)tracked_malloc(size * sizeof(double));
					if (!config.materials[i].QeZM) {
						fprintf(stderr, "Memory allocation failed for QeZM\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (!read_data(config.materials[i].QeZM_filename, config.materials[i].QeZM, natom * 9)) {
						printf("Error: Failed to read QeZM from %s\n", config.materials[i].QeZM_filename);
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (config.materials[i].switchxz == 1) {	// 调换XZ轴
						for (int jatom = 0; jatom < natom; jatom++) {
							double temp = config.materials[i].QeZM[jatom * 9 + 0 * 3 + 0];
							config.materials[i].QeZM[jatom * 9 + 0 * 3 + 0] = config.materials[i].QeZM[jatom * 9 + 2 * 3 + 2];
							config.materials[i].QeZM[jatom * 9 + 2 * 3 + 2] = temp;
							temp = config.materials[i].QeZM[jatom * 9 + 0 * 3 + 1];
							config.materials[i].QeZM[jatom * 9 + 0 * 3 + 1] = config.materials[i].QeZM[jatom * 9 + 2 * 3 + 1];
							config.materials[i].QeZM[jatom * 9 + 2 * 3 + 1] = temp;
							temp = config.materials[i].QeZM[jatom * 9 + 0 * 3 + 2];
							config.materials[i].QeZM[jatom * 9 + 0 * 3 + 2] = config.materials[i].QeZM[jatom * 9 + 2 * 3 + 0];
							config.materials[i].QeZM[jatom * 9 + 2 * 3 + 0] = temp;
							temp = config.materials[i].QeZM[jatom * 9 + 1 * 3 + 0];
							config.materials[i].QeZM[jatom * 9 + 1 * 3 + 0] = config.materials[i].QeZM[jatom * 9 + 1 * 3 + 2];
							config.materials[i].QeZM[jatom * 9 + 1 * 3 + 2] = temp;
						}
					}
					// Qe*Z(iatom,Edirection,rdirection)/Mass(iatom)
				}
				else {
					printf("Error: QeZM is not set!\n");
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
			if (natom > 0) {
				printf("Read DMM\n");
				if (config.materials[i].DMM_filename_set) {
					int size = natom * natom * 9;
					config.materials[i].DMM = (double*)tracked_malloc(size * sizeof(double));
					if (!config.materials[i].DMM) {
						fprintf(stderr, "Memory allocation failed for DMM\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (!read_data(config.materials[i].DMM_filename, config.materials[i].DMM, natom * natom * 9)) {
						printf("Error: Failed to read DMM from %s\n", config.materials[i].DMM_filename);
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					if (config.materials[i].switchxz == 1) {	// 调换XZ轴
						for (int iatom = 0; iatom < natom; iatom++) {
							for (int jatom = 0; jatom < natom; jatom++) {
								double temp = config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 0];
								config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 0] = config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 2];
								config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 2] = temp;
								temp = config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 1];
								config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 1] = config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 1];
								config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 1] = temp;
								temp = config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 2];
								config.materials[i].DMM[iatom * 9 * natom + 0 * 3 * natom + jatom * 3 + 2] = config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 0];
								config.materials[i].DMM[iatom * 9 * natom + 2 * 3 * natom + jatom * 3 + 0] = temp;
								temp = config.materials[i].DMM[iatom * 9 * natom + 1 * 3 * natom + jatom * 3 + 0];
								config.materials[i].DMM[iatom * 9 * natom + 1 * 3 * natom + jatom * 3 + 0] = config.materials[i].DMM[iatom * 9 * natom + 1 * 3 * natom + jatom * 3 + 2];
								config.materials[i].DMM[iatom * 9 * natom + 1 * 3 * natom + jatom * 3 + 2] = temp;
							}
						}
						// DMM(iatom*9*natom+idir*3*natom+jatom*3+jdir)
					}
				}
				else {
						printf("Error: DMM is not set!\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
		}
	}

	// 主进程处理介电函数文件
	if (rank == 0) {
		for (int i = 0; i < config.num_materials; i++) {
			if (config.materials[i].use_dielectric_file) {
				DielectricTensor file_eps;
				if (read_dielectric_from_file(config.materials[i].dielectric_filename,
					config.unique_source_frequency, &file_eps)) {
					// 使用文件中的介电函数，覆盖之前的值
					config.materials[i].eps = file_eps;
					if (config.materials[i].switchxz == 1) {	// 调换XZ轴
						double temp = config.materials[i].eps.xx;
						config.materials[i].eps.xx = config.materials[i].eps.zz;
						config.materials[i].eps.zz = temp;
						temp = config.materials[i].eps.xy;
						config.materials[i].eps.xy = config.materials[i].eps.zy;
						config.materials[i].eps.zy = temp;
						temp = config.materials[i].eps.yx;
						config.materials[i].eps.yx = config.materials[i].eps.yz;
						config.materials[i].eps.yz = temp;
						temp = config.materials[i].eps.xz;
						config.materials[i].eps.xz = config.materials[i].eps.zx;
						config.materials[i].eps.zx = temp;
						temp = config.materials[i].eps.xxi;
						config.materials[i].eps.xxi = config.materials[i].eps.zzi;
						config.materials[i].eps.zzi = temp;
						temp = config.materials[i].eps.xyi;
						config.materials[i].eps.xyi = config.materials[i].eps.zyi;
						config.materials[i].eps.zyi = temp;
						temp = config.materials[i].eps.yxi;
						config.materials[i].eps.yxi = config.materials[i].eps.yzi;
						config.materials[i].eps.yzi = temp;
						temp = config.materials[i].eps.xzi;
						config.materials[i].eps.xzi = config.materials[i].eps.zxi;
						config.materials[i].eps.zxi = temp;
					}
					config.materials[i].eps_inv = inverse_dielectric_tensor(config.materials[i].eps);

					printf("Loaded dielectric function for material %d from %s at frequency %.2e Hz\n",	i, config.materials[i].dielectric_filename, config.unique_source_frequency);
				}
				else {
					fprintf(stderr, "Failed to read dielectric function from %s for material %d\n",
						config.materials[i].dielectric_filename, i);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
			if (config.materials[i].use_ADE == 1) {
				config.materials[i].epsr1[0][0] = config.materials[i].eps.xx-EPS_INF; config.materials[i].epsr1[0][1] = config.materials[i].eps.xy;         config.materials[i].epsr1[0][2] = config.materials[i].eps.xz;
				config.materials[i].epsr1[1][0] = config.materials[i].eps.yx;         config.materials[i].epsr1[1][1] = config.materials[i].eps.yy-EPS_INF; config.materials[i].epsr1[1][2] = config.materials[i].eps.yz;
				config.materials[i].epsr1[2][0] = config.materials[i].eps.zx;         config.materials[i].epsr1[2][1] = config.materials[i].eps.zy;         config.materials[i].epsr1[2][2] = config.materials[i].eps.zz-EPS_INF;
				if (config.materials[i].epsr1[0][0] >= 0 || config.materials[i].epsr1[1][1] >= 0 || config.materials[i].epsr1[2][2] >= 0) {
					printf("Error: EPS_INF is too small for material %d\n", i);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				if (inverse(config.materials[i].epsr1, config.materials[i].epsr1_inv) != 0) {
					printf("Error: epsr1 can not be inversed\n");
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				config.materials[i].gama[0][0] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[0][1] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[0][2] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[1][0] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[1][1] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[1][2] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[2][0] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[2][1] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].gama[2][2] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
				config.materials[i].wp2[0][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xxi + config.materials[i].gama[0][1] * config.materials[i].eps.yxi + config.materials[i].gama[0][2] * config.materials[i].eps.zxi);
				config.materials[i].wp2[0][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xyi + config.materials[i].gama[0][1] * config.materials[i].eps.yyi + config.materials[i].gama[0][2] * config.materials[i].eps.zyi);
				config.materials[i].wp2[0][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xzi + config.materials[i].gama[0][1] * config.materials[i].eps.yzi + config.materials[i].gama[0][2] * config.materials[i].eps.zzi);
				config.materials[i].wp2[1][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xxi + config.materials[i].gama[1][1] * config.materials[i].eps.yxi + config.materials[i].gama[1][2] * config.materials[i].eps.zxi);
				config.materials[i].wp2[1][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xyi + config.materials[i].gama[1][1] * config.materials[i].eps.yyi + config.materials[i].gama[1][2] * config.materials[i].eps.zyi);
				config.materials[i].wp2[1][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xzi + config.materials[i].gama[1][1] * config.materials[i].eps.yzi + config.materials[i].gama[1][2] * config.materials[i].eps.zzi);
				config.materials[i].wp2[2][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xxi + config.materials[i].gama[2][1] * config.materials[i].eps.yxi + config.materials[i].gama[2][2] * config.materials[i].eps.zxi);
				config.materials[i].wp2[2][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xyi + config.materials[i].gama[2][1] * config.materials[i].eps.yyi + config.materials[i].gama[2][2] * config.materials[i].eps.zyi);
				config.materials[i].wp2[2][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xzi + config.materials[i].gama[2][1] * config.materials[i].eps.yzi + config.materials[i].gama[2][2] * config.materials[i].eps.zzi);
				for (int ii = 0; ii < 3; ii++) {
					for (int jj = 0; jj < 3; jj++) {
						config.materials[i].gdt2p1[ii][jj] = +config.materials[i].gama[ii][jj] * config.dt / 2.0;
						config.materials[i].gdt2m1[ii][jj] = -config.materials[i].gama[ii][jj] * config.dt / 2.0;
						if (ii == jj) {
							config.materials[i].gdt2p1[ii][jj] += 1.0;
							config.materials[i].gdt2m1[ii][jj] += 1.0;
						}
					}
				}
				if (inverse(config.materials[i].gdt2m1, config.materials[i].gdt2m1_inv) != 0) {
					printf("Error: gdt2m1 can not be inversed\n");
					printf("gdt2m1:%1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n", config.materials[i].gdt2m1[0][0], config.materials[i].gdt2m1[0][1], config.materials[i].gdt2m1[0][2], config.materials[i].gdt2m1[1][0], config.materials[i].gdt2m1[1][1], config.materials[i].gdt2m1[1][2], config.materials[i].gdt2m1[2][0], config.materials[i].gdt2m1[2][1], config.materials[i].gdt2m1[2][2]);
					fflush(stdout);
				}
				if (inverse(config.materials[i].gdt2p1, config.materials[i].gdt2p1_inv) != 0) {
					printf("Error: gdt2p1 can not be inversed\n");
					printf("gdt2p1:%1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n", config.materials[i].gdt2p1[0][0], config.materials[i].gdt2p1[0][1], config.materials[i].gdt2p1[0][2], config.materials[i].gdt2p1[1][0], config.materials[i].gdt2p1[1][1], config.materials[i].gdt2p1[1][2], config.materials[i].gdt2p1[2][0], config.materials[i].gdt2p1[2][1], config.materials[i].gdt2p1[2][2]);
					fflush(stdout);
				}
			}
		}
	}

	// 广播材料信息
	if (config.num_materials > 0) {
		if (rank != 0) {
			// 从进程分配内存
			config.materials = tracked_malloc(config.num_materials * sizeof(Material));
			if (!config.materials) {
				fprintf(stderr, "Memory allocation failed\n");
				exit(EXIT_FAILURE);
			}
			// 初始化材料结构体，避免未定义行为
			for (int i = 0; i < config.num_materials; i++) {
				memset(&config.materials[i], 0, sizeof(Material));
				// 确保所有指针初始为NULL
				config.materials[i].born = NULL;
				config.materials[i].QeZV = NULL;
				config.materials[i].QeZM = NULL;
				config.materials[i].DMM = NULL;
				// 确保标志位也初始化
				config.materials[i].QeZV_filename_set = 0;
				config.materials[i].QeZM_filename_set = 0;
				config.materials[i].DMM_filename_set = 0;
			}
		}

		// 广播每个材料的数据
		for (int i = 0; i < config.num_materials; i++) {
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

			// 广播材料名称长度和内容
			int name_len = 0;
			if (rank == 0) {
				name_len = strlen(config.materials[i].name) + 1;
			}
			MPI_Bcast(&name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (rank != 0) {
				if (sizeof(config.materials[i].name) < name_len) {
					fprintf(stderr, "Material name buffer too small\n");
					exit(EXIT_FAILURE);
				}
			}
			MPI_Bcast(config.materials[i].name, name_len, MPI_CHAR, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].use_dielectric_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].switchxz, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (config.materials[i].use_dielectric_file) {
				int dielectric_filename_len = 0;
				if (rank == 0) {
					dielectric_filename_len = strlen(config.materials[i].dielectric_filename) + 1;
				}
				MPI_Bcast(&dielectric_filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
				MPI_Bcast(config.materials[i].dielectric_filename, dielectric_filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);
			}
			// 广播介电张量和电导率
			MPI_Bcast(&config.materials[i].eps, sizeof(DielectricTensor), MPI_BYTE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].epsr1[0][0], 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].epsr1_inv[0][0], 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].sigma_m, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// 广播born_filename（无论是否有born数组）
			int born_filename_len = 0;
			if (rank == 0) {
				born_filename_len = strlen(config.materials[i].born_filename) + 1;
			}
			MPI_Bcast(&born_filename_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (rank != 0) {
				if (born_filename_len > sizeof(config.materials[i].born_filename)) {
					fprintf(stderr, "Rank %d: born_filename buffer too small\n", rank);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
			}
			MPI_Bcast(config.materials[i].born_filename, born_filename_len, MPI_CHAR, 0, MPI_COMM_WORLD);

			// 广播原子数（已存在）
			MPI_Bcast(&config.materials[i].natom, 1, MPI_INT, 0, MPI_COMM_WORLD);

			// 广播use_born_file_for_eps标志
			MPI_Bcast(&config.materials[i].use_born_file_for_eps, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].use_ADE, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].anglez, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].anglex, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			// 广播标志
			MPI_Bcast(&config.materials[i].QeZV_filename_set, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].QeZM_filename_set, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&config.materials[i].DMM_filename_set, 1, MPI_INT, 0, MPI_COMM_WORLD);

			// 新增：广播born数组存在标志（基于natom而非born指针）
			int has_born = 0;
			if (rank == 0) {
				has_born = (config.materials[i].natom > 0 &&
					config.materials[i].born_filename[0] != '\0') ? 1 : 0;
			}
			MPI_Bcast(&has_born, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (has_born) {
				int born_size = 0;
				if (rank == 0) {
					born_size = 3 * config.materials[i].natom * 3;
					// 验证born数组是否确实有足够的数据
					if (config.materials[i].born == NULL) {
						fprintf(stderr, "Error: born array is NULL despite has_born=1\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
				}
				MPI_Bcast(&born_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

				// 检查born_size是否合理
				if (born_size <= 0) {
					if (rank == 0) {
						fprintf(stderr, "Error: invalid born_size=%d for material %d\n", born_size, i);
					}
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}

				if (rank != 0) {
					config.materials[i].born = (double*)tracked_malloc(born_size * sizeof(double));
					if (!config.materials[i].born) {
						fprintf(stderr, "Memory allocation failed for born array\n");
						exit(EXIT_FAILURE);
					}
					// 分配QeZV
					if (config.materials[i].QeZV_filename_set) {
						int size = config.materials[i].natom * 9;
						config.materials[i].QeZV = (double*)tracked_malloc(size * sizeof(double));
						if (!config.materials[i].QeZV) {
							fprintf(stderr, "Memory allocation failed for QeZV\n");
							exit(EXIT_FAILURE);
						}
					}
					// 分配QeZM
					if (config.materials[i].QeZM_filename_set) {
						int size = config.materials[i].natom * 9;
						config.materials[i].QeZM = (double*)tracked_malloc(size * sizeof(double));
						if (!config.materials[i].QeZM) {
							fprintf(stderr, "Memory allocation failed for QeZM\n");
							exit(EXIT_FAILURE);
						}
					}
					// 分配DMM
					if (config.materials[i].DMM_filename_set) {
						int size = config.materials[i].natom * config.materials[i].natom * 9;
						config.materials[i].DMM = (double*)tracked_malloc(size * sizeof(double));
						if (!config.materials[i].DMM) {
							fprintf(stderr, "Memory allocation failed for DMM\n");
							exit(EXIT_FAILURE);
						}
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(config.materials[i].born, born_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				MPI_Barrier(MPI_COMM_WORLD);
				if (rank == 0 && config.materials[i].QeZV == NULL) {
					fprintf(stderr, "Error: QeZV is NULL on rank 0 for material %d\n", i);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				if (config.materials[i].QeZV_filename_set) {
					MPI_Bcast(config.materials[i].QeZV, config.materials[i].natom * 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				}
				MPI_Barrier(MPI_COMM_WORLD);
				if (config.materials[i].QeZV == NULL) {
					fprintf(stderr, "Rank %d: QeZV is NULL after broadcast for material %d\n", rank, i);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				if (config.materials[i].QeZM_filename_set) {
					MPI_Bcast(config.materials[i].QeZM, config.materials[i].natom * 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				}
				MPI_Barrier(MPI_COMM_WORLD);
				if (config.materials[i].DMM_filename_set) {
					MPI_Bcast(config.materials[i].DMM, config.materials[i].natom * config.materials[i].natom * 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				}
			}

			// 从进程计算逆矩阵
			if (rank != 0) {
				config.materials[i].eps_inv = inverse_dielectric_tensor(config.materials[i].eps);
				if (config.materials[i].use_ADE == 1) {
					if (inverse(config.materials[i].epsr1, config.materials[i].epsr1_inv) != 0) {
						printf("epsr1:%g  %g  %g\n      %g  %g  %g\n      %g  %g  %g\n", config.materials[i].epsr1[0][0], config.materials[i].epsr1[0][1], config.materials[i].epsr1[0][2], config.materials[i].epsr1[1][0], config.materials[i].epsr1[1][1], config.materials[i].epsr1[1][2], config.materials[i].epsr1[2][0], config.materials[i].epsr1[2][1], config.materials[i].epsr1[2][2]);
						printf("Error: epsr1 can not be inversed\n");
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
					}
					config.materials[i].gama[0][0] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[0][1] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[0][2] = -(config.materials[i].eps.xxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.xyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.xzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[1][0] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[1][1] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[1][2] = -(config.materials[i].eps.yxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.yyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.yzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[2][0] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][0] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][0] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][0]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[2][1] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][1] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][1] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][1]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].gama[2][2] = -(config.materials[i].eps.zxi * config.materials[i].epsr1_inv[0][2] + config.materials[i].eps.zyi * config.materials[i].epsr1_inv[1][2] + config.materials[i].eps.zzi * config.materials[i].epsr1_inv[2][2]) * config.unique_source_frequency * 2.0 * M_PI;
					config.materials[i].wp2[0][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xxi + config.materials[i].gama[0][1] * config.materials[i].eps.yxi + config.materials[i].gama[0][2] * config.materials[i].eps.zxi);
					config.materials[i].wp2[0][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xyi + config.materials[i].gama[0][1] * config.materials[i].eps.yyi + config.materials[i].gama[0][2] * config.materials[i].eps.zyi);
					config.materials[i].wp2[0][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[0][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[0][0] * config.materials[i].eps.xzi + config.materials[i].gama[0][1] * config.materials[i].eps.yzi + config.materials[i].gama[0][2] * config.materials[i].eps.zzi);
					config.materials[i].wp2[1][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xxi + config.materials[i].gama[1][1] * config.materials[i].eps.yxi + config.materials[i].gama[1][2] * config.materials[i].eps.zxi);
					config.materials[i].wp2[1][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xyi + config.materials[i].gama[1][1] * config.materials[i].eps.yyi + config.materials[i].gama[1][2] * config.materials[i].eps.zyi);
					config.materials[i].wp2[1][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[1][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[1][0] * config.materials[i].eps.xzi + config.materials[i].gama[1][1] * config.materials[i].eps.yzi + config.materials[i].gama[1][2] * config.materials[i].eps.zzi);
					config.materials[i].wp2[2][0] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][0] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xxi + config.materials[i].gama[2][1] * config.materials[i].eps.yxi + config.materials[i].gama[2][2] * config.materials[i].eps.zxi);
					config.materials[i].wp2[2][1] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][1] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xyi + config.materials[i].gama[2][1] * config.materials[i].eps.yyi + config.materials[i].gama[2][2] * config.materials[i].eps.zyi);
					config.materials[i].wp2[2][2] = -config.unique_source_frequency * config.unique_source_frequency * 4.0 * M_PI * M_PI * config.materials[i].epsr1[2][2] + config.unique_source_frequency * 2.0 * M_PI * (config.materials[i].gama[2][0] * config.materials[i].eps.xzi + config.materials[i].gama[2][1] * config.materials[i].eps.yzi + config.materials[i].gama[2][2] * config.materials[i].eps.zzi);
					for (int ii = 0; ii < 3; ii++) {
						for (int jj = 0; jj < 3; jj++) {
							config.materials[i].gdt2p1[ii][jj] = +config.materials[i].gama[ii][jj] * config.dt / 2.0;
							config.materials[i].gdt2m1[ii][jj] = -config.materials[i].gama[ii][jj] * config.dt / 2.0;
							if (ii == jj) {
								config.materials[i].gdt2p1[ii][jj] += 1.0;
								config.materials[i].gdt2m1[ii][jj] += 1.0;
							}
						}
					}
					if (inverse(config.materials[i].gdt2m1, config.materials[i].gdt2m1_inv) != 0) {
						printf("Error: gdt2m1 can not be inversed\n");
						printf("gdt2m1:%1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n", config.materials[i].gdt2m1[0][0], config.materials[i].gdt2m1[0][1], config.materials[i].gdt2m1[0][2], config.materials[i].gdt2m1[1][0], config.materials[i].gdt2m1[1][1], config.materials[i].gdt2m1[1][2], config.materials[i].gdt2m1[2][0], config.materials[i].gdt2m1[2][1], config.materials[i].gdt2m1[2][2]);
						fflush(stdout);
					}
					if (inverse(config.materials[i].gdt2p1, config.materials[i].gdt2p1_inv) != 0) {
						printf("Error: gdt2p1 can not be inversed\n");
						printf("gdt2p1:%1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n      %1.10g  %1.10g  %1.10g\n", config.materials[i].gdt2p1[0][0], config.materials[i].gdt2p1[0][1], config.materials[i].gdt2p1[0][2], config.materials[i].gdt2p1[1][0], config.materials[i].gdt2p1[1][1], config.materials[i].gdt2p1[1][2], config.materials[i].gdt2p1[2][0], config.materials[i].gdt2p1[2][1], config.materials[i].gdt2p1[2][2]);
						fflush(stdout);
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	// 计算域分解 (z方向)
int base_nz = config.nz / size;
int remainder = config.nz % size;
int local_z_start = 0;

// 计算前rank个进程占用的网格
for (int r = 0; r < rank; r++) {
    local_z_start += (r < remainder) ? (base_nz + 1) : base_nz;
}

// 计算当前进程的网格大小
int local_nz = (rank < remainder) ? (base_nz + 1) : base_nz;
    
// 检查网格有效性
if (local_nz < 1) {
    fprintf(stderr, "Process %d has no grid points! Reduce MPI processes.\n", rank);
}

    // Halo区域大小
    const int halo = 1;
    
    // 打印配置信息 (仅一个进程)
	int print_rank = size - 1;
    if (rank == print_rank) {
        printf("==========================Simulation Configuration===========================\n");
        printf("  Grid size: %d x %d x %d\n", config.nx, config.ny, config.nz);
        printf("  Spatial steps: dx=%.4e m, dy=%.4e m, dz=%.4e m\n", 
               config.delta_x, config.delta_y, config.delta_z);
        printf("  Time steps: %d, dt=%.4e s\n", config.tsteps, config.dt);
        printf("  PML thickness: %d cells\n", config.pml_thickness);
		printf("  PML reflection: %.4g\n", config.pml_reflection);  // 显示反射率
		printf("  Medium file: %s\n", config.medium_file);
        printf("  Sources: %d\n", config.num_sources);
        
        for (int i = 0; i < config.num_sources; i++) {
            Source *src = &config.sources[i];
            const char *type_str = "gaussian";
			if (src->type == 1) type_str = "sinusoidal";
			else if (src->type == 2) type_str = "ricker";
			else if (src->type == 3) type_str = "dipole";
            
            printf("    Source %d: type=%s, position=(%d,%d,%d), amplitude=%.2f\n",
                   i+1, type_str, src->position[0], src->position[1], src->position[2], src->amplitude);
            printf("               component=%c, frequency=%.2e Hz, T=%.2e s, t0=%.2e s, tau=%.2e s\n",
                   src->component, src->frequency, 1/src->frequency, src->t0, src->tau);
			printf("               wavelength in vacuum: %.2e m\n", 3e8 / src->frequency);
        }
		printf("Output %.2s at time interval: %.2e s\n", config.output_component, config.output_interval* config.dt);
		printf("Output slice: axis=%c, position=%d\n", config.output_slice_axis, config.output_slice_position);
		if (config.monitor_enabled) {
			const char* dir_str = (config.monitor_dir == 0) ? "x" :
				(config.monitor_dir == 1) ? "y" : "z";
			printf("\n  Monitor point: (%d, %d, %d), Atom %d, Direction %s\n",
				config.monitor_x, config.monitor_y, config.monitor_z,
				config.monitor_atom, dir_str);
			printf("  Monitor output file: %s\n", config.monitor_output_file);
			}
			if (config.full_monitor_enabled) {
			printf("  Full monitor output file: %s\n", config.full_monitor_output_file);
			}
			}
    
    // 创建材料库 (所有进程)
	Material* materials = config.materials;
	int num_materials = config.num_materials;

    // 打印材料信息 (仅主进程)
	if (rank == print_rank) {
		printf("\nMaterial library details:\n");
		for (int i = 0; i < num_materials; i++) {
			Material* mat = &materials[i];
			printf("  Material %d: %s\n", i, mat->name);
			printf("    Dielectric tensor (real):\n");
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.xx, mat->eps.xy, mat->eps.xz);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.yx, mat->eps.yy, mat->eps.yz);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.zx, mat->eps.zy, mat->eps.zz);
			printf("    Dielectric tensor (imag) should be >= 0:\n");
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.xxi, mat->eps.xyi, mat->eps.xzi);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.yxi, mat->eps.yyi, mat->eps.yzi);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps.zxi, mat->eps.zyi, mat->eps.zzi);
			/*printf("    Inverse Dielectric tensor (real):\n");
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.xx, mat->eps_inv.xy, mat->eps_inv.xz);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.yx, mat->eps_inv.yy, mat->eps_inv.yz);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.zx, mat->eps_inv.zy, mat->eps_inv.zz);
			printf("    InverseDielectric tensor (imag, useless):\n");
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.xxi, mat->eps_inv.xyi, mat->eps_inv.xzi);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.yxi, mat->eps_inv.yyi, mat->eps_inv.yzi);
			printf("      [%.4f, %.4f, %.4f]\n",
				mat->eps_inv.zxi, mat->eps_inv.zyi, mat->eps_inv.zzi);*/
			printf("    Conductivity: σ = %.4e S/m, σ_m = %.4e Ω/m\n", mat->sigma, mat->sigma_m);
			printf("    use_ADE: %d\n", mat->use_ADE);
			printf("    Switchxz: %d\n", mat->switchxz);
			if (mat->natom == 0 && mat->use_dielectric_file == 0 && mat->switchxz != 0) {
				printf("Error: Please switch X and Z by hand for this simple material\n");
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
			printf("    Z rotate angle %.2f rad %.2f deg\n", mat->anglez, mat->anglez / M_PI * 180.0);
			printf("    X rotate angle %.2f rad %.2f deg\n", mat->anglex, mat->anglex / M_PI * 180.0);
			if (fabs(mat->anglex) > 1e-10 && fabs(mat->anglez) > 1e-10) {
				printf("Error: Can not rotate along X and Z axis at the same time\n");
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
			printf("    Number of atoms: %d\n", mat->natom);
		}
	}

    // 从文件读取介质分布 (并行读取)
    if (rank == print_rank) {
        printf("Using medium file: %s\n", config.medium_file);
    }
    int ***medium = read_medium_map_parallel(config.medium_file, config.nx, config.ny, config.nz,
                                           local_z_start, local_nz, halo, rank, size);
 
	// 在读取介质分布后初始化 r 和 v
	size_t _mem_before_4d = g_total_allocated;
	r = (double****)tracked_malloc(config.nx * sizeof(double***));
	print_alloc(config.nx * sizeof(double***), "r (4D ptr)");
	v = (double****)tracked_malloc(config.nx * sizeof(double***));
	print_alloc(config.nx * sizeof(double***), "v (4D ptr)");
	P_ADE = (double****)tracked_malloc(config.nx * sizeof(double***));
	print_alloc(config.nx * sizeof(double***), "P_ADE (4D ptr)");
	P_ADEo = (double****)tracked_malloc(config.nx * sizeof(double***));
	print_alloc(config.nx * sizeof(double***), "P_ADEo (4D ptr)");
		for (int i = 0; i < config.nx; i++) {
		r[i] = (double***)tracked_malloc(config.ny * sizeof(double**));
		v[i] = (double***)tracked_malloc(config.ny * sizeof(double**));
		P_ADE[i] = (double***)tracked_malloc(config.ny * sizeof(double**));
		P_ADEo[i] = (double***)tracked_malloc(config.ny * sizeof(double**));
		for (int j = 0; j < config.ny; j++) {
			r[i][j] = (double**)tracked_malloc(local_nz * sizeof(double*));
			v[i][j] = (double**)tracked_malloc(local_nz * sizeof(double*));
			P_ADE[i][j] = (double**)tracked_malloc(local_nz * sizeof(double*));
			P_ADEo[i][j] = (double**)tracked_malloc(local_nz * sizeof(double*));
			for (int k = 0; k < local_nz; k++) {
				int mat_id = medium[i][j][k + halo];
				int natom = materials[mat_id].natom;
				int use_ADE = materials[mat_id].use_ADE;
				// 只在 natom > 0 时分配 r 和 v 数组
				if (natom > 0) {
					r[i][j][k] = (double*)tracked_calloc(natom * 3, sizeof(double));
					v[i][j][k] = (double*)tracked_calloc(natom * 3, sizeof(double));
				}
				else {
					r[i][j][k] = NULL;
					v[i][j][k] = NULL;
				}

				// 只在 use_ADE == 1 时分配 P_ADE 数组
				if (use_ADE == 1) {
					P_ADE[i][j][k] = (double*)tracked_calloc(3, sizeof(double));  // 固定为3，不是 use_ADE*3
					P_ADEo[i][j][k] = (double*)tracked_calloc(3, sizeof(double));
				}
				else {
					P_ADE[i][j][k] = NULL;
					P_ADEo[i][j][k] = NULL;
				}
			}
		}
	}
	print_alloc(g_total_allocated - _mem_before_4d, "r/v/P_ADE/P_ADEo");

	// Auto-enable full_monitor if monitor_point is on a material with atoms
	if (config.monitor_enabled && !config.full_monitor_enabled) {
		int mz = config.monitor_z;
		if (mz >= local_z_start && mz < local_z_start + local_nz) {
			int mk = mz - local_z_start + halo;
			if (config.monitor_x >= 0 && config.monitor_x < config.nx &&
				config.monitor_y >= 0 && config.monitor_y < config.ny &&
				mk >= halo && mk < halo + local_nz) {
				int mat_id = medium[config.monitor_x][config.monitor_y][mk];
				if (materials[mat_id].natom > 0) {
					config.full_monitor_enabled = 1;
				}
			}
		}
		// Use Allreduce so the decision from the rank owning the monitor point propagates
		{
			int tmp = config.full_monitor_enabled;
			MPI_Allreduce(&tmp, &config.full_monitor_enabled, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		}
	}

// 初始化PML参数 (每个方向使用不同的delta)
    PML_Params pml_x = init_pml(config.pml_thickness, config.delta_x, config.dt, config.epsilon0, config.mu0, config.pml_reflection);
    PML_Params pml_y = init_pml(config.pml_thickness, config.delta_y, config.dt, config.epsilon0, config.mu0, config.pml_reflection);
    PML_Params pml_z = init_pml(config.pml_thickness, config.delta_z, config.dt, config.epsilon0, config.mu0, config.pml_reflection);
    
    // 分配场分量内存 (带halo)
    track_begin_group();
    Array3D arr_Ex = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Ex");
    Array3D arr_Ey = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Ey");
    Array3D arr_Ez = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Ez");
    track_end_group("E-field (3 comp)");
    track_begin_group();
    Array3D arr_Dx = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Dx");
    Array3D arr_Dy = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Dy");
    Array3D arr_Dz = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Dz");
    track_end_group("D-field (3 comp)");
    
    // 优化17：预分配MPI通信缓冲区（避免每迭代malloc/free）
    int buf_size = config.nx * config.ny * 3;
    double *halo_buf_E_send_lower = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo E send_lower");
    double *halo_buf_E_recv_lower = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo E recv_lower");
    double *halo_buf_E_send_upper = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo E send_upper");
    double *halo_buf_E_recv_upper = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo E recv_upper");
    double *halo_buf_H_send_lower = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo H send_lower");
    double *halo_buf_H_recv_lower = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo H recv_lower");
    double *halo_buf_H_send_upper = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo H send_upper");
    double *halo_buf_H_recv_upper = (double*)tracked_malloc(buf_size * sizeof(double));
    print_alloc(buf_size * sizeof(double), "halo H recv_upper");
    track_begin_group();
    Array3D arr_Hx = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Hx");
    Array3D arr_Hy = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Hy");
    Array3D arr_Hz = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "Hz");
    track_end_group("H-field (3 comp)");
	Array3D arr_E = { 0 };  // 初始化为0
	double*** E = NULL;   // 初始化为NULL

	// 只有在需要输出电场模时才分配内存
	if (strcmp(config.output_component, "E") == 0) {
		arr_E = malloc_3d_halo(config.nx, config.ny, local_nz, halo, "E");
		E = arr_E.ptr_array;
	}
    // 获取指针
    double ***Ex = arr_Ex.ptr_array;
    double ***Ey = arr_Ey.ptr_array;
    double ***Ez = arr_Ez.ptr_array;
    double ***Dx = arr_Dx.ptr_array;
    double ***Dy = arr_Dy.ptr_array;
    double ***Dz = arr_Dz.ptr_array;
    double ***Hx = arr_Hx.ptr_array;
    double ***Hy = arr_Hy.ptr_array;
    double ***Hz = arr_Hz.ptr_array;

    // PML辅助场 (halo = 0)
    Array3D arr_PSI_Dxy = malloc_3d_halo(2 * config.pml_thickness, config.ny, local_nz, 0, "PSI_Dxy");
    Array3D arr_PSI_Dxz = malloc_3d_halo(2 * config.pml_thickness, config.ny, local_nz, 0, "PSI_Dxz");
    Array3D arr_PSI_Dyx = malloc_3d_halo(config.nx, 2 * config.pml_thickness, local_nz, 0, "PSI_Dyx");
    Array3D arr_PSI_Dyz = malloc_3d_halo(config.nx, 2 * config.pml_thickness, local_nz, 0, "PSI_Dyz");
    Array3D arr_PSI_Hxy = malloc_3d_halo(2 * config.pml_thickness, config.ny, local_nz, 0, "PSI_Hxy");
    Array3D arr_PSI_Hxz = malloc_3d_halo(2 * config.pml_thickness, config.ny, local_nz, 0, "PSI_Hxz");
    Array3D arr_PSI_Hyx = malloc_3d_halo(config.nx, 2 * config.pml_thickness, local_nz, 0, "PSI_Hyx");
    Array3D arr_PSI_Hyz = malloc_3d_halo(config.nx, 2 * config.pml_thickness, local_nz, 0, "PSI_Hyz");
    
    // 获取PML场指针
	double ***PSI_Dxy = arr_PSI_Dxy.ptr_array;
    double ***PSI_Dxz = arr_PSI_Dxz.ptr_array;
    double ***PSI_Dyx = arr_PSI_Dyx.ptr_array;
    double ***PSI_Dyz = arr_PSI_Dyz.ptr_array;
    double ***PSI_Hxy = arr_PSI_Hxy.ptr_array;
    double ***PSI_Hxz = arr_PSI_Hxz.ptr_array;
    double ***PSI_Hyx = arr_PSI_Hyx.ptr_array;
    double ***PSI_Hyz = arr_PSI_Hyz.ptr_array;

	Array3D arr_PSI_Dzx = { 0 };
	Array3D arr_PSI_Dzy = { 0 };
	Array3D arr_PSI_Hzx = { 0 };
	Array3D arr_PSI_Hzy = { 0 };
	double*** PSI_Dzx = NULL;
	double*** PSI_Dzy = NULL;
	double*** PSI_Hzx = NULL;
	double*** PSI_Hzy = NULL;
	if (local_z_start < config.pml_thickness || local_z_start + local_nz > config.nz - config.pml_thickness) {
		arr_PSI_Dzx = malloc_3d_halo(config.nx, config.ny, local_nz, 0, "PSI_Dzx");
		arr_PSI_Dzy = malloc_3d_halo(config.nx, config.ny, local_nz, 0, "PSI_Dzy");
		arr_PSI_Hzx = malloc_3d_halo(config.nx, config.ny, local_nz, 0, "PSI_Hzx");
		arr_PSI_Hzy = malloc_3d_halo(config.nx, config.ny, local_nz, 0, "PSI_Hzy");
		PSI_Dzx = arr_PSI_Dzx.ptr_array;
		PSI_Dzy = arr_PSI_Dzy.ptr_array;
		PSI_Hzx = arr_PSI_Hzx.ptr_array;
		PSI_Hzy = arr_PSI_Hzy.ptr_array;
	}

    // 初始化所有场为0
	for (int i = 0; i < config.nx; i++) {
        for (int j = 0; j < config.ny; j++) {
            for (int k = -halo; k < local_nz + halo; k++) {
				Ex[i][j][k] = 0.0; Ey[i][j][k] = 0.0; Ez[i][j][k] = 0.0; if (E) E[i][j][k] = 0.0;
                Dx[i][j][k] = 0.0; Dy[i][j][k] = 0.0; Dz[i][j][k] = 0.0;
                Hx[i][j][k] = 0.0; Hy[i][j][k] = 0.0; Hz[i][j][k] = 0.0;
				if (PSI_Dzx && PSI_Dzy && PSI_Hzx && PSI_Hzy && k >= 0 && k < local_nz) {
					PSI_Dzx[i][j][k] = 0.0; PSI_Dzy[i][j][k] = 0.0;
					PSI_Hzx[i][j][k] = 0.0; PSI_Hzy[i][j][k] = 0.0;
				}
				if (k >= 0 && k < local_nz) {
					int mat_id = medium[i][j][k + halo];
					if (materials[mat_id].use_ADE == 1) {
						for (int id = 0; id < 3; id++) {
							P_ADE[i][j][k][id] = 0.0; 
							P_ADEo[i][j][k][id] = 0.0;
						}
					}
					int natom = materials[mat_id].natom;
					for (int ia = 0; ia < natom; ia++) {
						for (int id = 0; id < 3; id++) {
							r[i][j][k][ia * 3 + id] = 0.0;
							v[i][j][k][ia * 3 + id] = 0.0;
						}
					}
				}
            }
        }
    }
	for (int i = 0; i < 2 * config.pml_thickness; i++) {
		for (int j = 0; j < config.ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				PSI_Dxy[i][j][k] = 0.0;
				PSI_Dxz[i][j][k] = 0.0;
				PSI_Hxy[i][j][k] = 0.0;
				PSI_Hxz[i][j][k] = 0.0;
			}
		}
	}
	for (int i = 0; i < config.nx; i++) {
		for (int j = 0; j < 2 * config.pml_thickness; j++) {
			for (int k = 0; k < local_nz; k++) {
				PSI_Dyx[i][j][k] = 0.0; 
				PSI_Dyz[i][j][k] = 0.0;
				PSI_Hyx[i][j][k] = 0.0; 
				PSI_Hyz[i][j][k] = 0.0;
			}
		}
	}

	if (rank == 0) {
		source_file = fopen("source.txt", "w");
		if (source_file == NULL) {
			fprintf(stderr, "Error opening source.txt for writing\n");
		}
	}

	// 开始计时
    double start_time = MPI_Wtime();
	if(rank == 0) printf("Entering iteration:\n");
	fflush(stdout);
    
	print_mem_total("Before time loop");
    // FDTD主循环
    for (int t = 0; t < config.tsteps; t++) { 
        double current_time = t * config.dt;
        
        // 交换电场边界 (用于磁场更新) - 优化13+17：打包+预分配缓冲区
        exchange_halo_z_vec3_with_buffers(Ex, Ey, Ez, config.nx, config.ny, local_nz, halo, rank, size, halo_buf_E_send_lower, halo_buf_E_recv_lower, halo_buf_E_send_upper, halo_buf_E_recv_upper);
        
		// 1. 更新磁场分量 (H场) - 包括PML修正
        for (int i = 0; i < config.nx - 1; i++) {
            for (int j = 0; j < config.ny - 1; j++) {
                for (int k = 0; k < local_nz ; k++) {

					double PSI_Hxy_val = 0.0, PSI_Hxz_val = 0.0, PSI_Hyx_val = 0.0, PSI_Hyz_val = 0.0, PSI_Hzx_val = 0.0, PSI_Hzy_val = 0.0;
					double bh_x = 1.0, ch_x = config.dt, bh_y = 1.0, ch_y = config.dt, bh_z = 1.0, ch_z = config.dt;
					double kappa_x = 1.0, kappa_y = 1.0, kappa_z = 1.0;

                    // 计算原始场导数 (使用各自的空间步长)
                    double dEy_dz = (Ey[i][j][k+1] - Ey[i][j][k]) / config.delta_z;
                    double dEz_dy = (Ez[i][j+1][k] - Ez[i][j][k]) / config.delta_y;
                    double dEz_dx = (Ez[i+1][j][k] - Ez[i][j][k]) / config.delta_x;
                    double dEx_dz = (Ex[i][j][k+1] - Ex[i][j][k]) / config.delta_z;
                    double dEx_dy = (Ex[i][j+1][k] - Ex[i][j][k]) / config.delta_y;
                    double dEy_dx = (Ey[i+1][j][k] - Ey[i][j][k]) / config.delta_x;
                    
                    // === CPML: H-field PSI update (Roden & Gedney, 2000) ===
                    int global_k = k + local_z_start;
                    
                    // x方向PML (整个域的边界)
                    if (config.pml_thickness > 0 && i < config.pml_thickness) {
                        PML_Params pml = pml_x;
						int idx = i;
						double sx = pml.sigma[idx];
						kappa_x = pml.kappa[idx];
						double ax = pml.alpha[idx];
						double aw = sx / (kappa_x * config.epsilon0) + ax / config.epsilon0;
						bh_x = exp(-aw * config.dt);
						ch_x = (aw > 1e-30) ? sx / (kappa_x * kappa_x * config.epsilon0) * (bh_x - 1.0) / aw : 0.0;
						PSI_Hxy_val = PSI_Hxy[i][j][k];
						PSI_Hxz_val = PSI_Hxz[i][j][k];
						PSI_Hxy[i][j][k] = bh_x * PSI_Hxy_val + ch_x * ( dEz_dx / config.mu0);
						PSI_Hxz[i][j][k] = bh_x * PSI_Hxz_val + ch_x * (-dEy_dx / config.mu0);
					}
                    else if (config.pml_thickness > 0 && i >= config.nx - config.pml_thickness) {
                        PML_Params pml = pml_x;
						int idx = config.nx - 1 - i;
						double sx = pml.sigma[idx];
						kappa_x = pml.kappa[idx];
						double ax = pml.alpha[idx];
						double aw = sx / (kappa_x * config.epsilon0) + ax / config.epsilon0;
						bh_x = exp(-aw * config.dt);
						ch_x = (aw > 1e-30) ? sx / (kappa_x * kappa_x * config.epsilon0) * (bh_x - 1.0) / aw : 0.0;
						PSI_Hxy_val = PSI_Hxy[config.pml_thickness + idx][j][k];
						PSI_Hxz_val = PSI_Hxz[config.pml_thickness + idx][j][k];
						PSI_Hxy[config.pml_thickness + idx][j][k] = bh_x * PSI_Hxy_val + ch_x * ( dEz_dx / config.mu0);
						PSI_Hxz[config.pml_thickness + idx][j][k] = bh_x * PSI_Hxz_val + ch_x * (-dEy_dx / config.mu0);
					}

                    // y方向PML
                    if (config.pml_thickness > 0 && j < config.pml_thickness) {
                        PML_Params pml = pml_y;
						int idx = j;
						double sy = pml.sigma[idx];
						kappa_y = pml.kappa[idx];
						double ay = pml.alpha[idx];
						double aw = sy / (kappa_y * config.epsilon0) + ay / config.epsilon0;
						bh_y = exp(-aw * config.dt);
						ch_y = (aw > 1e-30) ? sy / (kappa_y * kappa_y * config.epsilon0) * (bh_y - 1.0) / aw : 0.0;
						PSI_Hyx_val = PSI_Hyx[i][j][k];
						PSI_Hyz_val = PSI_Hyz[i][j][k];
						PSI_Hyx[i][j][k] = bh_y * PSI_Hyx_val + ch_y * (-dEz_dy / config.mu0);
						PSI_Hyz[i][j][k] = bh_y * PSI_Hyz_val + ch_y * ( dEx_dy / config.mu0);
					}
                    else if (j >= config.ny - config.pml_thickness) {
                        PML_Params pml = pml_y;
						int idx = config.ny - 1 - j;
						double sy = pml.sigma[idx];
						kappa_y = pml.kappa[idx];
						double ay = pml.alpha[idx];
						double aw = sy / (kappa_y * config.epsilon0) + ay / config.epsilon0;
						bh_y = exp(-aw * config.dt);
						ch_y = (aw > 1e-30) ? sy / (kappa_y * kappa_y * config.epsilon0) * (bh_y - 1.0) / aw : 0.0;
						PSI_Hyx_val = PSI_Hyx[i][config.pml_thickness + idx][k];
						PSI_Hyz_val = PSI_Hyz[i][config.pml_thickness + idx][k];
						PSI_Hyx[i][config.pml_thickness + idx][k] = bh_y * PSI_Hyx_val + ch_y * (-dEz_dy / config.mu0);
						PSI_Hyz[i][config.pml_thickness + idx][k] = bh_y * PSI_Hyz_val + ch_y * ( dEx_dy / config.mu0);
					}
                    
                    // z方向PML (仅在整个域边界应用)
					if (global_k < config.pml_thickness || global_k >= config.nz - config.pml_thickness) {
                        PML_Params pml = pml_z;
                        int idx;
                        if (global_k < config.pml_thickness) {
							idx = global_k;
                        } else {
							idx = config.nz - 1 - global_k;
                        }
						double sz = pml.sigma[idx];
						kappa_z = pml.kappa[idx];
						double az = pml.alpha[idx];
						double aw = sz / (kappa_z * config.epsilon0) + az / config.epsilon0;
						bh_z = exp(-aw * config.dt);
						ch_z = (aw > 1e-30) ? sz / (kappa_z * kappa_z * config.epsilon0) * (bh_z - 1.0) / aw : 0.0;
						PSI_Hzx_val = PSI_Hzx[i][j][k];
						PSI_Hzy_val = PSI_Hzy[i][j][k];
					}
                    
                    // === CPML: H-field update (explicit curl/kappa + psi form) ===
					Hx[i][j][k] += config.dt * ((-dEz_dy / config.mu0) / kappa_y + PSI_Hyx_val + ( dEy_dz / config.mu0) / kappa_z + PSI_Hzx_val);
					Hy[i][j][k] += config.dt * (( dEz_dx / config.mu0) / kappa_x + PSI_Hxy_val + (-dEx_dz / config.mu0) / kappa_z + PSI_Hzy_val);
					Hz[i][j][k] += config.dt * ((-dEy_dx / config.mu0) / kappa_x + PSI_Hxz_val + ( dEx_dy / config.mu0) / kappa_y + PSI_Hyz_val);
					if (global_k < config.pml_thickness || global_k >= config.nz - config.pml_thickness) {
						PSI_Hzx[i][j][k] = bh_z * PSI_Hzx_val + ch_z * ( dEy_dz / config.mu0);
						PSI_Hzy[i][j][k] = bh_z * PSI_Hzy_val + ch_z * (-dEx_dz / config.mu0);
					}
				}
            }
        }
        
        // 交换磁场边界 (用于电场更新)
        // 交换磁场边界 (用于电场更新) - 优化13+17：打包+预分配缓冲区
        exchange_halo_z_vec3_with_buffers(Hx, Hy, Hz, config.nx, config.ny, local_nz, halo, rank, size, halo_buf_H_send_lower, halo_buf_H_recv_lower, halo_buf_H_send_upper, halo_buf_H_recv_upper);
        
		// 2. 更新电位移场 (D场) - 包括PML修正
        for (int i = 1; i < config.nx - 1; i++) {
            for (int j = 1; j < config.ny - 1; j++) {
                for (int k = 0; k < local_nz ; k++) {

					double PSI_Dxy_val = 0.0, PSI_Dxz_val = 0.0, PSI_Dyx_val = 0.0, PSI_Dyz_val = 0.0, PSI_Dzx_val = 0.0, PSI_Dzy_val = 0.0;
					double b_x = 1.0, c_x = config.dt, b_y = 1.0, c_y = config.dt, b_z = 1.0, c_z = config.dt;
					double kappa_x = 1.0, kappa_y = 1.0, kappa_z = 1.0;
					
					// 计算原始场导数 (使用各自的空间步长)
                    double dHz_dy = (Hz[i][j][k] - Hz[i][j-1][k]) / config.delta_y;
                    double dHy_dz = (Hy[i][j][k] - Hy[i][j][k-1]) / config.delta_z;
                    double dHx_dz = (Hx[i][j][k] - Hx[i][j][k-1]) / config.delta_z;
                    double dHz_dx = (Hz[i][j][k] - Hz[i-1][j][k]) / config.delta_x;
                    double dHy_dx = (Hy[i][j][k] - Hy[i-1][j][k]) / config.delta_x;
                    double dHx_dy = (Hx[i][j][k] - Hx[i][j-1][k]) / config.delta_y;
                    
                    // === CPML: D-field PSI update (Roden & Gedney, 2000) ===
                    int global_k = k + local_z_start;
                    
                    // x方向PML
                    if (config.pml_thickness > 0 && i < config.pml_thickness) {
                        PML_Params pml = pml_x;
						int idx = i;
						double sx = pml.sigma[idx];
						kappa_x = pml.kappa[idx];
						double ax = pml.alpha[idx];
						double aw = sx / (kappa_x * config.epsilon0) + ax / config.epsilon0;
						b_x = exp(-aw * config.dt);
						c_x = (aw > 1e-30) ? sx / (kappa_x * kappa_x * config.epsilon0) * (b_x - 1.0) / aw : 0.0;
						PSI_Dxy_val = PSI_Dxy[i][j][k];
						PSI_Dxz_val = PSI_Dxz[i][j][k];
						PSI_Dxy[i][j][k] = b_x * PSI_Dxy_val + c_x * (-dHz_dx);
						PSI_Dxz[i][j][k] = b_x * PSI_Dxz_val + c_x * ( dHy_dx);
					}
                    else if (config.pml_thickness > 0 && i >= config.nx - config.pml_thickness) {
                        PML_Params pml = pml_x;
						int idx = config.nx - 1 - i;
						double sx = pml.sigma[idx];
						kappa_x = pml.kappa[idx];
						double ax = pml.alpha[idx];
						double aw = sx / (kappa_x * config.epsilon0) + ax / config.epsilon0;
						b_x = exp(-aw * config.dt);
						c_x = (aw > 1e-30) ? sx / (kappa_x * kappa_x * config.epsilon0) * (b_x - 1.0) / aw : 0.0;
						PSI_Dxy_val = PSI_Dxy[config.pml_thickness + idx][j][k];
						PSI_Dxz_val = PSI_Dxz[config.pml_thickness + idx][j][k];
						PSI_Dxy[config.pml_thickness + idx][j][k] = b_x * PSI_Dxy_val + c_x * (-dHz_dx);
						PSI_Dxz[config.pml_thickness + idx][j][k] = b_x * PSI_Dxz_val + c_x * ( dHy_dx);
					}
                    
                    // y方向PML
                    if (config.pml_thickness > 0 && j < config.pml_thickness) {
                        PML_Params pml = pml_y;
						int idx = j;
						double sy = pml.sigma[idx];
						kappa_y = pml.kappa[idx];
						double ay = pml.alpha[idx];
						double aw = sy / (kappa_y * config.epsilon0) + ay / config.epsilon0;
						b_y = exp(-aw * config.dt);
						c_y = (aw > 1e-30) ? sy / (kappa_y * kappa_y * config.epsilon0) * (b_y - 1.0) / aw : 0.0;
						PSI_Dyx_val = PSI_Dyx[i][j][k];
						PSI_Dyz_val = PSI_Dyz[i][j][k];
						PSI_Dyx[i][j][k] = b_y * PSI_Dyx_val + c_y * ( dHz_dy);
						PSI_Dyz[i][j][k] = b_y * PSI_Dyz_val + c_y * (-dHx_dy);
					}
                    else if (j >= config.ny - config.pml_thickness) {
                        PML_Params pml = pml_y;
						int idx = config.ny - 1 - j;
						double sy = pml.sigma[idx];
						kappa_y = pml.kappa[idx];
						double ay = pml.alpha[idx];
						double aw = sy / (kappa_y * config.epsilon0) + ay / config.epsilon0;
						b_y = exp(-aw * config.dt);
						c_y = (aw > 1e-30) ? sy / (kappa_y * kappa_y * config.epsilon0) * (b_y - 1.0) / aw : 0.0;
						PSI_Dyx_val = PSI_Dyx[i][config.pml_thickness + idx][k];
						PSI_Dyz_val = PSI_Dyz[i][config.pml_thickness + idx][k];
						PSI_Dyx[i][config.pml_thickness + idx][k] = b_y * PSI_Dyx_val + c_y * (dHz_dy);
						PSI_Dyz[i][config.pml_thickness + idx][k] = b_y * PSI_Dyz_val + c_y * (-dHx_dy);
					}
                    
                    // z方向PML
					if (global_k < config.pml_thickness || global_k >= config.nz - config.pml_thickness) {
                        PML_Params pml = pml_z;
                        int idx;
                        if (global_k < config.pml_thickness) {
							idx = global_k;
                        } else {
							idx = config.nz - 1 - global_k;
                        }
						double sz = pml.sigma[idx];
						kappa_z = pml.kappa[idx];
						double az = pml.alpha[idx];
						double aw = sz / (kappa_z * config.epsilon0) + az / config.epsilon0;
						b_z = exp(-aw * config.dt);
						c_z = (aw > 1e-30) ? sz / (kappa_z * kappa_z * config.epsilon0) * (b_z - 1.0) / aw : 0.0;
						PSI_Dzx_val = PSI_Dzx[i][j][k];
						PSI_Dzy_val = PSI_Dzy[i][j][k];
					}
                    
                    // === CPML: D-field update (explicit curl/kappa + psi form) ===
					Dx[i][j][k] += config.dt * (( dHz_dy) / kappa_y + PSI_Dyx_val + (-dHy_dz) / kappa_z + PSI_Dzx_val);
					Dy[i][j][k] += config.dt * ((-dHz_dx) / kappa_x + PSI_Dxy_val + ( dHx_dz) / kappa_z + PSI_Dzy_val);
					Dz[i][j][k] += config.dt * (( dHy_dx) / kappa_x + PSI_Dxz_val + (-dHx_dy) / kappa_y + PSI_Dyz_val);
					if (global_k < config.pml_thickness || global_k >= config.nz - config.pml_thickness) {
						PSI_Dzx[i][j][k] = b_z * PSI_Dzx_val + c_z * (-dHy_dz);
						PSI_Dzy[i][j][k] = b_z * PSI_Dzy_val + c_z * ( dHx_dz);
					}
					// 获取当前网格的材料属性
					int mat_id = medium[i][j][k + halo];
					Material mat = materials[mat_id];
					DielectricTensor eps = mat.eps;
					// 介电函数的虚部相当于有个电导率 sigma = omega * imag(eps)
					if (mat.use_ADE == 0) {
						Dx[i][j][k] -= config.dt * config.unique_source_frequency * 2.0 * 3.14159265 * (eps.xxi * Ex[i][j][k] + eps.xyi * Ey[i][j][k] + eps.xzi * Ez[i][j][k]) * config.epsilon0;
						Dy[i][j][k] -= config.dt * config.unique_source_frequency * 2.0 * 3.14159265 * (eps.yxi * Ex[i][j][k] + eps.yyi * Ey[i][j][k] + eps.yzi * Ez[i][j][k]) * config.epsilon0;
						Dz[i][j][k] -= config.dt * config.unique_source_frequency * 2.0 * 3.14159265 * (eps.zxi * Ex[i][j][k] + eps.zyi * Ey[i][j][k] + eps.zzi * Ez[i][j][k]) * config.epsilon0;
					}
				}
            }
        }

        // 3. 添加激励源
		if (t % config.output_interval == 0 && rank == 0) {
			fprintf(source_file, "%.4e  ",current_time);
		}

        for (int s = 0; s < config.num_sources; s++) {
            Source *src = &config.sources[s];
            double value = calculate_source_value(src, current_time);
			if (t % config.output_interval == 0 && rank ==0) {
				fprintf(source_file, "%.4e  ", value);
			}

            int i = src->position[0];
            int j = src->position[1];
            int global_k = src->position[2];
            
            // 检查位置是否在当前进程
            if (global_k >= local_z_start && global_k < local_z_start + local_nz) {
                int k = global_k - local_z_start;
                
                // 检查位置是否在有效范围内
                if (i < 0 || i >= config.nx || j < 0 || j >= config.ny || k < 0 || k >= local_nz) {
                    continue;
                }
                
                // 根据分量添加到对应场
				if (i >= 0 && i < config.nx && 
					j >= 0 && j < config.ny && 
					k >= 0 && k < local_nz) 
				{
					switch (src->component) {
						case 'x': Dx[i][j][k] += value * config.dt; break;
						case 'y': Dy[i][j][k] += value * config.dt; break;
						case 'z': Dz[i][j][k] += value * config.dt; break;
					}
				}

			}
		}
		if (t % config.output_interval == 0 && rank ==0) {
			fprintf(source_file, "\n");
			fflush(source_file); // 立即刷新缓冲区
		}


		// 4. 从D场计算E场 (考虑介质特性和损耗)
        for (int i = 0; i < config.nx; i++) {
            for (int j = 0; j < config.ny; j++) {
                for (int k = 0; k < local_nz ; k++) {
                    // 获取当前网格的材料属性
                    int mat_id = medium[i][j][k + halo];
                    Material mat = materials[mat_id];
                    DielectricTensor eps_inv = mat.eps_inv;
					int natom = mat.natom;
                    
                    // 考虑电导率损耗: σE项
                    double loss_factor = 1.0 / (1.0 + mat.sigma * config.dt / (2 * config.epsilon0));
                    double loss_term = (mat.sigma * config.dt / (2 * config.epsilon0)) * loss_factor;
                    
					// 离子位移导致的极化强度
					double Pr[3];  // Pr是在晶体坐标系中的
					for (int iE = 0; iE < 3; iE++) {
						Pr[iE] = 0.0;
						for (int iatom = 0; iatom < natom; iatom++) {
							for (int idir = 0; idir < 3; idir++) {
								int ia = iatom * 3 + idir;
								Pr[iE] += mat.QeZV[iatom * 9 + iE * 3 + idir] * r[i][j][k][ia];
							}
						}
					}
					double Ea, Eb, Ec, Da, Db, Dc;
					if (fabs(mat.anglez) > 1e-10) {
						Ea = Ex[i][j][k] * cos(mat.anglez) - Ey[i][j][k] * sin(mat.anglez);
						Eb = Ex[i][j][k] * sin(mat.anglez) + Ey[i][j][k] * cos(mat.anglez);
						Ec = Ez[i][j][k];
						Da = Dx[i][j][k] * cos(mat.anglez) - Dy[i][j][k] * sin(mat.anglez);
						Db = Dx[i][j][k] * sin(mat.anglez) + Dy[i][j][k] * cos(mat.anglez);
						Dc = Dz[i][j][k];
					}
					else if (fabs(mat.anglex) > 1e-10) {
						Ea = Ex[i][j][k];
						Eb = Ey[i][j][k] * cos(-mat.anglex) - Ez[i][j][k] * sin(-mat.anglex);
						Ec = Ey[i][j][k] * sin(-mat.anglex) + Ez[i][j][k] * cos(-mat.anglex);
						Da = Dx[i][j][k];
						Db = Dy[i][j][k] * cos(-mat.anglex) - Dz[i][j][k] * sin(-mat.anglex);
						Dc = Dy[i][j][k] * sin(-mat.anglex) + Dz[i][j][k] * cos(-mat.anglex);
					}
					else {
						Ea = Ex[i][j][k];
						Eb = Ey[i][j][k];
						Ec = Ez[i][j][k];
						Da = Dx[i][j][k];
						Db = Dy[i][j][k];
						Dc = Dz[i][j][k];
					}
					double Da_val = Da - Pr[0];
					double Db_val = Db - Pr[1];
					double Dc_val = Dc - Pr[2];
					double Ea_new, Eb_new, Ec_new;
					if (mat.use_ADE == 0) {
						// E = ε⁻¹ · D (考虑损耗)
						Ea_new = loss_factor * (eps_inv.xx * Da_val + eps_inv.xy * Db_val + eps_inv.xz * Dc_val) / config.epsilon0;
						Eb_new = loss_factor * (eps_inv.yx * Da_val + eps_inv.yy * Db_val + eps_inv.yz * Dc_val) / config.epsilon0;
						Ec_new = loss_factor * (eps_inv.zx * Da_val + eps_inv.zy * Db_val + eps_inv.zz * Dc_val) / config.epsilon0;
					}
					else if (mat.use_ADE == 1) {
						// 使用辅助微分方程方法
						double Ptemp, tempvec[3];
						for (int id = 0; id < 3; id++) {
							Ptemp = P_ADE[i][j][k][id];
							tempvec[id] = 2.0 * P_ADE[i][j][k][id];
							tempvec[id] += -mat.gdt2m1[id][0] * P_ADEo[i][j][k][0] - mat.gdt2m1[id][1] * P_ADEo[i][j][k][1] - mat.gdt2m1[id][2] * P_ADEo[i][j][k][2] + config.epsilon0 * config.dt * config.dt * (mat.wp2[id][0] * Ea + mat.wp2[id][1] * Eb + mat.wp2[id][2] * Ec);
							P_ADE[i][j][k][id] = 0.0;
							for (int kk = 0; kk < 3; kk++) {
								P_ADE[i][j][k][id] += mat.gdt2p1_inv[id][kk] * tempvec[kk];
							}
							P_ADEo[i][j][k][id] = Ptemp;
						}
						Ea_new = (Da_val - P_ADE[i][j][k][0]) / config.epsilon0 / EPS_INF;
						Eb_new = (Db_val - P_ADE[i][j][k][1]) / config.epsilon0 / EPS_INF;
						Ec_new = (Dc_val - P_ADE[i][j][k][2]) / config.epsilon0 / EPS_INF;
					}
					else {
						printf("Error: use_ADE must be 0 or 1\n");
						return 1;
					}

					// 应用损耗项
					Ea = Ea_new - loss_term * Ea_new;
					Eb = Eb_new - loss_term * Eb_new;
					Ec = Ec_new - loss_term * Ec_new;
					if (fabs(mat.anglez) > 1e-10) {
						Ex[i][j][k] = Ea * cos(mat.anglez) + Eb * sin(mat.anglez);
						Ey[i][j][k] = -Ea * sin(mat.anglez) + Eb * cos(mat.anglez);
						Ez[i][j][k] = Ec;
					}
					else if (fabs(mat.anglex) > 1e-10) {
						Ex[i][j][k] = Ea;
						Ey[i][j][k] = Eb * cos(-mat.anglex) + Ec * sin(-mat.anglex);
						Ez[i][j][k] = -Eb * sin(-mat.anglex) + Ec * cos(-mat.anglex);
					}
					else {
						Ex[i][j][k] = Ea;
						Ey[i][j][k] = Eb;
						Ez[i][j][k] = Ec;
					}

					// 加速度、位移、速度是在晶体坐标系的
					// 更新加速度
					double a[natom][3];
					for (int iatom = 0; iatom < natom; iatom++) {
						for (int idir = 0; idir < 3; idir++) {
							a[iatom][idir] = mat.QeZM[iatom * 9 + 0 * 3 + idir] * Ea + mat.QeZM[iatom * 9 + 1 * 3 + idir] * Eb + mat.QeZM[iatom * 9 + 2 * 3 + idir] * Ec;

							for (int jatom = 0; jatom < natom; jatom++) {
								for (int jdir = 0; jdir < 3; jdir++) {
									int ja = jatom * 3 + jdir;
									a[iatom][idir] -= mat.DMM[iatom * 3 * natom * 3 + idir * natom * 3 + jatom * 3 + jdir] * r[i][j][k][ja];
								}
							}
						}
					}

					// 更新位移和速度
					for (int iatom = 0; iatom < mat.natom; iatom++) {
						for (int idir = 0; idir < 3; idir++) {
							int ia = iatom * 3 + idir;
							r[i][j][k][ia] += v[i][j][k][ia] * config.dt + 0.5 * a[iatom][idir] * config.dt * config.dt;
							v[i][j][k][ia] += a[iatom][idir] * config.dt;
						}
					}
                }
            }
        }
		// MPI_Barrier(MPI_COMM_WORLD); // 优化16：移除不必要的barrier
		
		// 输出进度和场值
        if (t % config.output_interval == 0) {            
            // 计算局部电场模最大值 (不包括halo区域)
            double local_max_E = 0.0;
            double local_max_Ex = 0.0;
            for (int i = 0; i < config.nx; i++) {
                for (int j = 0; j < config.ny; j++) {
                    for (int k = 0; k < local_nz; k++) {
                        double E_mag = sqrt(Ex[i][j][k]*Ex[i][j][k] +
                                          Ey[i][j][k]*Ey[i][j][k] +
                                          Ez[i][j][k]*Ez[i][j][k]);
                        double Ex_mag = fabs(Ex[i][j][k]);
                        if (E_mag  >local_max_E) {
                            local_max_E  = E_mag;
                        }
                        if(Ex_mag >local_max_Ex){
                            local_max_Ex = Ex_mag;
                        }
                    }
                }
            }
            
            // 获取全局最大值
            double global_max_E;
            double global_max_Ex;
            MPI_Allreduce(&local_max_E,   &global_max_E,   1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&local_max_Ex, &global_max_Ex, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            
            if (rank == 0) {
                printf("it %d/%d (%.1f%%), t = %.5e s,|E|_max = %.2e V/m,", 
                       t, config.tsteps, 100.0 * t / config.tsteps,  current_time, global_max_E);
            }
			// 检查r场最大值
			double local_max_r = 0.0;
			int max_i = -1, max_j = -1, max_k = -1, max_atom = -1, max_dir = -1;

			for (int i = 0; i < config.nx; i++) {
				for (int j = 0; j < config.ny; j++) {
					for (int k = 0; k < local_nz; k++) {
						int mat_id = medium[i][j][k + halo];
						int natom = materials[mat_id].natom;
						if (natom > 0) {
							for (int a = 0; a < natom; a++) {
								for (int d = 0; d < 3; d++) {
									double r_val = fabs(r[i][j][k][a * 3 + d]);
									if (r_val > local_max_r) {
										local_max_r = r_val;
										max_i = i;
										max_j = j;
										max_k = k;
										max_atom = a;
										max_dir = d;
									}
								}
							}
						}
					}
				}
			}

			// 获取全局r场最大值
			double global_max_r;
			MPI_Allreduce(&local_max_r, &global_max_r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (rank == 0) printf("r_max = %.2e A,", global_max_r/1e-10);
			if (rank == 0) {
				// 计算当前时间
				double current_wall_time = MPI_Wtime();
				double elapsed_time = current_wall_time - start_time;

				// 计算预估剩余时间
				double estimated_remaining_time = 0.0;
				if (t > 0) {
					// 平均每步时间 * 剩余步数
					double avg_time_per_step = elapsed_time / t;
					int remaining_steps = config.tsteps - t;
					estimated_remaining_time = avg_time_per_step * remaining_steps;

					// 格式化时间显示
					int elapsed_hours = (int)(elapsed_time / 3600);
					int elapsed_minutes = (int)((elapsed_time - elapsed_hours * 3600) / 60);
					int elapsed_seconds = (int) (elapsed_time - elapsed_hours * 3600 - elapsed_minutes * 60);

					int remaining_hours = (int)(estimated_remaining_time / 3600);
					int remaining_minutes = (int)((estimated_remaining_time - remaining_hours * 3600) / 60);
					int remaining_seconds = (int) (estimated_remaining_time - remaining_hours * 3600 - remaining_minutes * 60);

					printf(" elapsed: %03d:%02d:%02d, ETA: %03d:%02d:%02d",elapsed_hours, elapsed_minutes, elapsed_seconds,remaining_hours, remaining_minutes, remaining_seconds);

				}
				printf("\n");
				fflush(stdout);
			}
			if (global_max_r > 1e-10) {
				// 找到拥有最大值的进程
				int found = (local_max_r > global_max_r * 0.99) ? rank : -1;
				int owner_rank;
				MPI_Allreduce(&found, &owner_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
				if (rank == owner_rank) {
					printf("ERROR: r field too large at step %d\n", t);
					printf("Max r value: %e\n", global_max_r);
					printf("Location: i=%d, j=%d, k=%d (global k=%d)\n",
						max_i, max_j, max_k, max_k + local_z_start);
					printf("Atom: %d, Direction: %d\n", max_atom, max_dir);

					// 输出v和a值
					int idx = max_atom * 3 + max_dir;
					printf("v value: %e\n", v[max_i][max_j][max_k][idx]);

					// 重新计算a值
					Material mat = materials[medium[max_i][max_j][max_k + halo]];
					double a_val = 0.0;

					// 计算电场对原子的力（通过QeZM）
					for (int iE = 0; iE < 3; iE++) {
						double E_field = (iE == 0) ? Ex[max_i][max_j][max_k] :
							(iE == 1) ? Ey[max_i][max_j][max_k] :
							Ez[max_i][max_j][max_k];
						a_val += mat.QeZM[max_atom * 9 + iE * 3 + max_dir] * E_field;
					}

					// 计算恢复力（通过DMM）
					for (int jatom = 0; jatom < mat.natom; jatom++) {
						for (int jdir = 0; jdir < 3; jdir++) {
							int j_idx = jatom * 3 + jdir;
							a_val -= mat.DMM[max_atom * 3 * mat.natom * 3 + max_dir * mat.natom * 3 + jatom * 3 + jdir] *
								r[max_i][max_j][max_k][j_idx];
						}
					}

					printf("a value: %e\n", a_val);
				}

				// 终止所有进程
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			// 选择要输出的场分量
			double*** output_field = NULL;
			if (strcmp(config.output_component, "Ex") == 0) {
				output_field = Ex;
			}
			else if (strcmp(config.output_component, "Ey") == 0) {
				output_field = Ey;
			}
			else if (strcmp(config.output_component, "Ez") == 0) {
				output_field = Ez;
			}
			else if (strcmp(config.output_component, "E") == 0) {
				// 计算电场模（仅在需要输出时）
				for (int i = 0; i < config.nx; i++) {
					for (int j = 0; j < config.ny; j++) {
						for (int k = 0; k < local_nz; k++) {
							double E_mag = sqrt(Ex[i][j][k] * Ex[i][j][k] +
								Ey[i][j][k] * Ey[i][j][k] +
								Ez[i][j][k] * Ez[i][j][k]);
							E[i][j][k] = E_mag;
						}
					}
				}
				output_field = E; // E是电场模值
			}
			else if (strcmp(config.output_component, "Dx") == 0) {
				output_field = Dx;
			}
			else if (strcmp(config.output_component, "Dy") == 0) {
				output_field = Dy;
			}
			else if (strcmp(config.output_component, "Dz") == 0) {
				output_field = Dz;
			}
			else {
				// 默认输出Ex
				output_field = Ex;
			}
            
            // 保存场切片用于可视化
			char field_file[128];
			if (rank == 0) {
				sprintf(field_file, "field_slice_%c%d_%08d.bin", config.output_slice_axis, config.output_slice_position, t);
			}

			// 广播文件名长度和内容
			int name_len;
			if (rank == 0) {
				name_len = strlen(field_file) + 1; // 包含结束符
			}
			MPI_Bcast(&name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(field_file, name_len, MPI_CHAR, 0, MPI_COMM_WORLD);

			// 所有进程调用保存函数
			save_field_slice_parallel(field_file, output_field, config.nx, config.ny, config.nz,
                             halo, local_z_start, local_nz, config.output_slice_axis, config.output_slice_position,
                             rank, size);

		// Full monitor: output all field components + all atoms displacement/velocity
		if (config.full_monitor_enabled) {
			if (config.monitor_z >= local_z_start && config.monitor_z < local_z_start + local_nz) {
				int mk = config.monitor_z - local_z_start + halo;
				if (config.monitor_x >= 0 && config.monitor_x < config.nx &&
					config.monitor_y >= 0 && config.monitor_y < config.ny &&
					mk >= halo && mk < halo + local_nz) {

					int mat_id = medium[config.monitor_x][config.monitor_y][mk];
					Material mat = materials[mat_id];

					if (t == 0) {
						full_monitor_file = fopen(config.full_monitor_output_file, "w");
						if (full_monitor_file) {
							fprintf(full_monitor_file, "# Time(s)\tEx\tEy\tEz\tDx\tDy\tDz\tHx\tHy\tHz");
							for (int ia = 0; ia < mat.natom; ia++) {
								fprintf(full_monitor_file, "\tr%d_x\tr%d_y\tr%d_z\tv%d_x\tv%d_y\tv%d_z",
									ia, ia, ia, ia, ia, ia);
							}
							fprintf(full_monitor_file, "\n");
						}
					}

					if (full_monitor_file) {
						int field_k = mk;  // E/H/D are allocated with `halo` leading layers, so mk is the field index
						int atom_k = config.monitor_z - local_z_start;  // r/v are allocated WITHOUT halo
						fprintf(full_monitor_file, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e",
							current_time,
							Ex[config.monitor_x][config.monitor_y][field_k],
							Ey[config.monitor_x][config.monitor_y][field_k],
							Ez[config.monitor_x][config.monitor_y][field_k],
							Dx[config.monitor_x][config.monitor_y][field_k],
							Dy[config.monitor_x][config.monitor_y][field_k],
							Dz[config.monitor_x][config.monitor_y][field_k],
							Hx[config.monitor_x][config.monitor_y][field_k],
							Hy[config.monitor_x][config.monitor_y][field_k],
							Hz[config.monitor_x][config.monitor_y][field_k]);
						for (int ia = 0; ia < mat.natom; ia++) {
							for (int id = 0; id < 3; id++) {
								int idx = ia * 3 + id;
								fprintf(full_monitor_file, "\t%e", r[config.monitor_x][config.monitor_y][atom_k][idx]);
							}
							for (int id = 0; id < 3; id++) {
								int idx = ia * 3 + id;
								fprintf(full_monitor_file, "\t%e", v[config.monitor_x][config.monitor_y][atom_k][idx]);
							}
						}
						fprintf(full_monitor_file, "\n");
					}
				}
			}
		}
	}
		if (0) {
			if (t == 100) {
				char test_filename[256];
				sprintf(test_filename, "test_slice_t%d_rank%d.txt", t, rank);
				FILE* test = fopen(test_filename, "w");

				if (test == NULL) {
					fprintf(stderr, "Rank %d: Failed to open file %s\n", rank, test_filename);
				}
				else {
					//fprintf(test, "# Time step %d, Rank %d: local_z_start=%d, local_nz=%d, halo=%d\n",
					//	t, rank, local_z_start, local_nz, halo);
					//fprintf(test, "# j  global_k  Ex_value\n");

					for (int j = 0; j < config.ny; j++) {
						for (int k = 0; k < local_nz; k++) {
							int global_k = k + local_z_start;
							fprintf(test, "%d  %d  %e\n", j, global_k, Ex[251][j][k]);
						}
					}
					fclose(test);
				}
			}
		}

		if (config.monitor_enabled) {
			// 检查监测点是否在当前进程
			if (config.monitor_z >= local_z_start && config.monitor_z < local_z_start + local_nz) {
				int k = config.monitor_z - local_z_start;
				if (config.monitor_x >= 0 && config.monitor_x < config.nx &&
					config.monitor_y >= 0 && config.monitor_y < config.ny &&
					k >= 0 && k < local_nz) {

					int field_k = k + halo;  // E/H/D are allocated with `halo` leading layers
					int atom_k = k;          // r/v are allocated WITHOUT halo (local_nz entries)

					// 获取该点的材料信息
					int mat_id = medium[config.monitor_x][config.monitor_y][k + halo];
					Material mat = materials[mat_id];

					// 检查原子是否存在
					if (config.monitor_atom < mat.natom) {

						// 计算原子索引
						int atom_idx = config.monitor_atom * 3 + config.monitor_dir;

						// 打开文件（第一次）
						if (t == 0) {
							monitor_file = fopen(config.monitor_output_file, "w");
							if (monitor_file) {
								fprintf(monitor_file, "# Time(s)\tEx\tEy\tEz\tDx\tDy\tDz\tHx\tHy\tHz\tR\tV\n");
							}
						}

						// 写入数据
						if (monitor_file) {
							double r_val = r[config.monitor_x][config.monitor_y][atom_k][atom_idx];
							double v_val = v[config.monitor_x][config.monitor_y][atom_k][atom_idx];
							fprintf(monitor_file, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
								current_time,
								Ex[config.monitor_x][config.monitor_y][field_k],
								Ey[config.monitor_x][config.monitor_y][field_k],
								Ez[config.monitor_x][config.monitor_y][field_k],
								Dx[config.monitor_x][config.monitor_y][field_k],
								Dy[config.monitor_x][config.monitor_y][field_k],
								Dz[config.monitor_x][config.monitor_y][field_k],
								Hx[config.monitor_x][config.monitor_y][field_k],
								Hy[config.monitor_x][config.monitor_y][field_k],
								Hz[config.monitor_x][config.monitor_y][field_k],
								r_val,
								v_val);
						}
					}
					else {
						// 打开文件（第一次）
						if (t == 0) {
							monitor_file = fopen(config.monitor_output_file, "w");
							if (monitor_file) {
								fprintf(monitor_file, "# Time(s)\tEx\tEy\tEz\tDx\tDy\tDz\tHx\tHy\tHz\n");
							}
						}

						// 写入数据
						if (monitor_file) {
							fprintf(monitor_file, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
								current_time,
								Ex[config.monitor_x][config.monitor_y][field_k],
								Ey[config.monitor_x][config.monitor_y][field_k],
								Ez[config.monitor_x][config.monitor_y][field_k],
								Dx[config.monitor_x][config.monitor_y][field_k],
								Dy[config.monitor_x][config.monitor_y][field_k],
								Dz[config.monitor_x][config.monitor_y][field_k],
								Hx[config.monitor_x][config.monitor_y][field_k],
								Hy[config.monitor_x][config.monitor_y][field_k],
								Hz[config.monitor_x][config.monitor_y][field_k]
								);
						}

					}

				}
			}
		}
		}
	/*******************主循环结束***************************/

	if (rank == 0 && source_file) {
		fclose(source_file);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (config.monitor_enabled) {
		int has_file = (monitor_file != NULL) ? 1 : 0;
		int any_has = 0;
		MPI_Reduce(&has_file, &any_has, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (monitor_file) {
			fclose(monitor_file);
		}
		if (rank == 0 && any_has) {
			printf("Monitor data saved to %s\n", config.monitor_output_file);
		}
	}
	{
		int has_file = (full_monitor_file != NULL) ? 1 : 0;
		int any_has = 0;
		MPI_Reduce(&has_file, &any_has, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (full_monitor_file) {
			fclose(full_monitor_file);
		}
		if (rank == 0 && any_has) {
			printf("Full monitor data saved to %s\n", config.full_monitor_output_file);
		}
	}
    // 结束计时
	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("\nSimulation completed in %.2f seconds\n", end_time - start_time);
    }
	MPI_Barrier(MPI_COMM_WORLD);
	
    if (config.sources) {
        free(config.sources);
    }
	
	if (rank == 0) {
		printf("free E, D, H "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
    free_3d(arr_Ex, config.nx, config.ny);
    free_3d(arr_Ey, config.nx, config.ny);
    free_3d(arr_Ez, config.nx, config.ny);
    free_3d(arr_Dx, config.nx, config.ny);
    free_3d(arr_Dy, config.nx, config.ny);
    free_3d(arr_Dz, config.nx, config.ny);
    free_3d(arr_Hx, config.nx, config.ny);
    free_3d(arr_Hy, config.nx, config.ny);
    free_3d(arr_Hz, config.nx, config.ny);
	if (rank == 0) {
		printf("E, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (E != NULL) {
		free_3d(arr_E, config.nx, config.ny);
	}

	if (rank == 0) {
		printf("PSI, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free_3d(arr_PSI_Dxy, 2 * config.pml_thickness, config.ny);
	free_3d(arr_PSI_Dxz, 2 * config.pml_thickness, config.ny);
	free_3d(arr_PSI_Dyx, config.nx, 2 * config.pml_thickness);
	free_3d(arr_PSI_Dyz, config.nx, 2 * config.pml_thickness);
	free_3d(arr_PSI_Hxy, 2 * config.pml_thickness, config.ny);
	free_3d(arr_PSI_Hxz, 2 * config.pml_thickness, config.ny);
	free_3d(arr_PSI_Hyx, config.nx, 2 * config.pml_thickness);
	free_3d(arr_PSI_Hyz, config.nx, 2 * config.pml_thickness);
	if (rank == 0) {
		printf("PSI_Dz, PSI_Hz, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (PSI_Dzx) {
		free_3d(arr_PSI_Dzx, config.nx, config.ny);
		free_3d(arr_PSI_Dzy, config.nx, config.ny);
		free_3d(arr_PSI_Hzx, config.nx, config.ny);
		free_3d(arr_PSI_Hzy, config.nx, config.ny);
	}

	if (rank == 0) {
		printf("medium, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
    free_3d_int(medium, config.nx, config.ny);
    
	if (rank == 0) {
		printf("r, v, ");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < config.nx; i++) {
		for (int j = 0; j < config.ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				free(r[i][j][k]);
				free(v[i][j][k]);
			}
			free(r[i][j]);
			free(v[i][j]);
		}
		free(r[i]);
		free(v[i]);
	}
	free(r);
	free(v);

	// 释放 P_ADE 和 P_ADEo (4维数组, 内层仅 use_ADE==1 的格点有分配)
	if (rank == 0) {
		printf("ADE, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	for (int i = 0; i < config.nx; i++) {
		for (int j = 0; j < config.ny; j++) {
			for (int k = 0; k < local_nz; k++) {
				free(P_ADE[i][j][k]);
				free(P_ADEo[i][j][k]);
			}
			free(P_ADE[i][j]);
			free(P_ADEo[i][j]);
		}
		free(P_ADE[i]);
		free(P_ADEo[i]);
	}
	free(P_ADE);
	free(P_ADEo);

	if (rank == 0) {
		printf("materials, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	if (config.materials) {
		for (int i = 0; i < config.num_materials; i++) {
			if (rank == 0) {
				printf("%d, born, ",i); fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (config.materials[i].born) {
				free(config.materials[i].born);
				config.materials[i].born = NULL;
			}
			if (rank == 0) {
				printf("QeZV, "); fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (config.materials[i].QeZV_filename_set && config.materials[i].QeZV) {
				free(config.materials[i].QeZV);
				config.materials[i].QeZV = NULL;
			}
			if (rank == 0) {
				printf("QeZM, "); fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (config.materials[i].QeZM_filename_set && config.materials[i].QeZM) {
				free(config.materials[i].QeZM);
				config.materials[i].QeZM = NULL;
			}
			if (rank == 0) {
				printf("DMM, "); fflush(stdout);
			}
			MPI_Barrier(MPI_COMM_WORLD);
			if (config.materials[i].DMM_filename_set && config.materials[i].DMM) {
				free(config.materials[i].DMM);
				config.materials[i].DMM = NULL;
			}
		}
		free(config.materials);
		config.materials = NULL;
	}

	if (rank == 0) {
		printf("pml, "); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
    free_pml(pml_x);
    free_pml(pml_y);
    free_pml(pml_z);
	if (rank == 0) {
		printf("halo\n"); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free(halo_buf_E_send_lower);
	free(halo_buf_E_recv_lower);
	free(halo_buf_E_send_upper);
	free(halo_buf_E_recv_upper);
	free(halo_buf_H_send_lower);
	free(halo_buf_H_recv_lower);
	free(halo_buf_H_send_upper);
	free(halo_buf_H_recv_upper);
	if (rank == 0) {
		printf("Ready to exit\n"); fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
    return 0;
}
