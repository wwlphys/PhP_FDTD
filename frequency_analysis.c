#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fftw3.h>

// 配置文件结构体
typedef struct {
	double source_frequency;  // 源频率
	double dt;                // 时间步长
	int nx, ny, nz;           // 网格尺寸
	int output_interval;      // 输出间隔
	int tsteps;               // 总时间步数
} ConfigInfo;

// 切片文件信息
typedef struct {
	char filename[256];
	int time_step;
	double time;
} SliceFile;

// 频率结果结构体
typedef struct {
	double real;
	double imag;
	double magnitude;
	double phase;
} FrequencyResult;

// 从配置文件读取参数
ConfigInfo read_config(const char* config_file) {
	ConfigInfo config = { 0 };
	FILE* file = fopen(config_file, "r");
	if (!file) {
		fprintf(stderr, "Error opening config file: %s\n", config_file);
		exit(1);
	}

	char line[256];
	int source_count = 0;
	int has_dt = 0;
	double source_freq = 0.0;

	while (fgets(line, sizeof(line), file)) {
		// 去除换行符
		line[strcspn(line, "\n")] = 0;

		// 跳过注释和空行
		if (line[0] == '#' || line[0] == '\n') continue;

		if (strstr(line, "source") == line) {
			source_count++;

			// 初始化标志
			int has_frequency = 0;
			int has_wavenumber = 0;
			double frequency = 0.0;
			double wavenumber = 0.0;

			// 解析源参数
			char* token = strtok(line, ",");
			while (token != NULL) {
				// 查找频率参数
				if (strstr(token, "frequency") != NULL) {
					char* eq = strchr(token, '=');
					if (eq) {
						frequency = atof(eq + 1);
						has_frequency = 1;
					}
				}
				// 查找波数参数
				else if (strstr(token, "wavenumber") != NULL) {
					char* eq = strchr(token, '=');
					if (eq) {
						wavenumber = atof(eq + 1);
						has_wavenumber = 1;
					}
				}
				token = strtok(NULL, ",");
			}

			// 处理频率/波数逻辑
			if (has_frequency && has_wavenumber) {
				fprintf(stderr, "Error: Cannot specify both frequency and wavenumber for source\n");
				fclose(file);
				exit(1);
			}

			if (!has_frequency && !has_wavenumber) {
				fprintf(stderr, "Warning: No frequency or wavenumber specified for source\n");
			}
			else if (has_wavenumber) {
				// 转换波数为频率: frequency = wavenumber * 0.03e12
				frequency = wavenumber * 0.03e12;
				has_frequency = 1;
			}

			if (has_frequency) {
				source_freq = frequency;
			}
		}
		else if (strstr(line, "dt") == line) {
			char* eq = strchr(line, '=');
			if (eq) {
				config.dt = atof(eq + 1);
				has_dt = 1;
			}
		}
		else if (strstr(line, "tsteps") == line) {
			char* eq = strchr(line, '=');
			if (eq) config.tsteps = atoi(eq + 1);
		}
		else if (strstr(line, "nx") == line) {
			char* eq = strchr(line, '=');
			if (eq) config.nx = atoi(eq + 1);
		}
		else if (strstr(line, "ny") == line) {
			char* eq = strchr(line, '=');
			if (eq) config.ny = atoi(eq + 1);
		}
		else if (strstr(line, "nz") == line) {
			char* eq = strchr(line, '=');
			if (eq) config.nz = atoi(eq + 1);
		}
		else if (strstr(line, "output_interval") == line) {
			char* eq = strchr(line, '=');
			if (eq) config.output_interval = atoi(eq + 1);
		}
	}

	fclose(file);

	// dt 必须显式给出：切片按 output_interval 步输出，采样间隔 = output_interval * dt，
	// 本工具无法像 fdtd.x 那样用 CFL 条件反推（fdtd.x 自动取值时不会把结果写回配置）。
	if (!has_dt) {
		fprintf(stderr, "Error: no 'dt' entry found in config file: %s\n", config_file);
		fprintf(stderr, "  frequency_analysis needs an explicit dt, because the sampling interval of the\n");
		fprintf(stderr, "  field slices is output_interval * dt. Add a line to the [grid] section, e.g.\n");
		fprintf(stderr, "      dt = 3.8105e-17\n");
		fprintf(stderr, "  If the config omits dt, fdtd.x derives it from the CFL condition and prints it\n");
		fprintf(stderr, "  at start-up as \"Time steps: <n>, dt=<value> s\"; copy that value here.\n");
		exit(1);
	}
	if (!(config.dt > 0.0)) {
		fprintf(stderr, "Error: invalid dt = %g in config file: %s (dt must be a positive number)\n",
			config.dt, config_file);
		exit(1);
	}

	if (source_count > 0 && source_freq > 0.0) {
		config.source_frequency = source_freq;
	}
	else {
		fprintf(stderr, "Warning: No valid source frequency found in config file\n");
	}

	return config;
}

// 比较函数用于排序
int compare_files(const void* a, const void* b) {
	SliceFile* file_a = (SliceFile*)a;
	SliceFile* file_b = (SliceFile*)b;
	return file_a->time_step - file_b->time_step;
}

// 查找所有切片文件
SliceFile* find_slice_files(const char* pattern, int* file_count) {
	DIR* dir;
	struct dirent* entry;
	SliceFile* files = NULL;
	int count = 0;
	int capacity = 100;

	files = malloc(capacity * sizeof(SliceFile));
	if (!files) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(1);
	}

	dir = opendir(".");
	if (!dir) {
		fprintf(stderr, "Cannot open current directory\n");
		exit(1);
	}

	while ((entry = readdir(dir)) != NULL) {
		if (strstr(entry->d_name, pattern) && strstr(entry->d_name, ".bin")) {
			// 创建副本用于处理
			char filename[256];
			strcpy(filename, entry->d_name);

			// 提取时间步
			char* underscore = strrchr(filename, '_');
			if (underscore) {
				char* dot = strchr(underscore + 1, '.');
				if (dot) {
					*dot = '\0';
					int time_step = atoi(underscore + 1);

					if (count >= capacity) {
						capacity *= 2;
						files = realloc(files, capacity * sizeof(SliceFile));
						if (!files) {
							fprintf(stderr, "Memory reallocation failed\n");
							exit(1);
						}
					}

					strcpy(files[count].filename, entry->d_name);
					files[count].time_step = time_step;
					count++;
				}
			}
		}
	}

	closedir(dir);

	// 按时间步排序
	if (count > 0) {
		qsort(files, count, sizeof(SliceFile), compare_files);
	}

	*file_count = count;
	return files;
}

// 读取单个切片文件
float* read_slice_file(const char* filename, int* nx, int* ny) {
	FILE* file = fopen(filename, "rb");
	if (!file) {
		fprintf(stderr, "Error opening file: %s\n", filename);
		return NULL;
	}

	// 读取尺寸
	size_t read_nx = fread(nx, sizeof(int), 1, file);
	size_t read_ny = fread(ny, sizeof(int), 1, file);

	if (read_nx != 1 || read_ny != 1) {
		fprintf(stderr, "Error reading dimensions from %s\n", filename);
		fclose(file);
		return NULL;
	}

	// 分配内存
	float* data = malloc(*nx * *ny * sizeof(float));
	if (!data) {
		fprintf(stderr, "Memory allocation failed for %s\n", filename);
		fclose(file);
		return NULL;
	}

	// 读取数据
	size_t read_count = fread(data, sizeof(float), *nx * *ny, file);
	if (read_count != *nx * *ny) {
		fprintf(stderr, "Error reading data from %s: read %zu of %d\n",
			filename, read_count, *nx * *ny);
		free(data);
		fclose(file);
		return NULL;
	}

	fclose(file);
	return data;
}

// 保存频率结果到文件
void save_frequency_result(const char* filename, double freq_hz, int freq_index,
	double* real_results, double* imag_results,
	double* magnitude_results, double* phase_results,
	int nx, int ny, ConfigInfo config, double fs, int N) {
	FILE* output = fopen(filename, "w");
	if (!output) {
		fprintf(stderr, "Error creating output file: %s\n", filename);
		return;
	}

	// 写入文件头
	fprintf(output, "# Frequency slice at %.3e Hz (index: %d)\n", freq_hz, freq_index);
	fprintf(output, "# Source frequency: %.3e Hz\n", config.source_frequency);
	fprintf(output, "# Difference: %.3e Hz\n", fabs(freq_hz - config.source_frequency));
	fprintf(output, "# FFT parameters:\n");
	fprintf(output, "#   Time points: %d\n", N);
	fprintf(output, "#   Sampling frequency: %.3e Hz\n", fs);
	fprintf(output, "#   Frequency resolution: %.3e Hz\n", fs / N);
	fprintf(output, "# Slice dimensions: %d x %d\n", nx, ny);
	fprintf(output, "# Format: x_index y_index real_part imag_part magnitude phase(rad) phase(deg)\n");

	// 写入数据（只写入中心区域：25%-75%）
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			int idx = i * ny + j;
			if (i > nx * 0.25 && i < nx * 0.75 && j > ny * 0.25 && j < ny * 0.75) {
				fprintf(output, "%d %d %.6e %.6e %.6e %.6e %.6e\n",
					i, j,
					real_results[idx],
					imag_results[idx],
					magnitude_results[idx],
					phase_results[idx],
					phase_results[idx] * 180.0 / M_PI);
			}
		}
	}

	fclose(output);
	printf("  Results saved to %s\n", filename);
}

// 主函数
int main(int argc, char* argv[]) {
	if (argc < 2) {
		printf("Usage: %s <config_file.cfg>\n", argv[0]);
		printf("Example: %s simulation.cfg\n", argv[0]);
		return 1;
	}

	// 1. 读取配置文件
	ConfigInfo config = read_config(argv[1]);
	printf("Source frequency: %.3e Hz\n", config.source_frequency);
	printf("Time step (dt): %.3e s\n", config.dt);
	printf("Total time steps: %d\n", config.tsteps);
	printf("Output interval: %d\n", config.output_interval);

	// 2. 查找所有切片文件
	int file_count;
	SliceFile* files = find_slice_files("field_slice_", &file_count);

	if (file_count == 0) {
		fprintf(stderr, "No slice files found!\n");
		free(files);
		return 1;
	}

	printf("Found %d slice files\n", file_count);

	// 3. 读取第一个文件获取网格尺寸
	int nx, ny;
	float* first_slice = read_slice_file(files[0].filename, &nx, &ny);
	if (!first_slice) {
		fprintf(stderr, "Failed to read first slice file\n");
		free(files);
		return 1;
	}

	printf("Slice dimensions: %d x %d\n", nx, ny);
	free(first_slice);

	// 4. 计算时间数组
	int N = file_count;  // 时间点数

	// 检查是否有足够的数据点
	if (N < 2) {
		fprintf(stderr, "Not enough time points for FFT (need at least 2, got %d)\n", N);
		free(files);
		return 1;
	}

	double* time_array = malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) {
		time_array[i] = files[i].time_step * config.dt;
	}

	// 5. 计算采样频率和频率分辨率
	double T = time_array[N - 1] - time_array[0];  // 总时间
	double dt_avg = T / (N - 1);  // 平均时间间隔
	double fs = 1.0 / dt_avg;     // 采样频率

	printf("Total time: %.3e s\n", T);
	printf("Average dt: %.3e s\n", dt_avg);
	printf("Sampling frequency: %.3e Hz\n", fs);
	printf("Frequency resolution: %.3e Hz\n", fs / N);

	// 6. 为目标频率找到最接近的索引
	int target_index = -1;
	double min_diff = 1e100;
	for (int i = 0; i < N / 2; i++) {
		double freq = i * fs / N;
		double diff = fabs(freq - config.source_frequency);
		if (diff < min_diff) {
			min_diff = diff;
			target_index = i;
		}
	}

	if (target_index < 0 || target_index >= N) {
		fprintf(stderr, "Error finding target frequency index\n");
		free(time_array);
		free(files);
		return 1;
	}

	// 确定要输出的9个频率索引（包括目标频率及其前后各4个）
	int freq_indices[9];
	int freq_count = 0;

	// 确保索引在有效范围内
	for (int offset = -4; offset <= 4; offset++) {
		int idx = target_index + offset;
		if (idx >= 0 && idx < N / 2) {  // 只考虑正频率
			freq_indices[freq_count++] = idx;
		}
	}

	// 如果没找到9个频率，调整范围
	if (freq_count < 9) {
		freq_count = 0;
		// 尝试以目标频率为中心取9个，如果不行则尽可能多地取
		for (int offset = -4; offset <= 4 && freq_count < 9; offset++) {
			int idx = target_index + offset;
			if (idx >= 0 && idx < N / 2) {
				freq_indices[freq_count++] = idx;
			}
		}
	}

	printf("Will output %d frequencies around source frequency:\n", freq_count);
	for (int i = 0; i < freq_count; i++) {
		double freq = freq_indices[i] * fs / N;
		printf("  Index %d: %.3e Hz (diff: %.3e Hz)\n",
			freq_indices[i], freq, fabs(freq - config.source_frequency));
	}

	// 7. 分配内存并读取所有时间步数据
	printf("Allocating memory for time series data...\n");
	float** time_series_data = malloc(N * sizeof(float*));
	for (int t = 0; t < N; t++) {
		time_series_data[t] = malloc(nx * ny * sizeof(float));
		if (!time_series_data[t]) {
			fprintf(stderr, "Memory allocation failed for time step %d\n", t);
			// 清理已分配的内存
			for (int i = 0; i < t; i++) {
				free(time_series_data[i]);
			}
			free(time_series_data);
			free(time_array);
			free(files);
			return 1;
		}
	}

	// 读取所有文件
	printf("Reading slice files...\n");
	for (int t = 0; t < N; t++) {
		int file_nx, file_ny;
		float* slice_data = read_slice_file(files[t].filename, &file_nx, &file_ny);
		if (!slice_data) {
			fprintf(stderr, "Error reading file: %s\n", files[t].filename);
			// 清理内存
			for (int i = 0; i <= t; i++) {
				free(time_series_data[i]);
			}
			free(time_series_data);
			free(time_array);
			free(files);
			return 1;
		}

		// 检查尺寸是否一致
		if (file_nx != nx || file_ny != ny) {
			fprintf(stderr, "Dimension mismatch in file %s: expected %dx%d, got %dx%d\n",
				files[t].filename, nx, ny, file_nx, file_ny);
			free(slice_data);
			// 清理内存
			for (int i = 0; i <= t; i++) {
				free(time_series_data[i]);
			}
			free(time_series_data);
			free(time_array);
			free(files);
			return 1;
		}

		// 复制数据
		memcpy(time_series_data[t], slice_data, nx * ny * sizeof(float));
		free(slice_data);

		if ((t + 1) % 100 == 0 || t == N - 1) {
			printf("  Read %d/%d files\n", t + 1, N);
		}
	}

	// 8. 为每个空间点进行FFT
	printf("Performing FFT for each spatial point...\n");

	// 使用FFTW
	fftw_complex* in, * out;
	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
	fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	// 为每个频率分配存储结果的内存
	double** real_results = malloc(freq_count * sizeof(double*));
	double** imag_results = malloc(freq_count * sizeof(double*));
	double** magnitude_results = malloc(freq_count * sizeof(double*));
	double** phase_results = malloc(freq_count * sizeof(double*));

	if (!real_results || !imag_results || !magnitude_results || !phase_results) {
		fprintf(stderr, "Memory allocation failed for results\n");
		// 清理内存
		fftw_destroy_plan(plan);
		fftw_free(in);
		fftw_free(out);
		for (int t = 0; t < N; t++) {
			free(time_series_data[t]);
		}
		free(time_series_data);
		free(time_array);
		free(files);
		return 1;
	}

	for (int f = 0; f < freq_count; f++) {
		real_results[f] = malloc(nx * ny * sizeof(double));
		imag_results[f] = malloc(nx * ny * sizeof(double));
		magnitude_results[f] = malloc(nx * ny * sizeof(double));
		phase_results[f] = malloc(nx * ny * sizeof(double));

		if (!real_results[f] || !imag_results[f] || !magnitude_results[f] || !phase_results[f]) {
			fprintf(stderr, "Memory allocation failed for frequency %d\n", f);
			// 清理内存
			for (int i = 0; i < f; i++) {
				free(real_results[i]);
				free(imag_results[i]);
				free(magnitude_results[i]);
				free(phase_results[i]);
			}
			fftw_destroy_plan(plan);
			fftw_free(in);
			fftw_free(out);
			for (int t = 0; t < N; t++) {
				free(time_series_data[t]);
			}
			free(time_series_data);
			free(time_array);
			free(files);
			return 1;
		}
	}

	int total_points = nx * ny;
	for (int idx = 0; idx < total_points; idx++) {
		// 准备输入数据
		for (int t = 0; t < N; t++) {
			double* in_ptr = (double*)in;
			in_ptr[2 * t] = time_series_data[t][idx];  // 实部
			in_ptr[2 * t + 1] = 0.0;                   // 虚部
		}

		// 执行FFT
		fftw_execute(plan);

		// 获取每个目标频率的复数结果
		for (int f = 0; f < freq_count; f++) {
			int freq_idx = freq_indices[f];
			double* out_ptr = (double*)out;
			double real_part = out_ptr[2 * freq_idx] / N;
			double imag_part = out_ptr[2 * freq_idx + 1] / N;

			real_results[f][idx] = real_part;
			imag_results[f][idx] = imag_part;
			magnitude_results[f][idx] = sqrt(real_part * real_part + imag_part * imag_part);
			phase_results[f][idx] = atan2(imag_part, real_part);
		}

		if ((idx + 1) % 10000 == 0 || idx == total_points - 1) {
			//printf("  Processed %d/%d points\n", idx + 1, total_points);
		}
	}

	// 9. 保存结果到文件
	printf("Saving results to %d frequency files...\n", freq_count);
	for (int f = 0; f < freq_count; f++) {
		double freq_hz = freq_indices[f] * fs / N;
		char filename[256];
		snprintf(filename, sizeof(filename), "frequency_slice_%d_%.3eHz_%.3ecm-1.txt",
			freq_indices[f], freq_hz, freq_hz/0.03e12);

		save_frequency_result(filename, freq_hz, freq_indices[f],
			real_results[f], imag_results[f],
			magnitude_results[f], phase_results[f],
			nx, ny, config, fs, N);
	}

	// 10. 清理内存
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);

	for (int f = 0; f < freq_count; f++) {
		free(real_results[f]);
		free(imag_results[f]);
		free(magnitude_results[f]);
		free(phase_results[f]);
	}
	free(real_results);
	free(imag_results);
	free(magnitude_results);
	free(phase_results);

	for (int t = 0; t < N; t++) {
		free(time_series_data[t]);
	}
	free(time_series_data);

	free(time_array);
	free(files);

	fftw_cleanup();

	printf("Done!\n");
	return 0;
}