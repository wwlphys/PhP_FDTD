# fdtd — 3D MPI-parallel electromagnetic / phonon-polariton FDTD solver

[中文](README_Chinese.md) | **English**

A three-dimensional finite-difference time-domain (FDTD) full-wave electromagnetic solver written in C and parallelized with MPI, aimed at time-domain simulation of **anisotropic media**, **negative-permittivity media**, and **strong coupling between optical fields and optical phonons in polar crystals** (phonon polaritons, PhP); it can be used to model near-field problems such as scattering-type scanning near-field optical microscopy (s-SNOM).

The solver uses a Yee-grid leapfrog scheme, together with a CPML absorbing boundary, tensor-form auxiliary differential equation (ADE) constitutive relations, and ionic equations of motion driven by first-principles data (Born effective charges + dynamical matrix), realizing two-way coupling between Maxwell's equations and lattice dynamics.

```
   ┌───────────────────┐   E    ┌─────────────────────┐   Z*·E   ┌────────────────┐
   │ Maxwell equations │ ─────► │ Ion motion equation │ ───────► │ Lattice r, v   │
   │ ∂D/∂t = ∇×H       │        │ a = Z*·E/m − DMM·r  │          └───────┬────────┘
   │ ∂H/∂t = −∇×E      │ ◄───── │                     │                  │
   └───────────────────┘ P_ion  └─────────────────────┘ ◄────────────────┘
                        = Σ Z*·r
```

---

## Contents

- [1. Key Features](#1-key-features)
- [2. Directory Structure](#2-directory-structure)
- [3. Dependencies and Compilation](#3-dependencies-and-compilation)
- [4. Quick Start](#4-quick-start)
- [5. Configuration File Reference](#5-configuration-file-reference)
- [6. Units and Physical Conventions](#6-units-and-physical-conventions)
- [7. Input File Formats](#7-input-file-formats)
- [8. Output File Formats](#8-output-file-formats)
- [9. Numerical Method](#9-numerical-method)
- [10. Post-Processing Workflow](#10-post-processing-workflow)
- [11. Parallelization and Performance](#11-parallelization-and-performance)
- [12. Known Limitations and Caveats](#12-known-limitations-and-caveats)
- [13. Troubleshooting](#13-troubleshooting)
- [14. References and Acknowledgments](#14-references-and-acknowledgments)
- [15. License](#15-license)

---

## 1. Key Features

| Category | Description |
| --- | --- |
| Solver | Three-dimensional Yee-grid explicit leapfrog FDTD on a uniform grid; a different spatial step may be used in each direction (`delta_x/y/z`) |
| Absorbing boundary | **CPML** (convolutional perfectly matched layer, Roden & Gedney 2000), with polynomial profiles of the three parameters `σ`, `κ`, `α`; the target reflectivity is set through `pml_reflection` |
| Constitutive relation | Complete 3×3 **anisotropic complex permittivity tensor** (real + imaginary part, 12 independent components in total); the imaginary part of the dielectric function is treated as conductive loss (`σ` and `σ_m` may also be given explicitly) |
| Negative permittivity | **ADE (auxiliary differential equation) method**, tensor-form Drude / Drude-Lorentz dispersion models, for media whose dielectric function has a negative real part |
| Lattice coupling | Driven by first-principles data: Born effective charge tensors (`QeZV` / `QeZM`) + dynamical matrix (`DMM`); the ionic equations of motion are solved at every grid point and coupled two-way with Maxwell's equations, so that **PhP / optical phonons** can be simulated |
| Coordinate transformation | The material coordinate system can be rotated about the `z` axis (`anglez`) or the `x` axis (`anglex`), or the XZ axes can be swapped directly (`switchxz`), which makes it easy to place an arbitrary crystal orientation |
| Excitation source | Gaussian pulse, sine wave, Ricker wavelet, dipole; the frequency can be given in Hz (`frequency`) or as a wavenumber in cm⁻¹ (`wavenumber`) |
| Parallelization | MPI one-dimensional domain decomposition (along the `z` axis) + halo exchange; boundary data are packed and then communicated in batches |
| Output | Field slices (binary), single-point all-component time-domain monitoring, full monitoring (fields + displacements/velocities of all atoms) |
| Engineering | Memory usage statistics (`[MEM]` log lines), per-step progress and estimated time remaining (ETA) |

The accompanying Python / MATLAB / C post-processing tools can perform: medium geometry generation, medium distribution and geometry checking, field distribution snapshots and animations, extraction of near-field frequency slices by FFT of time-domain signals, and generation of Drude-Lorentz dielectric function tables.

---

## 2. Directory Structure

```
.
├── README.md                   # English documentation (default on GitHub)
├── README_Chinese.md           # Chinese documentation
├── LICENSE                     # MIT license
├── .gitignore                  # excludes the 文档/ directory and run artifacts
├── fdtd.c                      # main solver (C + MPI, single file, ~4000 lines)
├── frequency_analysis.c        # near-field frequency slice extraction: pointwise FFT of the field slices (requires FFTW3)
├── generate_medium5.py         # generates medium_map.bin from the [medium] section of the config file
├── visualize_results2.py       # field slice binary → quick-look figures + max|field| listing
├── visualize_results2_switchxy.py  # same, but swaps the x/y axes when plotting
├── visualize_input3.m          # (MATLAB) checks the geometry and source positions of a config file
├── visualize_medium.m          # (MATLAB) checks the medium distribution in medium_map.bin
├── v_source.m                  # (MATLAB) quick preview of the excitation source time-domain waveform
├── Material_data/              # material data (hBN, MoO3)
│   ├── hBN/                    # OUTCAR.txt, POSCAR, QeZV_clean.txt,
│   │                           # QeZM_clean.txt, DMM_clean.txt, README.txt
│   └── MoO3/                   # same as above (note: the MoO3 file is named QeZm_clean.txt, lowercase m)
├── Lorentz_dielectric/         # Drude-Lorentz dielectric function tables + the MATLAB scripts that generate them
└── examples/                   # example cases (see 4.1 / 4.2)
    ├── example.cfg             #   minimal example: 80×80×48, MoO3 block, runs in a few minutes
    ├── Lorentz_zdipole_650cm/  #   research-grade case: Lorentz dispersion model MoO3, z dipole @650 cm⁻¹
    │   ├── simulation.cfg
    │   └── dielectric_MoO3_Lorentz.txt
    └── MoO3_zdipole_800cm/     #   research-grade case: phonon-coupled MoO3 (Born + dynamical matrix) @800 cm⁻¹
        ├── simulation.cfg
        ├── OUTCAR.txt
        ├── QeZV_clean.txt
        ├── QeZM_clean.txt
        └── DMM_clean.txt
```

> Note: this repository does not contain any `.docx` theoretical derivation documents; the corresponding directory is excluded by `.gitignore`.

---

## 3. Dependencies and Compilation

### 3.1 Dependencies

| Component | Purpose | Notes |
| --- | --- | --- |
| C compiler | compiles `fdtd.c`, `frequency_analysis.c` | GCC / Clang / ICC all work |
| MPI | required by `fdtd.c` | OpenMPI, MPICH, Intel MPI all work (the code only uses the standard MPI-2 API) |
| math / rt | link the math library | `-lm -lrt` |
| FFTW3 | needed **only** by `frequency_analysis.c` | `libfftw3-dev` |
| Python 3 + NumPy | `generate_medium5.py` | only `numpy` and the standard library are used |
| Python 3 + Matplotlib | `visualize_results2*.py` | used for plotting only |
| MATLAB | `*.m` scripts | Octave may work, but compatibility has not been tested |

One-line installation on Debian / Ubuntu:

```bash
sudo apt update
sudo apt install build-essential libopenmpi-dev openmpi-bin libfftw3-dev
pip3 install numpy matplotlib
```

### 3.2 Compiling the Main Solver

```bash
mpicc -o fdtd.x fdtd.c -lm -lrt -Ofast -flto -funroll-loops
```

With an Intel compiler you can use `mpiicc -o fdtd.x fdtd.c -lm -O3 -xHost -ipo`.

> **Tip**: `-Ofast` enables `-ffast-math`. All constants and physical quantities in this program are finite values and do not rely on IEEE special-value semantics, so this is normally safe; if you modify the code in a way that introduces logic depending on NaN/Inf (for example using `isnan()` in a condition), switch to `-O3`.

### 3.3 Compiling the Frequency Analysis Tool

```bash
# if FFTW3 is installed in a standard path
gcc -o frequency_analysis frequency_analysis.c -lfftw3 -lm

# if FFTW3 is installed under a custom prefix (for example /opt/fftw-3.3.6)
gcc -o frequency_analysis frequency_analysis.c \
    -I/opt/fftw-3.3.6/include -L/opt/fftw-3.3.6/lib -lfftw3 -lm
```

> **FFTW3 under a custom prefix needs the prefix at two different times: at compile time (`-I`/`-L`) *and* at run time.** `frequency_analysis` links against the *shared* `libfftw3.so`, and the compiler flags are not recorded in a way that lets the loader find it. Compiling successfully is therefore not enough — without `LD_LIBRARY_PATH` (or an entry in `/etc/ld.so.conf` followed by `ldconfig`) the program stops immediately with
>
> ```
> ./frequency_analysis: error while loading shared libraries: libfftw3.so.3: cannot open shared object file: No such file or directory
> ```
>
> ```bash
> # 1) compile time — tell the compiler where the header and the library are
> gcc -o frequency_analysis frequency_analysis.c \
>     -I/opt/fftw-3.3.6/include -L/opt/fftw-3.3.6/lib -lfftw3 -lm
>
> # 2) run time — tell the dynamic loader where the same library is
> export LD_LIBRARY_PATH=/opt/fftw-3.3.6/lib:$LD_LIBRARY_PATH
> ./frequency_analysis simulation.cfg
> ```
>
> You can check which library the binary is looking for with `ldd ./frequency_analysis`; an entry reading `not found` means the run-time path is still missing. Putting the export in a job script (or in `~/.bashrc`) avoids having to repeat it for every run.

### 3.4 Compile-Time Constants

The following constants are defined as macros at the top of `fdtd.c`; after changing them you must **recompile**:

| Macro | Default | Meaning |
| --- | --- | --- |
| `EPS_INF` | `20.0` | **High-frequency permittivity ε∞** used by the ADE method (scalar, only for materials with `use_ADE=1`) |
| `N` | `3` | Dimension of the dielectric tensor/matrix inversion, fixed at 3 |
| `TRACK_MIN_PRINT` | `1e8` | Minimum print threshold for the memory log (bytes) |

---

## 4. Quick Start

### 4.1 Minimal Example

`examples/example.cfg` is a small case that finishes in a few minutes (80 × 80 × 48 grid + a small MoO3 block; measured ≈4.5 min with 4 MPI processes and ≈1.5 min with 16 processes on a 64-core machine); it is meant to verify compilation, the data files and the post-processing chain. Run the following in the repository root:

```bash
# 1) compile
mpicc -o fdtd.x fdtd.c -lm -lrt -Ofast

# 2) generate the medium distribution file medium_map.bin from the [medium] section of the config file
python3 generate_medium5.py examples/example.cfg medium_map.bin

# 3) run the FDTD (the number is the MPI process count, recommended ≤ nz)
mpirun -np 4 ./fdtd.x examples/example.cfg > out.log 2>&1

# 4) quick look at the results
python3 visualize_results2.py          # field slice snapshots

# 5) extract near-field frequency slices (field slices from the period in which the source is still oscillating must be deleted first)
#    (FFTW3 under a custom prefix: add -I/-L here and set LD_LIBRARY_PATH at run time — see 3.3)
gcc -o frequency_analysis frequency_analysis.c -lfftw3 -lm
./frequency_analysis examples/example.cfg
```

> **Run all five steps from the repository root, or copy the data along with the configuration file.**
> `examples/example.cfg` refers to the material data with **relative paths** (`Material_data/MoO3/...`), and `medium_file = ./medium_map.bin` is resolved against the **current working directory**, not against the directory the configuration file lives in. Running the commands above from somewhere else therefore fails with `Error opening medium file` or a missing `QeZV`/`QeZM`/`DMM` file.
>
> Because every output (`field_slice_*.bin`, `monitor_point.dat`, `full_monitor.dat`, `source.txt`) is also written to the current working directory, it is often preferable to run in a dedicated directory — but then the configuration file **and** `Material_data/` must be copied there:
>
> ```bash
> mkdir -p run && cp examples/example.cfg run/ && cp -r Material_data run/
> cd run
> python3 ../generate_medium5.py example.cfg          # writes ./medium_map.bin
> mpirun -np 4 ../fdtd.x example.cfg > out.log 2>&1
> python3 ../visualize_results2.py                    # scans the current directory
> ../frequency_analysis example.cfg                   # also scans the current directory
> ```
>
> (The two research-grade cases in [4.2](#42-research-grade-case-moo3-surface-near-field-under-dipole-excitation) already ship as self-contained directories and are run from inside them; the same rule applies to any case you copy elsewhere.)

During the run, standard output reports progress information of the form:

```
it 1200/20000 (6.0%), t = 4.57383e-14 s,|E|_max = 3.42e+05 V/m,r_max = 1.23e-03 A, elapsed: 000:00:37, ETA: 000:09:43
```

(where `r_max` is the **maximum modulus of the displacement over all atoms**, in Å), and standard error reports the memory usage. Individual arrays are only reported when they reach 0.1 GB (`TRACK_MIN_PRINT`, see [3.4](#34-compile-time-constants)), so a small case such as this one prints just the start-up summary:

```
[MEM] Before time loop: per-rank = 0.017825 GB, total (4 ranks) = 0.061462 GB
```

A production-scale run additionally reports every array above the threshold, for example:

```
[MEM] +1.234567 GB (Ex), total=1.234567 GB (est. all: 4.938268 GB)
```

### 4.2 Research-Grade Case: MoO3 Surface Near Field under Dipole Excitation

Under `examples/` there are two further cases of near-research scale. The two have **exactly the same geometry and source positions**; the only difference is "how material 4 (the MoO3 layer) is described", so they can be used as a pair to compare the two modeling routes:

| Case | Modeling of material 4 | Source frequency | Notes |
| --- | --- | --- | --- |
| `MoO3_zdipole_800cm/` | **Phonon coupling**: `natom=16` + `born`/`QeZV`/`QeZM`/`DMM`, the ionic equations of motion are solved at every grid point | 800 cm⁻¹ | Explicitly includes the optical-phonon degrees of freedom (PhP) |
| `Lorentz_zdipole_650cm/` | **Effective dispersive medium**: `dielectric=dielectric_MoO3_Lorentz.txt` + `use_ADE=1` | 650 cm⁻¹ | Replaces the phonon degrees of freedom by a Drude-Lorentz dielectric function table |

Settings shared by both:

| Item | Value | Notes |
| --- | --- | --- |
| Grid | `400 × 800 × 800`, `Δx = 5 nm`, `Δy = Δz = 10 nm` | Domain size 2 µm × 8 µm × 8 µm; x is the surface normal, hence the finer sampling along the normal |
| Time | `tsteps = 157000`, `dt = 9.5263e-18 s` | Total duration about 1.5 ps; `dt` is taken close to the three-dimensional CFL limit (determined by Δx = 5 nm) |
| Boundary | `pml_thickness = 150`, `pml_reflection = 1e-6` | 150 × 5 nm = 750 nm along x, 1.5 µm along y/z |
| Geometry (`cubes`) | `x ∈ [0,160]` → material 1 (built-in SiO2 substrate); `x ∈ [161,200]` → material 4 (MoO3, thickness 40 × 5 nm = 200 nm); `x > 200` is vacuum | Material 1 extends all the way to `x = 0` (through the PML), avoiding extra reflections from the substrate interface |
| Source | `(220, 400, 400)`, z-polarized dipole → 100 nm outside the MoO3 surface | Point-dipole near-field excitation |
| Output | `output_component = Ex`, `output_slice_axis = x`, `output_slice_position = 201` | i.e. the y-z near-field plane one cell (5 nm) outside the surface |
| Monitor point | `(200, 400, 400)` | Last MoO3 layer, `dir = x` |

Both case directories are **self-contained**: `QeZV_clean.txt`, `QeZM_clean.txt` and `DMM_clean.txt` in `MoO3_zdipole_800cm/` are identical to the files of the same names under `Material_data/MoO3/`, and `OUTCAR.txt` has the same content (line endings are LF, which is convenient for version control); `dielectric_MoO3_Lorentz.txt` in `Lorentz_zdipole_650cm/` is identical to the file of the same name under `Lorentz_dielectric/`. This way each case can be copied elsewhere on its own and still run.

The configuration files of both cases use **relative paths** (`medium_file = ./medium_map.bin`, with bare file names for the material data), so they must be run **inside their respective directories**:

```bash
cd examples/MoO3_zdipole_800cm            # or Lorentz_zdipole_650cm

# generate the medium distribution (about 1.02 GB, not shipped with the repository)
python3 ../../generate_medium5.py simulation.cfg

# run (adjust the process count to the available memory, see "Resource requirements" below)
mpirun -np 28 ../../fdtd.x simulation.cfg > out.log 2>&1

# post-processing (frequency_analysis only has to be compiled once)
python3 ../../visualize_results2.py
gcc -o frequency_analysis ../../frequency_analysis.c -lfftw3 -lm
./frequency_analysis simulation.cfg
```

> **Resource requirements** (rough estimates, for choosing the process count and the disk quota)
>
> | Item | `MoO3_zdipole_800cm` | `Lorentz_zdipole_650cm` |
> | --- | --- | --- |
> | Field arrays (9 components) | \~18.4 GB | \~18.4 GB |
> | `medium` | \~1.0 GB | \~1.0 GB |
> | `r`/`v` (only the 10% of grid points in MoO3) | \~19.7 GB | — |
> | `P_ADE`/`P_ADEo` | — | \~1.2 GB |
> | CPML auxiliary fields `ψ` | \~20.5 GB | \~20.5 GB |
> | **Total memory** | **\~60 GB** | **\~41 GB** |
> | `medium_map.bin` | 1.02 GB | 1.02 GB |
> | `field_slice_*.bin` | \~4.0 GB (1571 slices) | \~4.0 GB |
>
> Memory is decomposed over processes along z: with 28 processes each process holds about 1.8 GB, with 56 processes about 0.9 GB; processes whose z range overlaps the z-direction PML additionally allocate about 0.3 GB of z-normal `ψ` arrays.
>
> **Output notes**: the monitor point of `MoO3_zdipole_800cm` falls on a material with `natom > 0`, so the program **automatically enables** `full_monitor` and additionally produces `full_monitor.dat` (1571 rows × 106 columns, containing the displacements and velocities of all 16 atoms); material 4 of `Lorentz_zdipole_650cm` contains no atoms, the `atom=`/`dir=` entries in `monitor_point` have no effect, and the output has only the 10 field-component columns.
>
> **Sampling and frequency resolution**: `output_interval = 100` corresponds to a slice interval of 9.5263e-16 s, i.e. 1571 slices in total; the frequency resolution of an FFT over these data is **22.3 cm⁻¹** (Nyquist about 17500 cm⁻¹). When a finer frequency grid is needed, reduce `output_interval` or increase `tsteps`; before computing a near-field spectrum, remember to delete the slices from the period in which the source is still oscillating (around `t < t0 + 6·tau`).

---

## 5. Configuration File Reference

The configuration file is **plain text**, parsed line by line, with the following rules:

- Lines starting with `#` are comments; empty lines are ignored. **Inline comments (a trailing `#`) are not supported** and are treated as part of the value.
- Section headers such as `[grid]` and `[medium]` are merely **decorative** for the solver (they are skipped during parsing), but `generate_medium5.py` **relies on the `[medium]` section header** to locate the medium geometry definitions, so keep them.
- `source ...` lines and `material ...` lines must **start in column 0** (the parser tests them with `strstr(line, "source") == line`); no leading spaces are allowed.
- Ordinary parameter lines have the form `key = value` (spaces around the equals sign are optional, and the value may contain spaces).
- Inside `source` / `material` lines, multiple `key=value` pairs are separated by **commas**; a `value` **must not contain spaces or commas** (it is parsed with `%s`), so file paths must not contain spaces.

### 5.1 Grid and Run Parameters

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `nx` | int | 50 | Number of grid cells along x (global) |
| `ny` | int | 50 | Number of grid cells along y (global) |
| `nz` | int | 50 | Number of grid cells along z (global; MPI decomposes along this direction) |
| `tsteps` | int | 500 | Number of time steps |
| `delta_x` | double | 0.01 | Spatial step along x (**meters**) |
| `delta_y` | double | 0.01 | Spatial step along y (meters) |
| `delta_z` | double | 0.01 | Spatial step along z (meters) |
| `dt` | double | automatic | Time step (**seconds**). If `dt ≤ 0` or it exceeds the CFL limit, `dt = 0.99 · min(Δx,Δy,Δz) / (c√3)` is used automatically |
| `pml_thickness` | int | 10 | CPML layer thickness (in grid cells); the same thickness is used on all six faces |
| `pml_reflection` | double | 0.001 | CPML target reflectivity (used to compute `σ_max`); the smaller, the stronger the absorption |
| `output_interval` | int | 20 | Output interval (in steps) for field slices and `full_monitor`; `monitor_point` **outputs every step** |
| `output_component` | string | `Ex` | Output component, one of `Ex`/`Ey`/`Ez`/`E` (modulus)/`Dx`/`Dy`/`Dz` |
| `output_slice_axis` | char | `z` | Normal axis of the slice, `x`/`y`/`z` |
| `output_slice_position` | int | `nz/2` | Slice position (grid index) |
| `medium_file` | string | `medium_map.bin` | Path of the binary medium distribution file |
| `monitor_point` | list | off | Monitor point, see [5.4](#54-monitor-points) |
| `monitor_output_file` | string | `monitor_point.dat` | Output file name of the single-point monitor |
| `full_monitor` | int | 0 | Set to 1 to write "fields + displacements/velocities of all atoms" to a file |
| `full_monitor_output_file` | string | `full_monitor.dat` | Output file name of the full monitor |

> **Not implemented**: `bc_x` / `bc_y` / `bc_z` (periodic boundaries), which appear in the configuration file examples, are **not parsed**; the program silently ignores them. The current version only supports the CPML absorbing boundary.

**Important reminder about `dt`**: the program silently replaces an oversized `dt` by the CFL limit value, but it does **not** adjust `tsteps` automatically. Hence the total physical duration you intended to simulate, `tsteps × dt`, changes accordingly; if you specify `dt` explicitly, check `tsteps` as well.

> Example: with `delta_* = 10e-9 m` the CFL limit is `dt ≈ 1.9e-17 s`. If the configuration says `dt = 1e-15` (50 times larger), the program silently switches to `1.9e-17 s`, and then `tsteps = 50000` corresponds to only about 0.96 ps instead of 50 ps. Always rely on the `Time steps: ..., dt= ...` line printed at start-up.

### 5.2 Excitation Source `source`

```
source type=<...>, x=<int>, y=<int>, z=<int>, amplitude=<double>, component=<x|y|z>, \
       frequency=<Hz> | wavenumber=<cm^-1>, t0=<s>, tau=<s>, start_time=<s>, end_time=<s>
```

| Key | Default | Description |
| --- | --- | --- |
| `type` | `gaussian` | `gaussian`, `sinusoidal`, `ricker`, `dipole` |
| `x` / `y` / `z` | — | Grid index of the source; please always give them explicitly |
| `amplitude` | 1.0 | Amplitude |
| `component` | `z` | Injected component: `x`/`y`/`z` → added to `Dx`/`Dy`/`Dz` respectively |
| `frequency` | 1e9 | Frequency (Hz), used by `sinusoidal`/`ricker`/`dipole` |
| `wavenumber` | — | Wavenumber (cm⁻¹), converted as `frequency = wavenumber × 0.03e12` Hz |
| `t0` | — | Center time of the pulse (s); please always give it explicitly |
| `tau` | — | Pulse width (s); please always give it explicitly |
| `start_time` | 0 | Time at which the source starts (s) |
| `end_time` | `tsteps × dt` | Time at which the source stops (s) |

Time-domain waveforms (`A` = `amplitude`, `f` = `frequency`):

| Type | Expression |
| --- | --- |
| `gaussian` | `A · exp(−((t − t0)/tau)²)` |
| `sinusoidal` | `A · sin(2πft)` |
| `ricker` | `A · (1 − 2π²f²(t−t0)²) · exp(−π²f²(t−t0)²)` |
| `dipole` | `A · sin(2πft) · exp(−((t − t0)/tau)²)` |

For the `dipole` type, **exactly one** of `frequency` or `wavenumber` must be given, otherwise the program reports an error and exits.

> **Note 1**: the source is a "soft source" — the current term is added directly to `D` (`D += value · dt`), it does not overwrite the field values.
>
> **Note 2**: it is recommended to **write out `x`, `y`, `z`, `t0` and `tau` explicitly, item by item**. Source lines are parsed in the first pass, at which point keys such as `nx` and `dt` have not been read yet, so the defaults of these items are based on the compile-time initial values rather than on your configuration.
>
> **Note 3**: the `frequency` of the **first** `source` line is recorded as `unique_source_frequency` and is used to:
> 1. pick the row **closest to that frequency** from the dielectric function table given by `dielectric=`;
> 2. compute the ADE coefficients (`γ`, `ω_p²`) of materials with `use_ADE=1`.
>
> Therefore put the **source corresponding to the main frequency first**. When several sources have different frequencies, the ADE/dielectric function is exact only at the frequency of the first source.
>
> **Note 4**: the source position is given by the three independent keys `x=`, `y=`, `z=` (a notation such as `position=(10,10,10)` is not part of this program's syntax).

### 5.3 Material `material`

```
material id=<int>, name=<string>, eps_xx=..., ..., eps_zzi=..., sigma=..., sigma_m=..., \
         dielectric=<file>, use_ADE=0|1, anglez=<rad>, anglex=<rad>, switchxz=0|1, \
         natom=<int>, born=<file>|[array], QeZV=<file>, QeZM=<file>, DMM=<file>
```

| Key | Description |
| --- | --- |
| `id` | **Required**. Material number; `0`–`3` are built-in materials, custom materials **must start from `4`** |
| `name` | Material name (at most 63 characters), used for printing only |
| `eps_xx` … `eps_zz` | Real part of the dielectric tensor (9 components; you do **not** have to enter the symmetric entries, the code uses them according to symmetry) |
| `eps_xxi` … `eps_zzi` | Imaginary part of the dielectric tensor (9 components, convention **≥ 0**), treated as loss |
| `sigma` | Electrical conductivity σ (S/m), default 0 |
| `sigma_m` | Magnetic loss σ_m (Ω/m), default 0 |
| `dielectric` | Name of the dielectric function table file. When set, a row is interpolated at `unique_source_frequency` and **overwrites** `eps_*` and `eps_*i` (see [7.4](#74-dielectric-function-table-dielectric)) |
| `use_ADE` | `1` enables the ADE method (must be 1 when the real part of the dielectric function is negative), `0` uses the ordinary `E = ε⁻¹D/ε₀` |
| `anglez` | Rotation angle of the material about the **z axis**, in **radians** (clockwise/counter-clockwise, see [9.4](#94-anisotropic-constitutive-relation-and-coordinate-rotation)) |
| `anglex` | Rotation angle of the material about the **x axis**, in radians. `anglez` takes precedence over `anglex` (when both are non-zero only `anglez` is used) |
| `switchxz` | When `1`, swaps the XZ axes of the material (acting on `eps`, `born`, `QeZV`, `QeZM`, `DMM` alike) |
| `natom` | Number of atoms in the primitive cell. When `> 0` the program enters "ion coupling" mode and **must** be given `QeZV`, `QeZM`, `DMM` at the same time, otherwise `MPI_Abort` |
| `born` | If the value contains no `[`, it is treated as a **VASP OUTCAR file name** (the Born effective charges and the macroscopic static dielectric tensor are read from it); if it has the form `[v1,v2,...]`, then `3·natom·3` numbers are given directly |
| `QeZV` | File with `9·natom` numbers (see [7.2](#72-phonon-polariton-data-qezv--qezm--dmm)) |
| `QeZM` | File with `9·natom` numbers |
| `DMM` | File with `9·natom²` numbers |

**Built-in materials (`id = 0…3`)**

| id | Name | Description |
| --- | --- | --- |
| 0 | Vacuum | ε = 1 |
| 1 | SiO2 | Drude-Lorentz model (evaluated at `unique_source_frequency`); if the real part of ε at that frequency is ≤ 0, `use_ADE` is enabled automatically |
| 2 | Uniaxial Crystal | ε = diag(8, 7, 5) |
| 3 | Lossy Medium | ε = 9, σ = 0.1 S/m |

At start-up the program prints the complete information of every material (dielectric tensor, σ, use_ADE, rotation angles, atom count), which you can use to check that the configuration was parsed as intended.

> **On the behavior of `born=<OUTCAR>`**: setting `born=` makes the program read `MACROSCOPIC STATIC DIELECTRIC TENSOR` from the OUTCAR and **overwrite** the `eps_*` of that material (the imaginary part is set to 0); the Born charge array that is read **currently does not participate in any field/ion update** (the ion coupling is carried entirely by `QeZV`/`QeZM`/`DMM`). Hence, for phonon-polariton calculations, note that the OUTCAR gives the **static** dielectric tensor ε(0), whereas the coupling formula `E = (D − P_ion)/(ε₀ε∞)` should use the **high-frequency (clamped-ion) dielectric tensor ε∞**; using the static value directly double-counts the lattice contribution. For PhP materials it is recommended to use `dielectric=` or to give ε∞ explicitly through `eps_xx=...`.

### 5.4 Monitor Points

```
monitor_point = x=500, y=500, z=250, atom=0, dir=z
monitor_output_file = monitor_point.dat
```

- `x`, `y`, `z`: grid indices of the monitor point (global).
- `atom`: atom number (starting from 0); `dir`: displacement direction, `x`/`y`/`z`.
- If the material at that point has `natom > 0` and `atom < natom`, the displacement `R` and the velocity `V` of that atom are output; otherwise only the 9 field components are output.
- When **no `monitor_point` line** is present, the feature is off (`monitor_enabled` defaults to 0).
- When the monitor point falls on a material with `natom > 0`, the program **automatically enables** `full_monitor`.

### 5.5 Complete Example

```ini
# ===== Grid and time =====
[grid]
nx = 400
ny = 400
nz = 200
delta_x = 10e-9
delta_y = 10e-9
delta_z = 10e-9
tsteps = 20000
# if dt is omitted the CFL limit is used automatically
pml_thickness = 20
pml_reflection = 1e-3
output_interval = 50
output_component = Ez
output_slice_axis = x
output_slice_position = 200

# ===== Excitation source (the frequency of the first source determines the ADE / dielectric function values) =====
source type=dipole, x=200,y=200,z=100, amplitude=1.0, wavenumber=930, component=x, t0=0.2e-12, tau=0.05e-12

# ===== Medium geometry (used by generate_medium5.py only) =====
[medium]
cubes = [ {"xrange": (0, 399), "yrange": (0, 399), "zrange": (60, 140), "material": 4} ]

# ===== Materials =====
material id=4, name=MoO3, natom=16, born=Material_data/MoO3/OUTCAR.txt, \
  QeZV=Material_data/MoO3/QeZV_clean.txt, QeZM=Material_data/MoO3/QeZm_clean.txt, \
  DMM=Material_data/MoO3/DMM_clean.txt
# the backslash line continuations in the three lines above are invalid! material must be written on one line

material id=5, name=Lorentz_MoO3, dielectric=Lorentz_dielectric/dielectric_MoO3_Lorentz.txt, \
  use_ADE=1, anglex=0.523598, switchxz=1

medium_file = ./medium_map.bin
monitor_point = x=200,y=200,z=100,atom=0,dir=z
monitor_output_file = monitor_point.dat
full_monitor = 1
full_monitor_output_file = full_monitor.dat
```

> ⚠️ The `\` line continuations above are **only a typesetting device of this document**. The program **does not support `source` / `material` definitions spanning several lines**; every `material` and `source` must be written on a **single line**. Please refer to the style used in `examples/example.cfg`.

### 5.6 `[medium]` Geometry Definition (used by `generate_medium5.py`)

`generate_medium5.py` parses the `[medium]` section with regular expressions + `ast`, and supports three kinds of geometric bodies (they can be used together and **overwrite each other in the order `cubes → hexahedra → cylinders`**):

```ini
[medium]
# cube (grid index ranges, closed intervals)
cubes = [ {"xrange": (0, 999), "yrange": (0, 999), "zrange": (187, 250), "material": 4} ]

# arbitrary parallelepiped (vertices ordered "small z first, large z second, counter-clockwise within each xy plane")
hexahedra = [
    { "vertices": [ (201,406,593), (204,406,593), (204,408,593), (201,408,593),
                    (201,406,595), (204,406,595), (204,408,595), (201,408,595) ],
      "material": 6 }
]

# cylinder: p1/p2 are the two end points (grid indices), radius is in meters and is converted using delta_x/y/z
cylinders = [ {"p1": (212, 500, 506), "p2": (212, 700, 494), "radius": 120e-9, "material": 6} ]
```

> The `spheres`, `slabs` and `points` keywords that may appear in the configuration file examples are **not implemented in the current version**; writing them has no effect.
>
> Between the `[ ]` of `hexahedra` and `cylinders` **do not put comments and do not insert line breaks** (`cubes` does support inline comments). Grid points outside any geometric body (not covered by any body) automatically become material 0 (vacuum).

---

## 6. Units and Physical Conventions

The whole program uses the **SI system of units**.

| Physical quantity | Unit | Notes |
| --- | --- | --- |
| Spatial steps `delta_*`, `radius` | m | grid index × step = physical coordinate |
| Time `dt`, `t0`, `tau` | s | |
| Frequency `frequency` | Hz | |
| Wavenumber `wavenumber` | cm⁻¹ | conversion: `f [Hz] = wavenumber × 0.03e12` (i.e. c = 3×10¹⁰ cm/s; differs from the true vacuum value 2.998×10¹⁰ cm/s by about 0.07%) |
| Electric field `E` | V/m | |
| Magnetic field `H` | A/m | |
| Electric displacement `D` | C/m² | |
| Conductivity `sigma` | S/m | |
| Magnetic loss `sigma_m` | Ω/m | |
| Atomic displacement `r` | m | converted to Å when printed (`r/1e-10`) |
| Atomic velocity `v` | m/s | |
| Dielectric tensor `eps_*` | dimensionless (relative permittivity) | imaginary part convention **≥ 0** |

---

## 7. Input File Formats

### 7.1 Medium Distribution `medium_map.bin`

A **binary** file generated by `generate_medium5.py` (native byte order, `int32`):

```
offset 0      : int32 nx
offset 4      : int32 ny
offset 8      : int32 nz
offset 12     : int32 medium[nx][ny][nz], x slowest, y next, z fastest
                i.e. the value at (i,j,k) is at  12 + 4·((i·ny + j)·nz + k)
```

- The `nx/ny/nz` in the file header must **match exactly** the grid dimensions in the configuration file, otherwise the program reports an error and exits.
- The values are **material ids**; `0` means vacuum.
- The file size should be `12 + 4·nx·ny·nz` bytes.
- When reading, the program decomposes the file along `z` and distributes it to the MPI processes (equivalent to `MPI_Scatterv`); only rank 0 actually reads from disk.

### 7.2 Phonon-Polariton Data `QeZV` / `QeZM` / `DMM`

All three files are **plain text with one number per line** (read sequentially with `fscanf("%lf")`; whitespace-separated is fine). Let `natom = N`:

| File | Number of elements | Memory index | Meaning and dimension |
| --- | --- | --- | --- |
| `QeZV_*.txt` | `9N` | `[iatom][iE][idir]`, atom-major | makes the polarization `P_iE = Σ_{idir} QeZV·r_idir` (C/m²), dimension **C/m³**; physically ≈ `Z*e/Ω_cell` |
| `QeZM_*.txt` | `9N` | `[iatom][iE][idir]`, atom-major | makes the acceleration `a_idir = Σ_{iE} QeZM·E_iE` (m/s²), dimension **C/kg**; physically ≈ `Z*e/m_atom` |
| `DMM_*.txt` | `9N²` | `[iatom][idir][jatom][jdir]`, `iatom` slowest, `jdir` fastest | mass-weighted dynamical matrix, dimension **s⁻²** (eigenvalues are ω²) |

That is, the file order is:

```
QeZV/QeZM : the 9 numbers of atom0, the 9 numbers of atom1, ...
            within each atom grouped by iE = x,y,z, and within a group by idir = x,y,z
DMM       : iatom from 0 to N-1
              for each iatom, idir = x,y,z
                for each idir, jatom from 0 to N-1
                  for each jatom, jdir = x,y,z
```

**Self-consistency requirement**: the three data sets of the same material must come from the same first-principles calculation and use the same primitive-cell volume and atomic-mass convention, and must satisfy

```
QeZV / QeZM = m_atom / Ω_cell
```

i.e. "the mass of a given atom divided by the volume of the primitive cell it belongs to". The data in the repository can be used to check this:

| Material | First atom | `QeZV/QeZM` | Mass of that atom / POSCAR primitive-cell volume |
| --- | --- | --- | --- |
| hBN | B | 1.25086e10 / 2.41267e7 = 518.45 | 10.811 amu / 34.626 Å³ = 1.7956e-26 kg / 3.4626e-29 m³ = 518.5 |
| MoO3 | Mo | 794.60 | 95.96 amu / 200.54 Å³ = 1.5934e-25 kg / 2.0054e-28 m³ = 794.6 |

Agreement between the two shows that `QeZV`, `QeZM` and the `POSCAR` primitive-cell volume and atomic masses follow the same convention.

> Note that `DMM` uses the **equation-of-motion form** (`D = Φ(i,j)/m_i`, obtained from the mass weighting of the Γ-point dynamical matrix), so it is **not a symmetric matrix**; do not treat it as a symmetric matrix and diagonalize it directly. Its eigenvalues are `ω²` (rad²/s²), and three acoustic modes close to 0 should appear at the low-frequency end, which can be used to verify the integrity of the data.

> The coordinate conventions of the material data files differ (for example hBN: the `z` axis is perpendicular to the hBN plane; MoO3: `x=[100]`, `y=[001]`, `z=[010]`). When building a model, use `switchxz` / `anglez` / `anglex` to rotate the crystal axes into the simulation coordinate system.
>
> The capitalization of the file names is **inconsistent**: the hBN directory contains `QeZM_clean.txt`, the MoO3 directory contains `QeZm_clean.txt`. On case-sensitive systems such as Linux, copy the actual file names exactly.

**Data provenance**: these files come from first-principles calculations — the Born effective charge tensors come from static VASP calculations (the `BORN EFFECTIVE CHARGES` section of the `OUTCAR`), and `DMM` is the result of mass-weighting the Γ-point dynamical matrix. The hBN and MoO3 data under `Material_data/` in this repository can be used directly; **the complete workflow and scripts needed to generate the data yourself are not part of this repository**, so to extend to other materials please prepare them yourself according to the physical definitions and dimensional conventions given above (or contact the author).

> The coordinate convention must be consistent with `POSCAR`: the order of the coordinate axes of `QeZV`/`QeZM`/`DMM` corresponds to the lattice orientation of the `POSCAR`; if the data files have been `switchxz`-ed while the `POSCAR` has not been swapped accordingly, the two no longer describe the same coordinate system.


### 7.3 Born Effective Charges `OUTCAR.txt`

A VASP output fragment; the parser locates it by two keywords:

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

- Dielectric tensor section: 3 consecutive rows × 3 numbers (row numbers may be present).
- Born section: `ion <n>` starts an atom (`n` starting from 1), followed by 3 rows, each of the form `direction number + 3 numbers`.
- This file is only read when the `born=` key is given, and `natom` is required to match it (number of `ion` entries ≥ `natom`).

### 7.4 Dielectric Function Table `dielectric=`

Plain text, **15 columns** per line (lines starting with `#` are comments):

| Column | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meaning | f (Hz) | λ | wavenumber (cm⁻¹) | ε_xx | ε_xy | ε_xz | ε_yy | ε_yz | ε_zz | ε_xxi | ε_xyi | ε_xzi | ε_yyi | ε_yzi | ε_zzi |

- The program scans the whole table and selects the row whose **f is closest to `unique_source_frequency` (the frequency of the first source)**; if the relative deviation is > 10% a warning is issued but the row is still used.
- Only the 6 independent components are read (real and imaginary parts of `xx, xy, xz, yy, yz, zz`) and filled in as a symmetric tensor (`yx=xy`, `zx=xz`, `zy=yz`).
- The second column (wavelength) is not used by the program and may be set to 0.
- The files in the `Lorentz_dielectric/` directory are generated by the MATLAB scripts in the same directory, over the frequency range 1–2000 cm⁻¹ with a step of 0.3 cm⁻¹:

```matlab
% Lorentz_MoO3.m
epsfx=4.0; epsfy=5.2; epsfz=2.4;
wLOx=972; wLOy=851; wLOz=1004;      % cm^-1
wTOx=820; wTOy=545; wTOz=958;       % cm^-1
gx=4; gy=4; gz=2;
% ε(ω) = ε∞·(1 + (ω_LO² − ω_TO²)/(ω_TO² − ω² − iγω))
```
Changing only the parameters at the top of the script is enough to generate a dielectric function table for a new material (anisotropic Drude-Lorentz).

### 7.5 Structure File `POSCAR`

`Material_data/hBN/` and `Material_data/MoO3/` each contain a `POSCAR` in VASP format (line 1 comment, line 2 scaling factor, lines 3–5 lattice vectors, line 6 element symbols, line 7 atom counts, fractional coordinates from line 8 on), used to check the primitive-cell volume, the atomic species and the atomic order — the axis conventions and atom numbering of `QeZV`/`QeZM`/`DMM` are based on it (see [7.2](#72-phonon-polariton-data-qezv--qezm--dmm)).

---

## 8. Output File Formats

| File | Generated when | Frequency | Format |
| --- | --- | --- | --- |
| `field_slice_<axis><pos>_<step:08d>.bin` | always | every `output_interval` steps | binary: two `int32` dimensions + `float32` data; the layout depends on the slice axis, see below |
| `monitor_point.dat` | when `monitor_point` is set | **every step** | text, `#` header + one line per step `Time(s) Ex Ey Ez Dx Dy Dz Hx Hy Hz [R V]` |
| `full_monitor.dat` | `full_monitor=1` (or enabled automatically when the monitor point lies on a material containing atoms) | every `output_interval` steps | text, `#` header + one line per step `Time(s)` + 9 field components + `r{x,y,z}` and `v{x,y,z}` of every atom |
| `source.txt` | always | every `output_interval` steps | text, one line per source (the source amplitude at that instant); the columns are separated by **two spaces**, not tabs, so split on whitespace |
| `frequency_slice_<idx>_<f>Hz_<k>cm-1.txt` | generated by `frequency_analysis` | — | text, see below |

**Field slice** file name examples: `field_slice_x201_00000100.bin`, `field_slice_z100_00000500.bin`. The file content is "two `int32` dimensions + `float32` data", where the data are arranged by the two remaining coordinate axes, with the **axis written first varying slowest and the axis written last varying fastest**:

| `output_slice_axis` | File header | Data length | Data index (slow → fast) |
| --- | --- | --- | --- |
| `z` | `nx`, `ny` | `nx·ny` | `field[i][j]` → `i*ny + j` |
| `x` | `ny`, `nz` | `ny·nz` | `field[j][k]` → `j*nz + k` |
| `y` | `nx`, `nz` | `nx·nz` | `field[i][k]` → `i*nz + k` |

For example `field_slice_x201_*.bin` is the **y-z plane** at `x = 201`, so `reshape((ny, nz))` suffices (both `visualize_results2.py` and `frequency_analysis` read it that way).

**`monitor_point.dat` columns**:

```
# Time(s)  Ex  Ey  Ez  Dx  Dy  Dz  Hx  Hy  Hz  R  V
```

When `atom >= natom` (or the material at that point has no atoms), the trailing `R` and `V` columns do not exist.

**`full_monitor.dat` columns**:

```
# Time(s) Ex Ey Ez Dx Dy Dz Hx Hy Hz  r0_x r0_y r0_z v0_x v0_y v0_z  r1_x ... v{N-1}_z
```

Number of columns = `10 + 6·natom`; the first row is the `#` header (tab-separated column names of the form `r0_x`, `v0_z`) and can be read directly with `numpy.loadtxt(..., skiprows=1)` or MATLAB `dlmread(..., '\t', 1, 0)`.

**`frequency_slice_*.txt` columns** (a temporal FFT is performed at every spatial point):

```
# Frequency slice at <f> Hz (index: <n>)
# ... comment lines such as FFT parameters and slice dimensions ...
# Format: x_index y_index real_part imag_part magnitude phase(rad) phase(deg)
< i  j  Re  Im  |E|  φ(rad)  φ(deg) >
```

Note: only points in the central region of the slice (`0.25·n < i,j < 0.75·n`) are output, in order to keep the file size down.

---

## 9. Numerical Method

### 9.1 Time Stepping (Leapfrog Scheme)

Each time step performs, in order:

1. **Exchange the z-direction halo of the E field** (so that the magnetic-field update can use values outside the boundary);
2. **Update H** (explicit finite-difference curl + CPML correction);
3. **Exchange the z-direction halo of the H field**;
4. **Update D** (`∂D/∂t = ∇×H`, with the CPML correction);
5. **Inject the source** (`D += J·dt`, soft source);
6. **Obtain E from D** (anisotropic constitutive relation / ADE), updating the ionic displacements `r` and velocities `v` at the same time;
7. **Output field slices, `full_monitor` and `source.txt` according to `output_interval`; output `monitor_point` every step**.

The time step is clamped by the three-dimensional CFL condition:

```
dt = 0.99 · min(Δx, Δy, Δz) / (c·√3)
```

(This value is adopted automatically when the configured `dt` is larger than it or negative.)

### 9.2 CPML Absorbing Boundary

The implementation uses Roden & Gedney's convolutional PML (CPML); each of the six faces uses its own set of `σ(ρ)`, `κ(ρ)`, `α(ρ)` profiles, where `ρ` is the depth from the inner PML interface:

```
r      = (L − i)/L                       # 0 at the inner interface, 1 at the outermost layer
σ(r)   = σ_max · r^m
κ(r)   = 1 + (κ_max − 1) · r^m
α(r)   = α_max · (1 − r)
σ_max  = −(m+1)·ln(R₀) / (2·η₀·Δ·L)
```

where `m = 3` (polynomial order), `κ_max = 7`, `α_max = 0.05`, `η₀ = √(μ₀/ε₀) ≈ 377 Ω`, `R₀ = pml_reflection`, `L = pml_thickness`.

**Tuning recommendations** (the usual engineering ranges for CPML):

| Parameter | Usual range | Default in this program |
| --- | --- | --- |
| `m` (order) | 3 ~ 4 | 3 |
| `κ_max` | 3 ~ 15 | 7 |
| `α_max` | 0.005 ~ 0.1 (often taken as `σ_max/κ_max`) | 0.05 |
| `R₀` (`pml_reflection`) | 1e-6 ~ 1e-8 | 1e-3 |
| `pml_thickness` | 10 ~ 20 cells | 10 |

> Note that the default `pml_reflection = 1e-3` is much looser than the recommended values; for problems sensitive to reflections (for example when a clean near-field phase is required) it is advisable to set it explicitly to `1e-6` or smaller.

The field updates use the recursive form with the auxiliary variable `ψ` (one `ψ` for every coordinate direction that has a PML and every H/D component):

```
b = exp(−(σ/(κε₀) + α/ε₀)·Δt)
c = σ/(κ²ε₀) · (b − 1) / (σ/(κε₀) + α/ε₀)
ψ^{n+1} = b·ψ^n + c·(spatial derivative)
update  = …/κ + ψ
```

### 9.3 Obtaining E from D (Anisotropy + Loss)

For every grid point (`mat` is the material at that point), let `x = σΔt/(2ε₀)`:

```
P_ion = Σ_atom Z*ᵀ·r                          # polarization contributed by the ionic displacements (crystal coordinate system)
D'    = D − P_ion
if use_ADE = 0:  E = ε⁻¹·D' / (ε₀·(1+x)²)     # anisotropic constitutive relation + conductive loss
if use_ADE = 1:  E = (D' − P_ADE) / (ε₀·EPS_INF·(1+x))
```

That is: `D` and `E` are first rotated into the material crystal coordinate system, `E` is obtained with the inverse of the material permittivity tensor, and the result is rotated back into the simulation coordinate system; the loss due to `σ` is applied as a factor `1/(1+x)` (when `use_ADE=0` this factor appears twice in the code, so it is equivalent to `1/(1+x)²`, which to first order equals `exp(−σΔt/ε₀)`). `σ_m` is the magnetic loss parameter and does not take part in the update in the current version.

### 9.4 Anisotropic Constitutive Relation and Coordinate Rotation

- **About the z axis** (`anglez = θ`): `E_crystal = R_z(θ)·E` (`R_z` is the standard rotation matrix); after the update, the fields are transformed back to the simulation frame with `R_z(−θ)`.
- **About the x axis** (`anglex = φ`): likewise, a rotation in the y-z plane.
- **`switchxz = 1`**: the x/z components of the tensors are swapped directly at read-in time (`ε_xx ↔ ε_zz`, `ε_xy ↔ ε_zy`, `ε_yx ↔ ε_yz`, `ε_xz ↔ ε_zx`, together with the corresponding imaginary parts; `born`/`QeZV`/`QeZM`/`DMM` are handled as well). This is cheaper than a rotation and suits the case where "the crystal axes differ from the simulation axes by a single permutation".
- If `anglez` is non-zero, `anglex` is ignored.

### 9.5 Negative Permittivity and the ADE Method

When the real part of the dielectric function of a material is negative at some frequency (metals, the reststrahlen band of polar crystals, etc.), inverting the constitutive relation directly gives non-physical results, and `use_ADE=1` should then be set. The program writes the dispersive dielectric function in the Drude / Drude-Lorentz form of a "high-frequency background ε∞ + tensor-form oscillator":

```
ε(ω) = ε∞·I − ω_p² / (ω² + i·γ·ω)        (in the anisotropic case γ and ω_p² are 3×3 tensors)
```

This yields the time-domain equation of motion for the polarization (the auxiliary differential equation, ADE):

```
d²P/dt² + γ·dP/dt = ε₀·ω_p²·E
```

which is discretized in the program as (`P_ADE` is the current step, `P_ADEo` the previous one):

```
(1 + γΔt/2)·P^{n+1} = 2P^n − (1 − γΔt/2)·P^{n−1} + ε₀Δt²·ω_p²·E^n
E = (D − P^{n+1}) / (ε₀·EPS_INF)
```

where the tensors `γ` and `ω_p²` are derived from the **frequency of the first source** ω and the `eps_*`/`eps_*i` of the material (`εr1 = ε_real − EPS_INF·I` must be invertible; `γ = −ω·ε_imag·εr1⁻¹`; `ω_p² = −ω²·εr1 + ω·γ·ε_imag`), so that the dielectric function at ω is reproduced exactly as the input value.

> **Important**: `EPS_INF` is a **compile-time constant** (`#define EPS_INF 20.0`). It only affects materials with `use_ADE=1`; if the actual ε∞ of the material differs from it, change it and **recompile** (the program reports `EPS_INF is too small` when `εr1` is not invertible).
>
> **Important**: `γ` and `ω_p²` are exact only at the **frequency of the first source**; for multi-frequency or broadband sources the ADE is only an approximation at a single frequency point. In addition, the ADE assumes that the dispersion is a "single oscillator + constant background"; before broadband simulations, check the dielectric function table with the scripts in `Lorentz_dielectric/`.

### 9.6 Ionic Equations of Motion (PhP Coupling)

For materials with `natom > 0`, every grid point additionally carries `3·natom` displacement degrees of freedom (`r`) and velocities (`v`), which evolve in the material crystal coordinate system:

```
a_i = Σ_j Z*_ij · E_j − Σ_{j,β} DMM_{iα,jβ} · r_{jβ}     # acceleration
r ← r + v·Δt + ½·a·Δt²                                    # velocity Verlet
v ← v + a·Δt
P_ion,i = Σ_j Z*_ij · r_j                                  # fed back into Maxwell's equations
```

- `Z*` is given by `QeZM` (acceleration coupling) and `QeZV` (polarization coupling), and `DMM` is the mass-weighted dynamical matrix.
- This is a **symplectic** velocity Verlet step, advanced alternately with the Maxwell part, so that the total energy is well conserved under stable conditions.
- The `r` and `v` arrays are **allocated only for grid points with `natom > 0`**, so the memory is proportional to the "volume of the material containing atoms".
- Stability requires `Δt < 2/√λ_max(DMM)` (determined by the highest phonon frequency); if the simulation diverges, first check whether the units of `Δt` and `DMM` are self-consistent.

> ⚠️ **Divergence protection**: every `output_interval` steps the program checks the global maximum displacement, and once `r_max > 1e-10 m` (i.e. 1 Å) it considers the ionic subsystem to have diverged, prints diagnostic information and then **calls `MPI_Abort` to terminate immediately**. The diagnostics include: the step at which the threshold was exceeded, the location of the maximum displacement `(i, j, k)`, the atom number and direction, and the velocity `v` and acceleration `a` at that point (decomposed into the `QeZM·E` driving force and the `DMM·r` restoring force), from which you can judge whether the field is too strong, whether the data units are inconsistent, or whether `dt` is too large:
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
> Note that the 1 Å threshold also means that the model is valid only in the small-displacement regime where the **linear (harmonic) approximation** holds — beyond that regime the results are no longer trustworthy even if the termination is not triggered.

### 9.7 Equivalence of the Dielectric Imaginary Part and Conductivity

The imaginary part of the dielectric function of a material, `eps_*i`, and the conductivity `σ` are two descriptions of the same physical quantity. With the time-harmonic convention `D ∝ e^{−iωt}` used by this program, for the imaginary part of the **relative** permittivity one has

```
σ = ε₀ · ω · ε_imag        (hence ε_imag ≥ 0 means loss and must be positive)
```

Therefore a dielectric function table given through `dielectric=` only needs the (positive) imaginary part, with no additional conversion to `sigma`; conversely, if the loss of a material is given through `sigma=`, no imaginary part needs to be filled in either. When a material has `use_ADE = 1`, the imaginary part also takes part in deriving the ADE damping tensor `γ` (see [9.5](#95-negative-permittivity-and-the-ade-method)).

---

## 10. Post-Processing Workflow

```
simulation.cfg ──► generate_medium5.py ──► medium_map.bin
      │                                                │
      ├──► visualize_input3.m (geometry/source check)  │
      │                                                ▼
      └───────────────────────────────────────►  fdtd.x (MPI) ──► field_slice_*.bin
                                                       │        monitor_point.dat
                                                       │        full_monitor.dat
                                                       │        source.txt
                                                       ▼
   visualize_medium.m (medium distribution check)  ┌───┴───────────────────┐
                                                   ▼                       ▼
                                      visualize_results2*.py  frequency_analysis (FFT)
                                      (field snapshots/animation)          │
                                                                           ▼
                                                               frequency_slice_*.txt
                                                               (plot near-field maps with Origin, etc.)
```

### 10.1 `generate_medium5.py` — Generating the Medium Distribution

```bash
python3 generate_medium5.py simulation.cfg [medium_map.bin]
```

- 1st argument: the configuration file; 2nd argument: the output file name (defaults to `medium_map.bin`).
- Parses `cubes` / `hexahedra` / `cylinders` from the `[medium]` section, builds an `int32` array and writes it to the binary file.
- Requires NumPy; it prints the number of voxels and the percentage occupied by each material, which makes it easy to check that the geometry is correct.

### 10.2 `visualize_input3.m` / `visualize_medium.m` — Input Checking (MATLAB)

- `visualize_input3.m`: reads `simulation.cfg` (the default file name is written on line 4 of the script) and draws the cubes/hexahedra/cylinders and the source positions in 3D.
- `visualize_medium.m`: reads `medium_map.bin` (the default file name is written on line 5 of the script), reading slice by slice to save memory, and plots the medium distribution.

### 10.3 `visualize_results2.py` / `visualize_results2_switchxy.py` — Field Distribution Snapshots

```bash
python3 visualize_results2.py
```

- Automatically `glob`s the `field_slice_*.bin` files in the current directory, prints `max|field|` for every frame, and draws the first 30 frames as subplots.
- Output files: `field_evolution.png` (grid of the first 30 frames), `last_frame.png` (last frame), `field_evolution.gif` (animation of all frames, 200 ms per frame); an interactive window is also opened.
- The `switchxy` version swaps the x/y axes when plotting, for slices in which x/y are interchanged.
- Requires NumPy + Matplotlib. **Note**: the script reads the files in lexicographic order, so files from different slice positions/axes get mixed together; it is best to run it in a dedicated directory.

### 10.4 `frequency_analysis` — Near-Field Frequency Slices (Core s-SNOM Tool)

```bash
./frequency_analysis simulation.cfg
```

- Automatically scans **the current directory** for all files whose **names contain both `field_slice_` and `.bin`**, parses the time step from the digits after the last `_` in the file name, and sorts them into a time series;
- Reads `dt`, `output_interval` and the frequency of the **first source** from the configuration file (`tsteps`/`nx`/`ny`/`nz` are read but **not used**; the grid dimensions are taken from the binary file header), and determines the sampling rate and the target frequency from them;
- Performs an FFT over time at every spatial point (FFTW3, **no window, no detrending**, result normalized by dividing by `N`), and outputs the target frequency **together with 4 frequencies before and after it**, up to 9 slices in total, with file names
  `frequency_slice_<index>_<f>Hz_<k>cm-1.txt`;
- Every file contains `Re`, `Im`, `|E|` and the phase (radians/degrees) and can be used to plot near-field amplitude and phase maps directly with Origin / Python;
- To keep the file size down, the result files **contain only the central region between 25% and 75% of each dimension**; if complete slices are needed, modify the condition in `save_frequency_result()`.

> **Key points for use**:
> 1. Before running, **delete the field slices from the period in which the source is still oscillating** (for example those with `t < t0 + 6·tau`) and keep only the "free-oscillation / near-field response" stage after the source has decayed to zero, otherwise the spectrum of the source itself contaminates the result. The step number in the file name is `t/dt`, which can be used for this selection.
> 2. This method assumes that the field slices are **equally spaced** in time, so do not mix in leftover files with a different `output_interval`, and do not mix the results of several runs in the same directory.
> 3. The FFT frequency resolution is `1/(N·Δt)`, where `N` is the number of slices; a finer frequency resolution requires more slices (a smaller `output_interval` or more `tsteps`).

### 10.5 `Lorentz_dielectric/*.m` — Dielectric Function Table Generation (MATLAB)

`Lorentz_Ag.m`, `Lorentz_Au.m`, `Lorentz_ARP.m` and `Lorentz_MoO3.m` **generate analytically** (not by fitting) 15-column dielectric function tables using Drude-Lorentz (metals) or anisotropic Lorentz (polar crystals) models, for use with `dielectric=`. Changing the ε∞, ω_LO, ω_TO and γ parameters at the top of the script yields the data for a new material.

| Script | Model |
| --- | --- |
| `Lorentz_Ag.m` | Drude term + 5 Lorentz oscillators |
| `Lorentz_Au.m` | single free-electron Drude term |
| `Lorentz_ARP.m` | two Lorentz oscillators (of the form s²/(ω²−ω²−iγω)) |
| `Lorentz_MoO3.m` | anisotropic LO–TO single-oscillator model, one set of parameters for each of x/y/z |

> Note: when these scripts are run they **overwrite** the `.txt` files of the same names in the same directory (and they require the current directory to be `Lorentz_dielectric/`, because the output uses bare file names). Do not rerun them without a backup.

### 10.6 `v_source.m` — Source Waveform Preview (MATLAB)

A few lines of script that draw the time-domain waveform of the source from the `amplitude`/`wavenumber`/`t0`/`tau` at the top of `v_source.m`, used to confirm before a production run that the spectrum and duration of the source are appropriate.

---

## 11. Parallelization and Performance

### 11.1 Domain Decomposition

- Decomposition is one-dimensional along the **z direction** only: `base = nz/size`, and the first `nz % size` processes each receive one extra layer.
- The local array dimensions of each process are `nx × ny × (local_nz + 2)`; the extra 2 layers are used for the halo along z (including dead cells outside the global boundary).
- Two halo exchanges are performed per step (E is exchanged before the H update, H before the D update), using vector communication with **packing + preallocated buffers** (the boundaries are first exchanged with `MPI_Sendrecv`, then the halo values are broadcast with `MPI_Bcast`).
- `size ≤ nz` is required (otherwise some process gets no grid points and the program reports `Process N has no grid points!`).
- Since only the z direction is parallelized, the grid dimensions along x/y should not be too large while z is too small, otherwise both the parallel efficiency and the memory sharing get worse.

### 11.2 Memory Estimation

Let `nz_local = nz/size + 2` and `V = nx·ny·nz_local`:

| Array | Size |
| --- | --- |
| 9 field components (Ex…Hz) | `9 × V × 8 B` |
| Field modulus `E` (allocated only when `output_component = E`) | `V × 8 B` |
| Medium distribution `medium` | `V × 4 B` |
| Ionic displacements/velocities `r`,`v` | `2 × 3·natom × V_atom × 8 B` (**only** the `V_atom` grid points with `natom>0`) |
| ADE polarization `P_ADE`,`P_ADEo` | `2 × 3 × V_ADE × 8 B` (only grid points with `use_ADE=1`; a fixed 3 components per grid point) |
| CPML auxiliary fields `ψ` (the 8 with x/y normals) | `8 × 2·pml_thickness × (ny or nx) × nz_local × 8 B` (present in every process) |
| CPML auxiliary fields `ψ` (the 4 with z normal) | `4 × V × 8 B` (**only** in processes whose z range overlaps the z-direction PML) |

Here `V = nx·ny·nz_local`, `nz_local = local_nz + 2·halo`.

At start-up and during the run the program prints the actual usage to stderr (`[MEM]` lines, counting only allocations ≥ 0.1 GB), for example:

```
[MEM] Before time loop: per-rank = 1.234567 GB, total (28 ranks) = 34.567876 GB
```

> **Memory hotspot**: `r`/`v` are proportional to `natom`. A material with `natom=16` needs `2×48×8 = 768 B` per grid point, so millions of grid points occupy hundreds of GB. For PhP simulations, give priority to shrinking the region containing atoms, or increase the number of MPI processes (memory is shared over z layers).

### 11.3 Runtime Recommendations

- Compile with `-Ofast -flto -funroll-loops` (recommended by the author); this significantly speeds up the hot triple loops.
- The output frequency (`output_interval`) has a large effect on performance: every output writes a whole slice and performs collective communication, so too small an `output_interval` becomes an I/O bottleneck.
- The overhead of `full_monitor` and `monitor_point` is negligible (only a single point is written), but `monitor_point` writes every step, so the number of lines = `tsteps` and long simulations produce a large text file.
- Progress is printed only by rank 0; every print performs 2 `MPI_Allreduce` calls, which also introduces synchronization overhead when `output_interval` is very small.

---

## 12. Known Limitations and Caveats

1. **Only the CPML absorbing boundary is supported**; periodic boundaries (`bc_x/y/z`) are not implemented.
2. **Parallelization decomposes along z only**; the x/y directions are not decomposed.
3. **Time is advanced explicitly**, and `dt` is limited by the three-dimensional CFL condition (`dt ≤ 0.99·Δ_min/(c√3)`); unconditionally stable schemes such as ADI-FDTD are **not implemented**, so fine grids combined with long simulations are expensive.
4. **`EPS_INF` is a compile-time constant** (default 20.0) and only affects materials with `use_ADE=1`; changing the material requires recompiling.
5. **Both the ADE coefficients and the dielectric function table interpolation target the frequency of the first source only**; multi-frequency/broadband problems require several runs over different frequency bands.
6. **`source` / `material` lines must be written on a single line** and must start in column 0; values must not contain spaces or commas.
7. **Give all source parameters explicitly**: the defaults of `x/y/z/t0/tau` are computed before `nx` and `dt` are read in, so omitting them gives unexpected values; write them out item by item.
8. **After `dt` is clamped by the CFL condition, `tsteps` is not adjusted automatically**, so the total physical duration of the simulation may differ from what you expect.
9. **No stability parameter other than `dt` is parsed**, and there is no check whether `DMM` satisfies the stability condition of the ionic subsystem; if it diverges, reduce `dt` yourself.
10. **`born=<OUTCAR>` overwrites `eps_*` with the static dielectric tensor**, while the Born charge array itself does not take part in the update (see [5.3](#53-material-material) for details).
11. **The `spheres` / `slabs` / `points` geometries are not implemented** (`generate_medium5.py` only supports `cubes`, `hexahedra`, `cylinders`).
12. **The capitalization of the material data file names is inconsistent** (`QeZM_clean.txt` vs `QeZm_clean.txt`); this matters on case-sensitive systems.
13. **Field slice output is not isolated by directory**: all output is written to the current working directory, and the `field_slice_*` files of several runs get mixed together and affect `frequency_analysis` (which scans the whole directory by file name). Using a separate directory for every run is recommended.
14. **Input is not corrected automatically**: an out-of-range material id, a mismatch between `natom` and the length of the data files, a `medium_map.bin` with the wrong dimensions and so on all lead directly to `MPI_Abort`; check the input carefully against this document.
15. `-Ofast` enables floating-point optimizations that do not follow strict IEEE semantics; use `-O3` instead if strict reproducibility is required.
16. Code comments and runtime logs are mainly in **Chinese**.

---

## 13. Troubleshooting

| Symptom | Possible cause and remedy |
| --- | --- |
| `Error opening medium file` | wrong `medium_file` path, or `generate_medium5.py` was not run first |
| `Medium file dimensions (...) do not match simulation (...)` | the `nx/ny/nz` in the configuration file do not match those used when `medium_map.bin` was generated; regenerate it |
| `Process N has no grid points!` | MPI process count > `nz`; reduce `-np` or increase `nz` |
| `Error: QeZM is not set!` / `Error: DMM is not set!` | the material has `natom>0` but `QeZM`/`DMM` were not given (or the file name is misspelled or has the wrong capitalization) |
| `Error: Only N values read (expected M)` | the number of elements in the `QeZV`/`QeZM`/`DMM` file does not match `natom` (should be `9N`, `9N`, `9N²`) |
| `Error: epsr1 can not be inversed` | `use_ADE=1` but `epsr1 = ε_real − EPS_INF` is singular; check `EPS_INF` and the dielectric tensor |
| `EPS_INF is too small for material N` | `EPS_INF` is less than or equal to the real part of ε; change `#define EPS_INF` and recompile |
| `Warning: No exact frequency match in ...` | the frequency of the first source differs from the frequency grid of the dielectric function table by > 10%; check the `wavenumber` conversion or the frequency range of the table |
| `No slice files found!` | `frequency_analysis` was run in the wrong directory, or no field slices have been produced yet |
| Simulation diverges (field values grow exponentially) | ① `dt` exceeds the CFL limit; ② `use_ADE` is set incorrectly or `EPS_INF` does not match; ③ inconsistent units of `DMM`/`QeZM` make the ionic subsystem unstable; ④ the CPML parameters are too strong/too weak |
| Process killed after `ERROR: r field too large at step N` | divergence protection triggered by an ionic displacement exceeding 1 Å (see [9.6](#96-ionic-equations-of-motion-php-coupling)). Locate the problem using the reported `i,j,k`/atom/direction: usually the source amplitude is too large, the units or coordinate conventions of `QeZV`/`QeZM`/`DMM` are inconsistent, or `dt` is too large |
| Fields do not decay and boundary reflections are obvious | `pml_thickness` is too small (≥ 10–20 recommended), `pml_reflection` is too large, or the source/structure is too close to the PML |
| `frequency_analysis` results contain strange frequency components | slices from the period in which the source is oscillating, or files from several runs, were mixed into the field slice directory; clean it up and rerun |
| Chaotic phase in the near-field map | the time window in which the source has not yet decayed was not removed; use only the slices after `t > t0 + 6·tau` |
| OpenMPI 4 prints an error message when the program exits | this is a known issue of OpenMPI 4 itself and does not affect the results; OpenMPI 5 can be used instead |

---

## 14. References and Acknowledgments

The CPML implementation and the choice of parameters in the program refer to:

- J. A. Roden and S. D. Gedney, "Convolution PML (CPML): An efficient FDTD implementation of the CFS-PML for arbitrary media," *Microwave and Optical Technology Letters* **27**(5), 334–339 (2000).
- S. D. Gedney, *Introduction to the Finite-Difference Time-Domain (FDTD) Method for Electromagnetics*, Morgan & Claypool (2011).
- K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Trans. Antennas Propag.* **14**(3), 302–307 (1966).
- A. Taflove and S. C. Hagness, *Computational Electrodynamics: The Finite-Difference Time-Domain Method*, 3rd ed., Artech House (2005).

The material data come from first-principles calculations: the Born effective charge tensors and the dielectric tensors are taken from static VASP calculations, and the dynamical matrices from Γ-point phonon calculations (finite-displacement method). The coordinate conventions of the data files are described in the `README.txt` inside each directory. We thank the open-source/commercial first-principles tools such as VASP and phonopy, as well as the fundamental libraries such as FFTW, NumPy and Matplotlib.

---

## 15. License

This project is released under the **MIT License**; see [LICENSE](LICENSE) for details.

---

## Contributing

Issues and Pull Requests are welcome. Before submitting code, please try to ensure that:

1. no new compilation warnings are introduced;
2. the format of any new output file is documented in the README as well;
3. changes involving physical formulas are accompanied by a derivation or a reference.
