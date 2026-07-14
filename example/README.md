<!--
SPDX-FileCopyrightText: 2026 Theresa Pollinger

SPDX-License-Identifier: CC-BY-4.0
-->

# Example: Compressing the WDAS cloud step-by-step

1. Download and extract dataset from the
[WDAS download page](https://www.disneyanimation.com/resources/clouds/):

```shell
wget https://assets.disneyanimation.com/wdas_cloud.zip
unzip wdas_cloud.zip
```

2. set up dependencies, for example with micromamba

```shell
micromamba create -n cloud-openvdb-env -f cloud-openvdb-env.yaml
micromamba activate cloud-openvdb-env
pip install -e ..
```

3. execute compression, for example on the smallest cloud:

```shell
python3 cloud_compression.py --epsilon=0.01 wdas_cloud/wdas_cloud_sixteenth.vdb
```

4. evaluate errors and numbers of coefficients w.r.t. the original cloud

```shell
python3 openvdb_error_evaluation.py wdas_cloud/wdas_cloud_sixteenth.vdb spght_*_wdas_cloud_sixteenth.vdb
```

5. (optional) render both in blender, from the same perspective

```shell
for v in *.vdb ; do  ./blender-5.1.2-linux-x64/blender --background     --python ./wdas_cloud_setup.py --     --vdb $(pwd)/$v     --out /tmp/cloud.blend     --render ${v}_135.png     --rotation 135 --samples 64  ; done
```

# Example: Compressing Hasegawa–Wakatani plasma turbulence step-by-step

The [HW2D reference implementation](https://github.com/the-rccg/hw2d)
(Greif, [JOSS 8:5959, 2023](https://doi.org/10.21105/joss.05959)) 
simulates drift-wave turbulence at the tokamak edge.
Its fields (density n, electrostatic potential φ, vorticity Ω) are nodal values on a doubly
periodic 2^l x 2^l grid — a perfect fit for the periodic vertex-centered bases.
The dataset is generated locally.
The data illustrates the limits of sparse grids, as it cannot be compressed well with spght at all!

1. set up dependencies, for example with micromamba

```shell
micromamba create -n hw2d-env -f hw2d-env.yaml
micromamba activate hw2d-env
pip install -e ..
```

2. generate the dataset, seeded for reproducibility 
(the turbulence is saturated well before the run ends at t=300)

```shell
python3 -m hw2d --grid_pts=128 --end_time=300 --seed=42 --movie=0 --snaps=100 --output_path=hw2d_128.h5
```

3. execute compression, for example on the last density snapshot:

```shell
python3 hw2d_compression.py --epsilon=0.01 hw2d_128.h5
```

4. evaluate errors and numbers of coefficients w.r.t. the original field

```shell
python3 hdf5_error_evaluation.py hw2d_128.h5 spght_*_hw2d_128.h5
```
