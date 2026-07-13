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
