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
python3 cloud_compression.py --input=wdas_cloud/wdas_cloud_sixteenth.vdb --epsilon=0.01
```

4. evaluate errors and numbers of coefficients w.r.t. the original cloud

```shell

```

5. (optional) render both in blender, from the same perspective

```shell
for v in *.vdb ; do  ./blender-5.1.2-linux-x64/blender --background     --python ./wdas_cloud_setup.py --     --vdb $(pwd)/$v     --out /tmp/cloud.blend     --render ${v}_135.png     --rotation 135 --samples 64  ; done
```
