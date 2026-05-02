# SCC Setup Instructions

I used the SCC for the fine-tuning and evaluation of the model. Below are the instructions for setting up your environment on the SCC, including troubleshooting tips for common issues.

1. Log into the SCC using your credentials.
2. Select Interactive Apps > Jupyter Notebook
3. Enter the following settings:
    - modules: miniconda academic-ml/fall-2025
    - pre-launch command: conda activate fall-2025-pyt
    - interface: lab
    - working directory: your working dir
    -  number of hours: 4 (or more, if you have a long-running job)
    - number of cores: 6
    - number of gpus: 1
    - GPU compute capability: 8.0 A100
    - Project: your project
4. Click "Launch" and wait for the Jupyter Notebook to start.


## Troubleshooting:

Runnning out of space, this is how I got around it

```bash
# 1. Create your cache directory on scratch
mkdir -p /scratch/mlong6/huggingface_cache

# 2. Copy existing cache there (safe copy first)
cp -r /usr3/bustaff/mlong6/.cache/huggingface/* /scratch/mlong6/huggingface_cache/

# 3. Verify the copy looks correct
ls /scratch/mlong6/huggingface_cache/
# Should see: datasets  hub  xet  CACHEDIR.TAG

# 4. Remove original (only after confirming step 3 looks right)
rm -rf /usr3/bustaff/mlong6/.cache/huggingface/

# 5. Create symlink so nothing breaks
ln -s /scratch/mlong6/huggingface_cache /usr3/bustaff/mlong6/.cache/huggingface

# 6. Verify symlink
ls -la /usr3/bustaff/mlong6/.cache/
# Should show: huggingface -> /scratch/mlong6/huggingface_cache

echo 'export HF_HOME=/scratch/mlong6/huggingface_cache' >> ~/.bashrc
source ~/.bashrc
```


to check the quota, use
```bash
quota `-s
```