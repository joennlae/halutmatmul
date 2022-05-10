# Conda

## Export Conda env

Nice and neatly
```bash
conda env export --from-history > environment.yml
conda env export --no-builds > first_part.yml
```
and then merge it together.


### Reproducibility

```bash
$ conda env export > environment_lock.yml 
```

### ISS config

10GB is just not enough :-)

add a file `~/.condarc` and add

```yaml
pkgs_dirs:
  - /scratch/janniss/conda
```

### Create env in scratch

```
conda env create -f environment_gpu.yml --prefix /scratch2/janniss/conda/halutmatmul_gpu

conda activate /scratch2/janniss/conda/halutmatmul_gpu
```

On CPU servers

```
conda env create -f environment_cpu.yml --prefix /scratch/janniss/conda/halutmatmul_cpu
```

Don't forget to change path in `~/.condarc`