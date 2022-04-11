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