# GPU CI with vast.ai

* [Python CLI](https://github.com/vast-ai/vast-python)

## Debug

```bash
python vast_ai_helper.py -d
```

### New SSH keys

If SSH keys go lost:
1) Create a new pair. And store it in `.github/vast.ai/.ssh/id_rsa` for debugging.
2) Update public key on `vast.ai`
3) Update `VAST_AI_SSH_KEY` on GitHub


### Search for max RAM

```bash
./vast.py search offers 'reliability > 0.98  num_gpus==1 rentable==True inet_down > 100 disk_space > 52 dph_total < 0.35 inet_down_cost < 0.021 inet_up_cost < 0.021 cuda_vers >= 11.2 gpu_ram>16' -o 'gpu_ram-' --storage=32
```