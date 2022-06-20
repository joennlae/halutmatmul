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
