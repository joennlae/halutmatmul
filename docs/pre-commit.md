# Pre-commit Linting & Commitlint

Install `pre-commit` [Link](https://pre-commit.com/)

```bash
conda install -c conda-forge pre-commit
```

Activate with:
```bash
pre-commit
pre-commit install
```

## Install gitlint

[Link](https://jorisroovers.com/gitlint/#using-gitlint-as-a-commit-msg-hook)

```bash
pre-commit install --hook-type commit-msg
```

## Testing

```
pre-commit run gitlint --hook-stage commit-msg --commit-msg-filename .git/COMMIT_EDITMSG
pre-commit run --all-files 
```