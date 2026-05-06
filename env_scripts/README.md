# Environment Scripts

Scripts to set up and activate isolated environments for this project.

## What they do

- `setup_gg_env.sh`: creates/installs dependencies for **GG** environment.
- `activate_gg.sh`: activates **GG** in the current terminal session.
- `setup_gsw_env.sh`: creates/installs dependencies for **GSW** environment.
- `activate_gsw.sh`: activates **GSW** in the current terminal session.

## Quick start (Linux)

From project root:

```bash
cd env_scripts
chmod +x setup_gg_env.sh setup_gsw_env.sh activate_gg.sh activate_gsw.sh
```

## Example: Set up + use GG

```bash
cd env_scripts
./setup_gg_env.sh
source ./activate_gg.sh
```

## Example: Set up + use GSW

```bash
cd env_scripts
./setup_gsw_env.sh
source ./activate_gsw.sh
```

## Notes

- Always use `source` for `activate_*.sh` scripts.
- Setup is typically run once; activation is run each new terminal session.
- To debug setup:
  ```bash
  bash -x ./setup_gg_env.sh
  ```