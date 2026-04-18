#!/usr/bin/env bash
# Lanceur du Jeu de la Vie.
# Fedora ne livre pas les symlinks libGL.so / libEGL.so sans les paquets -devel,
# alors on pointe ld vers ~/.local/lib qui contient nos liens (cf. README).
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$HOME/.local/lib:${LD_LIBRARY_PATH:-}"
exec "$HOME/.pyenv/versions/3.12.8/bin/python3" life.py "$@"
