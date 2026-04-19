#!/usr/bin/env bash
# Lanceur du Jeu de la Vie.
# Cherche un python3 utilisable et, si présent, ajoute ~/.local/lib en tête
# de LD_LIBRARY_PATH — utile sur Fedora/RHEL où les symlinks libGL.so /
# libEGL.so ne sont pas livrés sans les paquets -devel (cf. README).
set -eu
cd "$(dirname "$0")"

# Permet de forcer un interpréteur via la variable d'environnement PYTHON.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
else
    echo "run.sh: aucun interpréteur python3 trouvé dans PATH." >&2
    echo "Installe Python 3.10+ ou définis la variable PYTHON=/chemin/vers/python." >&2
    exit 1
fi

# Sur Fedora on range souvent des symlinks libGL.so/libEGL.so dans ~/.local/lib.
# On ne l'ajoute que si le répertoire existe, sinon on laisse LD_LIBRARY_PATH tel quel.
if [ -d "$HOME/.local/lib" ]; then
    export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY" life.py "$@"
