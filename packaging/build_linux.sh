#!/usr/bin/env bash
# Build the Ubuntu .deb package: PyInstaller bundle -> self-check -> dpkg-deb.
#
#     bash packaging/build_linux.sh 1.2.0
#
# Needs uv and dpkg-deb on PATH. Output: dist/installer/metlepet_<Version>_amd64.deb
# The package installs the bundle to /opt/MetLePET, a `metlepet` command and a
# menu entry. Install with: sudo apt install ./metlepet_<Version>_amd64.deb
set -euo pipefail

VERSION="${1:-0.0.0-dev}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Installing locked dependencies"
uv sync --locked --no-dev --group build

echo "== Building app bundle (PyInstaller)"
uv run --no-sync pyinstaller packaging/petct_app.spec --noconfirm --clean

echo "== Self-check of the bundle"
# Offscreen Qt: build machines have no display. The frozen app logs to
# $XDG_DATA_HOME/MetLePET/logs/app.log; throw-away XDG dirs keep the builder's
# own app data and settings untouched.
CHECK_HOME="$(mktemp -d)"
trap 'rm -rf "$CHECK_HOME"' EXIT
status=0
XDG_DATA_HOME="$CHECK_HOME/data" XDG_CONFIG_HOME="$CHECK_HOME/config" QT_QPA_PLATFORM=offscreen \
    dist/MetLePET/MetLePET --selfcheck || status=$?
cat "$CHECK_HOME"/data/MetLePET/logs/app.log 2>/dev/null || true
if [ "$status" -ne 0 ]; then
    echo "Self-check failed (exit code $status)" >&2
    exit 1
fi

echo "== Building .deb"
PKG="dist/deb/metlepet"
rm -rf dist/deb
mkdir -p "$PKG/DEBIAN" "$PKG/opt" "$PKG/usr/bin" "$PKG/usr/share/applications"
cp -a dist/MetLePET "$PKG/opt/MetLePET"

cat > "$PKG/usr/bin/metlepet" <<'EOF'
#!/bin/sh
exec /opt/MetLePET/MetLePET "$@"
EOF

cat > "$PKG/usr/share/applications/metlepet.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=MetLePET
Comment=PET/CT lesion segmentation and quantification
Exec=/opt/MetLePET/MetLePET
Icon=applications-science
Terminal=false
Categories=Science;MedicalSoftware;
EOF

chmod -R u+rwX,go+rX,go-w "$PKG"
chmod 755 "$PKG/usr/bin/metlepet"
INSTALLED_KB="$(du -sk "$PKG/opt" "$PKG/usr" | awk '{s += $1} END {print s}')"

echo "== Working out Depends"
# The bundle ships no system libraries (petct_app.spec leaves them out, so the
# app uses the desktop's own GTK/GLib/X11/fonts like a source run). Depends is
# the package of every library the bundle links to but does not contain, looked
# up on this build machine. CI builds on Ubuntu 22.04: its package names also
# resolve on newer releases, where the renamed "t64" packages Provide them.
# Libraries needed only by optional parts (numba's TBB / OpenMP thread pools —
# numba falls back to its own — and Qt's TIFF image plugin) are not required.
OPTIONAL_LIBS="libtbb.so.12 libgomp.so.1 libgomp.so.1.0.0 libtiff.so.5 libtiff.so.6"
# Opened at runtime (dlopen) rather than linked, so readelf can't see them:
# OpenGL for napari's canvases (PyOpenGL), D-Bus for Qt's desktop integration.
RUNTIME_PKGS="libgl1 libegl1 libdbus-1-3"

LDCONFIG="$(command -v ldconfig || echo /sbin/ldconfig)"   # /sbin is not on every PATH
APP_DIR="$PKG/opt/MetLePET"
mapfile -t EXTERNAL < <(LC_ALL=C comm -23 \
    <(find "$APP_DIR" -type f \( -name '*.so*' -o -name MetLePET \) -exec readelf -d {} + 2>/dev/null \
        | sed -n 's/.*Shared library: \[\(.*\)\]$/\1/p' | LC_ALL=C sort -u) \
    <(find "$APP_DIR" \( -type f -o -type l \) -name '*.so*' -printf '%f\n' | LC_ALL=C sort -u))

PKGS=() UNRESOLVED=()
for lib in "${EXTERNAL[@]}"; do
    case " $OPTIONAL_LIBS " in *" $lib "*) echo "  optional: $lib"; continue ;; esac
    path="$("$LDCONFIG" -p | awk -v l="$lib" '$1 == l && /x86-64/ {print $NF; exit}')"
    pkg=""
    if [ -n "$path" ]; then
        real="$(readlink -f "$path")"
        # dpkg records /lib or /usr/lib depending on the package (merged /usr).
        for p in "$real" "$path" "${real#/usr}"; do
            pkg="$(dpkg -S "$p" 2>/dev/null | grep -v '^diversion' | head -1 | cut -d: -f1 || true)"
            [ -n "$pkg" ] && break
        done
    fi
    if [ -n "$pkg" ]; then
        echo "  $lib -> $pkg"
        PKGS+=("$pkg")
    else
        UNRESOLVED+=("$lib")
    fi
done
if [ "${#UNRESOLVED[@]}" -gt 0 ]; then
    echo "No package found on this machine for: ${UNRESOLVED[*]}" >&2
    echo "Install the package that provides them (CI: the apt step of build-linux)," >&2
    echo "or add them to OPTIONAL_LIBS if the app works without them." >&2
    exit 1
fi
# libc6 gets the floor of the oldest supported release (Ubuntu 22.04).
DEPENDS="$(printf '%s\n' "${PKGS[@]}" $RUNTIME_PKGS | grep -vx libc6 | LC_ALL=C sort -u | paste -sd, - | sed 's/,/, /g')"
DEPENDS="libc6 (>= 2.35), $DEPENDS"
echo "  Depends: $DEPENDS"

cat > "$PKG/DEBIAN/control" <<EOF
Package: metlepet
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Maintainer: HoangTung-Vu <minhhoangtungvu04@gmail.com>
Installed-Size: $INSTALLED_KB
Depends: $DEPENDS
Description: MetLePET - PET/CT lesion segmentation and quantification
 Desktop viewer for PET/CT studies (NIfTI or DICOM) with AI segmentation
 through a remote engine, manual refinement and per-lesion SUV, MTV and
 TLG reporting.
EOF

mkdir -p dist/installer
DEB="dist/installer/metlepet_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$PKG" "$DEB"
ls -lh "$DEB"
