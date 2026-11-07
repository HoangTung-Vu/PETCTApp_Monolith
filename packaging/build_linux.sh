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

# System libraries the bundle needs but does not ship: the GPU driver stack
# (OpenGL/EGL/libdrm, for napari's canvases), libxcb and the Wayland client
# libraries. PyQt6 ships X11/xcb-util/xkbcommon/fontconfig/GLib/D-Bus itself.
# Found with readelf on the bundle; check again after a Qt/PyQt6 upgrade.
cat > "$PKG/DEBIAN/control" <<EOF
Package: metlepet
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Maintainer: HoangTung-Vu <minhhoangtungvu04@gmail.com>
Installed-Size: $INSTALLED_KB
Depends: libc6 (>= 2.35), libgl1, libegl1, libopengl0, libdrm2, libxcb1, libwayland-client0, libwayland-cursor0, libwayland-egl1
Description: MetLePET - PET/CT lesion segmentation and quantification
 Desktop viewer for PET/CT studies (NIfTI or DICOM) with AI segmentation
 through a remote engine, manual refinement and per-lesion SUV, MTV and
 TLG reporting.
EOF

mkdir -p dist/installer
DEB="dist/installer/metlepet_${VERSION}_amd64.deb"
dpkg-deb --build --root-owner-group "$PKG" "$DEB"
ls -lh "$DEB"
