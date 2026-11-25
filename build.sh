#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
engine=${SLICER_BUNDLE:-/Applications/Slicer.app}
bundle=${PE_ATLAS_BUNDLE:-"$PWD/build/PE Atlas.app"}
engine=$(cd "$engine" && pwd -P)
test -x "$engine/Contents/MacOS/Slicer"
test -x "$engine/Contents/bin/PythonSlicer"
contents="$bundle/Contents"
assets="$contents/Resources/PEAtlas"
mkdir -p "$contents/MacOS" "$assets/tools" "$assets/models" "$assets/tests"
ln -sfn "$engine" "$contents/Resources/SlicerEngine.app"
cp workstation.py "$assets/"
cp tools/artery_landmarks.py tools/artery_picking.py tools/arterial_evidence.py tools/appearance.py tools/branch_unfold.py tools/forest.py tools/grid.py tools/lung_cutaway.py tools/prepare_case.py tools/spotlight_comparison.py "$assets/tools/"
clang -O3 -dynamiclib -arch x86_64 -arch arm64 tools/forest.c -o "$assets/tools/libforest.dylib"
cp README.md LICENSE "$assets/"
cp tests/fumpe-results.json tests/anatomy-results.json "$assets/tests/"
cp models/appearance.npz "$assets/models/"
clang -O2 launcher.c -o "$contents/MacOS/PEAtlas"
plist="$contents/Info.plist"
/usr/bin/plutil -create xml1 "$plist"
for pair in 'CFBundleExecutable PEAtlas' 'CFBundleIdentifier local.peatlas.public' 'CFBundleName PE Atlas' "CFBundleDisplayName $(basename "$bundle" .app)" 'CFBundlePackageType APPL' 'CFBundleVersion 1' 'CFBundleShortVersionString 0.3.0'; do
  /usr/bin/plutil -insert "${pair%% *}" -string "${pair#* }" "$plist"
done
printf '%s\n' "$bundle"
