# PE Atlas

PE Atlas is a macOS research app for exploring CT pulmonary angiograms. Click
the 3D pulmonary artery model to jump to the corresponding axial image and
coronal reformat. The app runs locally using 3D Slicer. Use it for research and
education only.

## Run the FUMPE example

1. Install [3D Slicer 5.10.0](https://download.slicer.org/?version=5.10.0) in
   `/Applications` and install Apple's Command Line Tools (`xcode-select --install`)
   if `clang` is unavailable. An Intel Slicer build on Apple Silicon needs Rosetta.
2. From the PE Atlas source folder, build the app.

   ```sh
   ./build.sh
   ```

3. Download [`fumpe-pat003.zip`](https://github.com/1JRobertson/pe-atlas-public/releases/download/research-v1/fumpe-pat003.zip)
   from the release and leave it zipped. Install the example using Python 3.9 or newer.

   ```sh
   python3 tools/install_demo.py "$HOME/Downloads/fumpe-pat003.zip"
   open "build/PE Atlas.app"
   ```

The example includes a CT volume, pulmonary artery model, automated proposals,
and the published expert outline. PE Atlas opens **FUMPE PAT003** from
`~/Library/Application Support/PE Atlas Public/Cases`. Set `SLICER_BUNDLE` when
building to use a different Slicer app path. Set `PE_ATLAS_HOME` before running
the installer and launching the executable directly to use a different study
folder. Keep the Slicer installation at its build-time path. The source build
was tested on Apple Silicon with Slicer 5.10.0 (Intel).

## Explore the case

- Click the 3D artery to jump to the source CT or scroll the axial image with the mouse or **A** and **D**.
- Open **Lung Cutaway** to move a plane through the CT. Pin a point to inspect it in **Spotlight**.
- Enable **⋯ → Display → Expert outlines** to show the FUMPE reference in violet.
- Open **⋯ → Edit outlines** to brush corrections. Edits save locally in the study folder.

Named vessel guides and paired scans require additional case data. The PAT003
example contains one scan.

## Detector limits

Candidate overlays come from an experimental classifier trained on eight FUMPE
cases. On three further held-out scans, its higher-suspicion layer touched five
of eight expert regions and covered about 2.5% of the expert-labeled voxels.
Six of eleven flags had no overlap with an expert lesion. See the
[evaluation results](tests/anatomy-results.json) for methods and per-case
counts. These small research evaluations cannot establish clinical performance.

## Credits and licensing

PE Atlas source code and documentation are available under the
[PE Atlas Research and Education License](LICENSE). Research and education use
is free. Resale, paid access, and inclusion in a paid product or service
require separate written permission from the copyright holder.
The FUMPE-derived example and model use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The example reformats the public Patient03 CT and expert mask from the FUMPE
dataset by Mojtaba Masoudi, Hamid-Reza Pourreza, Mehdi Saadatmand-Tarzjan,
Noushin Eftekhari, Fateme Shafiee Zargar, and Masoud Pezeshki Rad. See the
[dataset](https://doi.org/10.6084/m9.figshare.c.4107803) and
[data descriptor](https://doi.org/10.1038/sdata.2018.180).

The app uses [3D Slicer](https://www.slicer.org/) from your local installation.
The example arterial mask was generated with
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator). Upstream
projects retain their own licenses.
