# FUMPE PAT003 example

CT images and expert labels are from the Ferdowsi University of Mashhad
Pulmonary Embolism dataset by Mojtaba Masoudi, Hamid-Reza Pourreza, Mahdi
Saadatmand-Tarzjan, Noushin Eftekhari, Fateme Shafiee Zargar, and Masoud
Pezeshki Rad. See the [data descriptor](https://doi.org/10.1038/sdata.2018.180),
[Patient03 archive](https://doi.org/10.6084/m9.figshare.6265364), and
[Ground Truth archive](https://doi.org/10.6084/m9.figshare.6265721).
The source and this adaptation use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

This adaptation converts the native CT and reordered expert labels to NIfTI,
replaces source identifiers with generated identifiers, and adds a vessel
segmentation made with [TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
a 3D surface, and experimental PE Atlas candidate overlays. Use for research
and education only.
