#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Start the installed Slicer executable with the PE Atlas workstation.
int main(void) {
    char path[PATH_MAX], resolved[PATH_MAX], engine_path[PATH_MAX], engine[PATH_MAX], script[PATH_MAX];
    uint32_t length = sizeof(path);
    if (_NSGetExecutablePath(path, &length) || !realpath(path, resolved)) return 1;
    char *slash = strrchr(resolved, '/');
    if (!slash) return 1;
    *slash = 0;
    if (snprintf(engine_path, sizeof(engine_path), "%s/../Resources/SlicerEngine.app/Contents/MacOS/Slicer", resolved) >= sizeof(engine_path) ||
        snprintf(script, sizeof(script), "%s/../Resources/PEAtlas/workstation.py", resolved) >= sizeof(script)) return 1;
    if (!realpath(engine_path, engine)) {
        perror("PE Atlas needs the Slicer installation used when building");
        return 1;
    }
    execl(engine, engine, "--no-splash", "--no-main-window", "--disable-settings",
          "--ignore-slicerrc", "--disable-cli-modules", "--disable-scripted-loadable-modules",
          "--modules-to-ignore", "Annotations,CropVolume,DynamicModeler,GeneralizedReformat,MultiVolumeExplorer,Plots,Reformat,SceneViews,Sequences,Tables,Texts,ViewControllers,VolumeRendering,Welcome",
          "--python-script", script, (char*)NULL);
    perror("PE Atlas launch"); return 1;
}
