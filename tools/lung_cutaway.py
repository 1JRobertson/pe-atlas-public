"""Interactive lung cutaway sampled from a loaded CT in physical coordinates."""

import ctk
import numpy as np
import qt
import slicer
import vtk
from scipy import ndimage
from vtk.util.numpy_support import numpy_to_vtk


PLANES = {
    "Coronal": ((-1, 0, 0), (0, 0, 1)),
    "Axial": ((-1, 0, 0), (0, 1, 0)),
    "Sagittal": ((0, 1, 0), (0, 0, 1)),
}
WINDOWS = {"Lung": (1250, -560), "Soft tissue": (400, 40)}
SPOTLIGHT_RADIUS = 40  # mm from the original pin to the edge of each pane


def image_data(array):
    """Create a VTK image from KJI scalars or a single slice of RGB(A) pixels."""
    image = vtk.vtkImageData()
    if array.ndim == 4:
        depth, height, width, components = array.shape
        values = array.reshape(-1, components)
    else:
        depth, height, width = array.shape
        values = array.ravel()
    image.SetDimensions(width, height, depth)
    image.GetPointData().SetScalars(numpy_to_vtk(values, deep=True))
    return image


def windowed_rgb(intensities, window_name):
    window, level = WINDOWS[window_name]
    gray = np.uint8(np.clip((intensities - level) / window + 0.5, 0, 1) * 255)
    return np.repeat(gray[..., None], 3, axis=-1)


def sample_plane(volume, ras_to_ijk, center, u, v, half_size, shape, fill):
    """Resample a KJI volume onto a RAS plane with linear interpolation."""
    half_width, half_height = half_size
    height, width = shape
    origin = center - half_width * u - half_height * v
    steps = np.column_stack((
        np.zeros(3),
        v * (2 * half_height / (height - 1)),
        u * (2 * half_width / (width - 1)),
    ))
    ras_to_kji = ras_to_ijk[:3, :3][::-1]
    return ndimage.affine_transform(
        volume, ras_to_kji @ steps, offset=ras_to_kji @ origin + ras_to_ijk[:3, 3][::-1],
        output_shape=(1, height, width), output=np.float32, order=1, cval=fill,
    )[0]


def draw_orientation(painter, u, image_width, compact):
    font = qt.QFont()
    font.setPixelSize(8 if compact else 13)
    painter.setFont(font)
    painter.setPen(qt.QColor("#eef6fa"))
    axis = int(np.argmax(np.abs(u)))
    pair = (("L", "R"), ("P", "A"), ("I", "S"))[axis]
    if u[axis] < 0:
        pair = pair[::-1]
    inset, width, height = (5, 12, 13) if compact else (7, 19, 21)
    for label, x in zip(pair, (inset, image_width - inset - width)):
        rect = qt.QRect(x, inset, width, height)
        painter.fillRect(rect, qt.QColor(11, 21, 30, 220))
        painter.drawText(rect, qt.Qt.AlignCenter, label)


def draw_crosshair(painter, x, y):
    painter.setPen(qt.QPen(qt.QColor("#ffd76b")))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        painter.drawLine(qt.QLineF(x + 2 * dx, y + 2 * dy, x + 6 * dx, y + 6 * dy))


def lung_envelope(ct, ijk_to_ras):
    """Estimate a lightweight lung surface and its padded cut-face mask."""
    spacing = np.linalg.norm(ijk_to_ras[:3, :3], axis=0)
    strides = np.maximum(1, np.rint(1.8 / spacing).astype(int))
    i, j, k = strides
    coarse = ct[::k, ::j, ::i]
    # Rectangular closing can use separable max/min filters with the same zero border.
    patient = ndimage.maximum_filter(coarse > -350, size=(1, 5, 5), mode="constant")
    patient = ndimage.minimum_filter(patient, size=(1, 5, 5), mode="constant")
    for plane in patient:
        ndimage.binary_fill_holes(plane, output=plane)
    air = ndimage.binary_opening(patient & (coarse < -420))
    labels, count = ndimage.label(air)
    if not count:
        raise ValueError("Could not find a lung outline in this CT.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    components = np.argsort(sizes)[-2:]
    components = components[sizes[components] >= max(500, sizes[components[-1]] * 0.15)]
    if not len(components):
        raise ValueError("Could not find a lung outline in this CT.")
    mask = np.isin(labels, components)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2) | mask)
    mask = np.pad(mask, 1)
    coarse_to_ras = ijk_to_ras.copy()
    coarse_to_ras[:3, :3] *= strides
    coarse_to_ras[:3, 3] -= coarse_to_ras[:3, :3].sum(axis=1)

    density = ndimage.gaussian_filter(mask.astype(np.float32), 0.85)
    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(image_data(density))
    surface.SetValue(0, 0.48)
    transform = vtk.vtkTransform()
    transform.SetMatrix(coarse_to_ras.ravel())
    world = vtk.vtkTransformPolyDataFilter()
    world.SetTransform(transform)
    world.SetInputConnection(surface.GetOutputPort())
    world.Update()
    mesh = vtk.vtkPolyData()
    mesh.ShallowCopy(world.GetOutput())
    if not mesh.GetNumberOfCells():
        raise ValueError("Could not find a lung outline in this CT.")
    # A little surrounding CT stays on the cut face at the edge of the outline.
    face_mask = ndimage.binary_dilation(mask, iterations=4).astype(np.float32)
    return mesh, face_mask, np.linalg.inv(coarse_to_ras)


class LungCutaway(qt.QObject):
    def __init__(self, back, compare):
        super().__init__()
        self.volume = None
        self.pin = None
        self.spotlight_point = None
        self.press = None
        self.scene_press = None
        self.pixmap = qt.QPixmap()
        self.widget = qt.QWidget()
        self.widget.setObjectName("shell")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header, row = self.bar()
        layout.addWidget(header)
        exit_button = qt.QPushButton("‹ Back")
        exit_button.connect("clicked()", back)
        row.addWidget(exit_button)
        title = qt.QLabel("Lung Cutaway")
        title.setObjectName("title")
        row.addWidget(title)
        row.addStretch()
        self.spotlight_button = qt.QPushButton("Spotlight")
        self.spotlight_button.setCheckable(True)
        self.spotlight_button.setEnabled(False)
        self.spotlight_button.setToolTip("Pin a point to inspect it in three linked CT planes")
        self.spotlight_button.connect("toggled(bool)", self.toggle_spotlight)
        row.addWidget(self.spotlight_button)
        self.compare_button = qt.QPushButton("Compare scans")
        self.compare_button.connect("clicked()", compare)
        self.compare_button.setToolTip("Open the saved scans side by side")
        self.compare_button.hide()
        row.addWidget(self.compare_button)
        self.orientation = qt.QComboBox()
        self.orientation.setAccessibleName("Cut plane")
        self.orientation.addItems(list(PLANES))
        self.orientation.connect("currentIndexChanged(int)", self.change_orientation)
        row.addWidget(self.orientation)
        self.window = qt.QComboBox()
        self.window.setAccessibleName("CT window")
        self.window.addItems(list(WINDOWS))
        self.window.connect("currentIndexChanged(int)", self.update_cut)
        row.addWidget(self.window)

        split = qt.QSplitter(qt.Qt.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)
        self.scene = ctk.ctkVTKRenderView()
        self.scene.setMinimumWidth(320)
        self.scene.setAccessibleName("3D lung cutaway")
        self.scene.setToolTip("Drag to orbit. Scroll to move the cut. Click the cut face to pin.")
        split.addWidget(self.scene)
        slices = qt.QWidget()
        slices.setMinimumWidth(280)
        slices_layout = qt.QVBoxLayout(slices)
        slices_layout.setContentsMargins(0, 0, 0, 0)
        slices_layout.setSpacing(1)
        self.flat = self.slice_label()
        self.flat.setAccessibleName("CT cut. Click to pin a position.")
        slices_layout.addWidget(self.flat, 2)
        self.detail_strip = qt.QWidget()
        self.detail_strip.setObjectName("navigation")
        details = qt.QHBoxLayout(self.detail_strip)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(1)
        self.detail_titles = []
        self.detail_labels = []
        for _ in range(2):
            pane = qt.QWidget()
            pane_layout = qt.QVBoxLayout(pane)
            pane_layout.setContentsMargins(0, 0, 0, 0)
            pane_layout.setSpacing(0)
            label = self.slice_label()
            title = qt.QLabel()
            title.setAlignment(qt.Qt.AlignCenter)
            title.setFixedHeight(25)
            pane_layout.addWidget(title)
            pane_layout.addWidget(label, 1)
            details.addWidget(pane, 1)
            self.detail_titles.append(title)
            self.detail_labels.append(label)
        slices_layout.addWidget(self.detail_strip, 1)
        self.detail_strip.hide()
        self.detail_cuts = []
        split.addWidget(slices)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 520])

        footer, row = self.bar()
        footer.setFixedHeight(45)
        layout.addWidget(footer)
        self.hint = qt.QLabel()
        row.addWidget(self.hint)
        self.depth = qt.QSlider(qt.Qt.Horizontal)
        self.depth.setAccessibleName("Cut depth")
        self.depth.connect("valueChanged(int)", self.update_cut)
        row.addWidget(self.depth, 1)
        self.clear_pin_button = qt.QPushButton("Clear pin")
        self.clear_pin_button.connect("clicked()", self.clear_pin)
        row.addStretch(0)
        row.addWidget(self.clear_pin_button)

        self.setup_scene()
        for child in slicer.util.findChildren(self.scene):
            if child.isWidgetType():
                child.installEventFilter(self)
        interactor = self.scene.renderWindow().GetInteractor()
        interactor.AddObserver("LeftButtonPressEvent", self.press_scene, 1.1)
        interactor.AddObserver("LeftButtonReleaseEvent", self.release_scene, 1.1)

    def slice_label(self):
        label = qt.QLabel()
        label.setStyleSheet("background: #080f16")
        label.setAlignment(qt.Qt.AlignCenter)
        label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)
        label.setCursor(qt.Qt.CrossCursor)
        label.installEventFilter(self)
        return label

    @staticmethod
    def bar():
        widget = qt.QWidget()
        widget.setObjectName("navigation")
        row = qt.QHBoxLayout(widget)
        row.setContentsMargins(14, 7, 14, 7)
        row.setSpacing(8)
        return widget, row

    def setup_scene(self):
        render_window = self.scene.renderWindow()
        self.renderer = render_window.GetRenderers().GetFirstRenderer()
        self.renderer.SetBackground(0.034, 0.062, 0.086)
        self.renderer.SetBackground2(0.086, 0.145, 0.179)
        self.renderer.GradientBackgroundOn()
        render_window.SetAlphaBitPlanes(1)
        render_window.SetMultiSamples(0)
        self.renderer.UseDepthPeelingOff()

        self.clip = vtk.vtkPlane()
        self.lung_mapper = vtk.vtkPolyDataMapper()
        self.lung_mapper.AddClippingPlane(self.clip)
        self.shell_mapper = vtk.vtkPolyDataMapper()
        for mapper, opacity in ((self.lung_mapper, 0.52), (self.shell_mapper, 0.10)):
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.27, 0.70, 0.77)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetAmbient(0.26)
            actor.GetProperty().SetSpecular(0.22)
            actor.GetProperty().SetSpecularPower(25)
            actor.PickableOff()
            self.renderer.AddActor(actor)

        self.face = vtk.vtkPlaneSource()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(self.face.GetOutputPort())
        self.texture = vtk.vtkTexture()
        self.texture.SetColorModeToDirectScalars()
        self.texture.InterpolateOn()
        self.texture.EdgeClampOn()
        self.face_actor = vtk.vtkActor()
        self.face_actor.SetMapper(mapper)
        self.face_actor.SetTexture(self.texture)
        self.face_actor.GetProperty().LightingOff()
        self.face_actor.VisibilityOff()
        self.renderer.AddActor(self.face_actor)

        edges = vtk.vtkFeatureEdges()
        edges.SetInputConnection(self.face.GetOutputPort())
        edges.BoundaryEdgesOn()
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        frame_mapper = vtk.vtkPolyDataMapper()
        frame_mapper.SetInputConnection(edges.GetOutputPort())
        frame_mapper.ScalarVisibilityOff()
        self.spotlight_frame = vtk.vtkActor()
        self.spotlight_frame.SetMapper(frame_mapper)
        self.spotlight_frame.GetProperty().SetColor(1, 0.84, 0.42)
        self.spotlight_frame.GetProperty().SetOpacity(0.75)
        self.spotlight_frame.GetProperty().SetLineWidth(1.5)
        self.spotlight_frame.GetProperty().LightingOff()
        self.spotlight_frame.PickableOff()
        self.spotlight_frame.VisibilityOff()
        self.renderer.AddActor(self.spotlight_frame)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(2.1)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(12)
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(sphere.GetOutputPort())
        self.point = vtk.vtkActor()
        self.point.SetMapper(point_mapper)
        self.point.GetProperty().SetColor(1, 0.84, 0.42)
        self.point.GetProperty().LightingOff()
        self.point.PickableOff()
        self.point.VisibilityOff()
        self.renderer.AddActor(self.point)

    def set_volume(self, volume):
        if self.volume == volume:
            return
        ct = slicer.util.arrayFromVolume(volume)
        matrix = vtk.vtkMatrix4x4()
        volume.GetIJKToRASMatrix(matrix)
        ijk_to_ras = slicer.util.arrayFromVTKMatrix(matrix)
        mesh, self.mask, self.ras_to_mask = lung_envelope(ct, ijk_to_ras)
        self.ct = ct
        self.ras_to_ijk = np.linalg.inv(ijk_to_ras)
        bounds = np.asarray(mesh.GetBounds()).reshape(3, 2)
        self.center = bounds.mean(axis=1)
        self.extent = np.diff(bounds, axis=1).ravel() / 2
        self.lung_mapper.SetInputData(mesh)
        self.shell_mapper.SetInputData(mesh)
        self.volume = volume

    def release_volume(self):
        if self.volume is None:
            return
        self.close_spotlight(redraw=False)
        self.volume = None
        self.ct = None
        self.mask = None
        self.face_actor.VisibilityOff()
        self.spotlight_frame.VisibilityOff()
        self.lung_mapper.SetInputData(None)
        self.shell_mapper.SetInputData(None)
        self.texture.SetInputData(None)
        self.pixmap = qt.QPixmap()
        self.flat.clear()

    def open(self, focus):
        self.close_spotlight(redraw=False)
        self.pin = None
        self.press = None
        self.scene_press = None
        self.scroll_remainder = 0.0
        self.scroll_plane = None
        self.orientation.blockSignals(True)
        self.orientation.setCurrentIndex(0)
        self.orientation.blockSignals(False)
        self.window.blockSignals(True)
        self.window.setCurrentIndex(0)
        self.window.blockSignals(False)
        self.set_plane(focus)
        self.update_pin_controls()
        self.reset_camera()
        self.update_cut()

    def toggle_spotlight(self, checked):
        if not checked:
            self.close_spotlight()
            return
        self.previous_cut = (self.orientation.currentIndex, self.yaw, self.pitch)
        self.previous_camera = vtk.vtkCamera()
        self.previous_camera.DeepCopy(self.renderer.GetActiveCamera())
        self.spotlight_point = self.pin.copy()
        self.scroll_remainder = 0.0
        self.detail_strip.show()
        self.set_plane(self.pin)
        self.update_pin_controls()
        self.reset_camera()
        self.update_cut()

    def close_spotlight(self, redraw=True):
        if self.spotlight_point is None:
            return False
        self.spotlight_point = None
        self.spotlight_button.blockSignals(True)
        self.spotlight_button.setChecked(False)
        self.spotlight_button.blockSignals(False)
        self.detail_cuts.clear()
        self.detail_strip.hide()
        self.scroll_remainder = 0.0
        if redraw:
            index, yaw, pitch = self.previous_cut
            self.orientation.blockSignals(True)
            self.orientation.setCurrentIndex(index)
            self.orientation.blockSignals(False)
            self.set_plane(self.pin)
            self.yaw, self.pitch = yaw, pitch
            self.renderer.GetActiveCamera().DeepCopy(self.previous_camera)
            self.update_pin_controls()
            self.update_cut()
        self.previous_camera = None
        return True

    def move_spotlight(self, point):
        delta = np.clip(point - self.pin, -SPOTLIGHT_RADIUS, SPOTLIGHT_RADIUS)
        self.spotlight_point = self.pin + delta
        self.update_cut()

    def change_orientation(self, *_):
        if self.volume is None:
            return
        through = self.pin if self.pin is not None else self.cut_center
        self.set_plane(through)
        if self.pin is not None and self.spotlight_point is None:
            self.pin_offset = np.zeros(2)
        self.reset_camera()
        self.update_cut()

    def set_plane(self, through):
        self.base_u, self.base_v = np.asarray(PLANES[self.orientation.currentText], dtype=float)
        self.base_normal = np.cross(self.base_u, self.base_v)
        self.scene_extent = np.abs([self.base_u, self.base_v]) @ self.extent + 12
        self.yaw = self.pitch = 0.0
        if self.spotlight_point is not None:
            self.half_width = self.half_height = SPOTLIGHT_RADIUS
            spacing = 0.5
        else:
            limit = int(np.ceil(self.extent @ np.abs(self.base_normal))) + 5
            depth = round((through - self.center) @ self.base_normal)
            self.depth.blockSignals(True)
            self.depth.setRange(-limit, limit)
            self.depth.setValue(depth)
            self.depth.blockSignals(False)
            self.half_width, self.half_height = self.scene_extent
            spacing = 0.9
        self.width = 2 * int(np.ceil(self.half_width / spacing)) + 1
        self.height = 2 * int(np.ceil(self.half_height / spacing)) + 1

    def reset_camera(self):
        if self.volume is None:
            return
        camera = self.renderer.GetActiveCamera()
        direction = self.base_normal + 0.49 * self.base_u + 0.23 * self.base_v
        camera.SetFocalPoint(*self.center)
        camera.SetPosition(*(self.center + direction * np.linalg.norm(self.extent) * 3))
        camera.SetViewUp(*self.base_v)
        camera.ParallelProjectionOn()
        width, height = self.scene.renderWindow().GetSize()
        aspect = max(0.5, width / max(1, height))
        half_width, half_height = self.scene_extent
        camera.SetParallelScale(max(half_height, half_width / aspect) * 1.1)
        self.renderer.ResetCameraClippingRange()
        self.scene.scheduleRender()

    def update_cut(self, *_):
        if self.volume is None:
            return
        yaw, pitch = np.deg2rad([self.yaw, self.pitch])
        self.u = np.cos(yaw) * self.base_u + np.sin(yaw) * self.base_normal
        self.v = np.cos(pitch) * self.base_v + np.sin(pitch) * np.cross(self.u, self.base_v)
        normal = np.cross(self.u, self.v)
        if self.spotlight_point is not None:
            self.cut_center = self.spotlight_center(normal)
        elif self.pin is None:
            self.cut_center = self.center + self.depth.value * normal
        else:
            self.cut_center = self.pin - self.pin_offset[0] * self.u - self.pin_offset[1] * self.v
        plane = (
            self.cut_center, self.u, self.v,
            (self.half_width, self.half_height), (self.height, self.width),
        )
        self.intensities = sample_plane(self.ct, self.ras_to_ijk, *plane, -2048)
        rgb = windowed_rgb(self.intensities, self.window.currentText)
        mask = sample_plane(self.mask, self.ras_to_mask, *plane, 0)
        rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask * 255
        self.texture.SetInputData(image_data(rgba[None]))
        self.clip.SetOrigin(*self.cut_center)
        self.clip.SetNormal(*(-normal))
        horizontal = self.half_width * self.u
        vertical = self.half_height * self.v
        self.face.SetOrigin(*(self.cut_center - horizontal - vertical))
        self.face.SetPoint1(*(self.cut_center + horizontal - vertical))
        self.face.SetPoint2(*(self.cut_center - horizontal + vertical))
        self.face_actor.VisibilityOn()
        self.spotlight_frame.SetVisibility(self.spotlight_point is not None)

        self.pixmap = self.slice_pixmap(
            rgb, self.cut_center, self.u, self.v, compact_labels=self.spotlight_point is not None,
        )
        self.fit_slice(self.flat, self.pixmap)
        marker = self.spotlight_point if self.spotlight_point is not None else self.pin
        if marker is not None:
            self.point.SetPosition(*marker)
        if self.spotlight_point is not None:
            self.update_details()
        self.scene.scheduleRender()

    def spotlight_center(self, normal):
        return self.pin + ((self.spotlight_point - self.pin) @ normal) * normal

    def update_details(self):
        cuts = []
        for name in PLANES:
            if name == self.orientation.currentText:
                continue
            u, v = np.asarray(PLANES[name], dtype=float)
            center = self.spotlight_center(np.cross(u, v))
            intensities = sample_plane(
                self.ct, self.ras_to_ijk, center, u, v,
                (self.half_width, self.half_height), (self.height, self.width), -2048,
            )
            rgb = windowed_rgb(intensities, self.window.currentText)
            pixmap = self.slice_pixmap(rgb, center, u, v)
            cuts.append((name, center, u, v, pixmap))
        self.detail_cuts = cuts
        for (name, _, _, _, pixmap), title, label in zip(cuts, self.detail_titles, self.detail_labels):
            title.setText(name)
            label.setAccessibleName(f"{name} close-up. Click to move the crosshair.")
            self.fit_slice(label, pixmap)

    def slice_pixmap(self, rgb, center, u, v, *, compact_labels=False):
        image = qt.QImage()
        slicer.qMRMLUtils().vtkImageDataToQImage(image_data(rgb[None]), image)
        painter = qt.QPainter(image)
        draw_orientation(painter, u, self.width, compact_labels)
        point = self.spotlight_point if self.spotlight_point is not None else self.pin
        if point is not None:
            delta = point - center
            x = (delta @ u / self.half_width + 1) * (self.width - 1) / 2
            y = (1 - delta @ v / self.half_height) * (self.height - 1) / 2
            if self.spotlight_point is None:
                pen = qt.QPen(qt.QColor("#ffd76b"))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawEllipse(qt.QPointF(x, y), 5, 5)
            else:
                draw_crosshair(painter, x, y)
        painter.end()
        return qt.QPixmap.fromImage(image)

    @staticmethod
    def fit_slice(label, pixmap):
        if not pixmap.isNull() and label.width > 0 and label.height > 0:
            label.setPixmap(pixmap.scaled(label.size, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation))

    def flat_to_ras(self, position, label=None):
        if label is None:
            label = self.flat
        if label == self.flat:
            center, u_axis, v_axis = self.cut_center, self.u, self.v
        else:
            _, center, u_axis, v_axis, _ = self.detail_cuts[self.detail_labels.index(label)]
        pixmap = label.pixmap
        if pixmap is None or pixmap.isNull():
            return None
        x = position.x() - (label.width - pixmap.width()) / 2
        y = position.y() - (label.height - pixmap.height()) / 2
        if not (0 <= x < pixmap.width() and 0 <= y < pixmap.height()):
            return None
        u = (2 * (x + 0.5) / pixmap.width() - 1) * self.half_width
        v = (1 - 2 * (y + 0.5) / pixmap.height()) * self.half_height
        return center + u * u_axis + v * v_axis

    def set_pin(self, point):
        delta = point - self.cut_center
        self.pin = point
        self.pin_offset = np.array([delta @ self.u, delta @ self.v])
        self.update_pin_controls()
        self.update_cut()

    def clear_pin(self):
        through = self.pin
        self.pin = None
        self.set_plane(through)
        self.update_pin_controls()
        self.update_cut()

    def update_pin_controls(self):
        pinned = self.pin is not None
        spotlight = self.spotlight_point is not None
        self.point.SetVisibility(pinned)
        self.spotlight_button.setEnabled(pinned)
        self.depth.setVisible(not pinned)
        self.clear_pin_button.setVisible(pinned and not spotlight)
        if spotlight:
            self.hint.setText("Scroll through slices · Click to move crosshair")
            self.flat.setAccessibleName("CT close-up. Click to move the crosshair.")
        else:
            self.hint.setText("Drag CT to tilt · Scroll to rotate" if pinned else "Scroll to cut · Click CT to pin")
            self.flat.setAccessibleName("CT cut. Click to pin a position.")

    def tilt(self, yaw, pitch):
        self.yaw = float(np.clip(yaw, -70, 70))
        self.pitch = float(np.clip(pitch, -70, 70))
        self.update_cut()

    def step(self, amount, plane=None):
        if self.spotlight_point is not None:
            u, v = PLANES[plane or self.orientation.currentText]
            self.move_spotlight(self.spotlight_point - amount * np.cross(u, v))
        elif self.pin is None:
            self.depth.setValue(self.depth.value - amount)
        else:
            self.tilt(self.yaw + 2 * amount, self.pitch)

    def eventFilter(self, obj, event):
        if self.volume is None:
            return False
        detail = obj in self.detail_labels
        if detail and not self.detail_cuts:
            return False
        kind = event.type()
        if kind == qt.QEvent.Wheel and event.modifiers() == qt.Qt.NoModifier:
            plane = self.detail_cuts[self.detail_labels.index(obj)][0] if detail else self.orientation.currentText
            if self.scroll_plane != plane:
                self.scroll_remainder = 0.0
                self.scroll_plane = plane
            pixels = event.pixelDelta().y()
            self.scroll_remainder += pixels / 20 if pixels else event.angleDelta().y() / 120
            steps = int(self.scroll_remainder)
            if steps:
                self.scroll_remainder -= steps
                self.step(steps, plane)
            return True
        if obj != self.flat and not detail:
            return False
        if kind == qt.QEvent.Resize:
            pixmap = self.detail_cuts[self.detail_labels.index(obj)][-1] if detail else self.pixmap
            self.fit_slice(obj, pixmap)
        elif kind == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton:
            self.press = (event.pos(), self.yaw, self.pitch)
            return True
        elif kind in (qt.QEvent.MouseMove, qt.QEvent.MouseButtonRelease) and self.press:
            origin, yaw, pitch = self.press
            delta = event.pos() - origin
            if kind == qt.QEvent.MouseMove and self.pin is not None and self.spotlight_point is None:
                self.tilt(yaw + delta.x() * 0.24, pitch + delta.y() * 0.24)
            elif kind == qt.QEvent.MouseButtonRelease:
                self.press = None
                if delta.manhattanLength() <= 3:
                    point = self.flat_to_ras(event.pos(), obj)
                    if point is not None:
                        if self.spotlight_point is not None:
                            self.move_spotlight(point)
                        else:
                            self.set_pin(point)
            return True
        return False

    def press_scene(self, interactor, _):
        self.scene_press = interactor.GetEventPosition()

    def release_scene(self, interactor, _):
        start, self.scene_press = self.scene_press, None
        if start is None or self.volume is None:
            return
        x, y = interactor.GetEventPosition()
        if max(abs(x - start[0]), abs(y - start[1])) > 4:
            return
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self.face_actor)
        if picker.Pick(x, y, 0, self.renderer):
            point = np.asarray(picker.GetPickPosition())
            ijk = self.ras_to_mask[:3, :3] @ point + self.ras_to_mask[:3, 3]
            if ndimage.map_coordinates(self.mask, ijk[::-1, None], order=1)[0] > 0.25:
                if self.spotlight_point is not None:
                    self.move_spotlight(point)
                else:
                    self.set_pin(point)
