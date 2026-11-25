"""Curved planar CT reformat and position control for Branch Unfold."""

import numpy as np
import qt
import slicer
import vtk
from scipy.ndimage import map_coordinates
from vtk.util.numpy_support import vtk_to_numpy


RIGHT_POSTERIOR_ROUTE = ("trunk", "right-pa", "right-interlobar", "right-basal", "right-posterior")
ARTERY_FORKS = {
    "trunk": ("right-pa", "left-pa"),
    "right-pa": ("right-interlobar",),
    "right-interlobar": ("right-basal",),
    "right-basal": ("right-posterior", "right-lateral"),
    "right-posterior": ("right-posterior-lateral", "right-posterior-medial"),
    "right-lateral": ("right-lateral-larger", "right-lateral-smaller"),
    "right-posterior-lateral": (
        "right-posterior-outer-lateral-tip", "right-posterior-outer-inferior-tip",
    ),
    "right-posterior-medial": (
        "right-posterior-inner-anterior-tip", "right-posterior-inner-inferior-tip",
    ),
    "right-lateral-larger": (
        "right-lateral-larger-lateral-tip", "right-lateral-larger-inferior-tip",
    ),
    "left-pa": ("left-interlobar",),
    "left-interlobar": ("left-basal",),
    "left-basal": ("left-anterior", "left-posterior"),
}


def mapped_route(anatomy):
    """Join the scan's named guides at shared endpoints."""
    guides = {r["id"]: r for r in anatomy.guides}
    records = [guides.get(key) for key in RIGHT_POSTERIOR_ROUTE]
    if any(r is None or len(r["pathIJK"]) < 2 for r in records):
        return None

    sections = [anatomy.course_steps(record) for record in records]
    for first, second in zip(sections[:-1], sections[1:]):
        if not np.allclose(first[-1], second[0], atol=0.01, rtol=0):
            return None
    return join_sections(records, sections)


def join_sections(records, sections):
    starts = [0]
    parts = [sections[0]]
    length = len(parts[0])
    for previous, section in zip(sections[:-1], sections[1:]):
        starts.append(length - 1)
        shared = np.allclose(previous[-1], section[0], atol=0.01, rtol=0)
        part = section[1:] if shared else section
        parts.append(part)
        length += len(part)
    return records, np.concatenate(parts), np.asarray(starts)


def explore_routes(anatomy, vessels):
    """Build the route from the trunk to each selected guide and the next open fork."""
    records = {
        r["id"]: r for r in anatomy.guides
        if r.get("vessel", "artery") == "artery" and len(r["pathIJK"]) > 1
    }
    sections = {key: anatomy.course_steps(record) for key, record in records.items()}
    inv = np.linalg.inv(anatomy.affine)
    voxel_diagonal = np.linalg.norm(anatomy.affine[:3, :3])
    children = {}
    for parent, record in records.items():
        options = record.get("downstream", ARTERY_FORKS.get(parent, ()))
        options = [key for key in options if key in sections]
        # Some guides run a sample past a fork shared by both daughter guides.
        parent_steps = sections[parent]
        if len(options) == 2 and len(parent_steps) > 2 and all(
            np.allclose(sections[key][0], parent_steps[-2], atol=0.01, rtol=0)
            for key in options
        ):
            sections[parent] = parent_steps[:-1]
        end = sections[parent][-1]
        children[parent] = []
        for key in options:
            origin = sections[key][0]
            gap = np.linalg.norm(origin - end)
            if gap > 0.01:
                points = np.linspace(end, origin, 5) @ inv[:3, :3].T + inv[:3, 3]
                if gap > voxel_diagonal + 0.01 or any(vessels.at(point) != "artery" for point in points):
                    continue
            children[parent].append(key)

    routes = {}

    def options_after(path):
        return [key for key in children.get(path[-1], ()) if key not in path]

    def visit(path):
        selected = path[-1]
        route = list(path)
        options = options_after(path)
        while len(options) == 1:
            route.append(options[0])
            options = options_after(route)
        routes[selected] = (
            join_sections([records[key] for key in route], [sections[key] for key in route]),
            tuple(records[key] for key in options),
        )
        for child in options_after(path):
            visit((*path, child))

    if "trunk" in sections:
        visit(("trunk",))
    return routes


def sample_strip(ct, ras_to_ijk, frames):
    """Sample one longitudinal plane through the course, with distance in mm along x."""
    points = frames[:, :3, 3]
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    columns = np.linspace(0, distance[-1], int(np.ceil(distance[-1] / 0.25)) + 1)
    fraction = distance / distance[-1]
    chord = points[0] + fraction[:, None] * (points[-1] - points[0])
    bend = (points - chord) @ frames[len(frames) // 2, :3, :2]
    axis = int(np.argmax(np.ptp(bend, axis=0)))

    def interpolate(vectors):
        return np.column_stack([np.interp(columns, distance, v) for v in vectors.T])

    centers = interpolate(points)
    across = interpolate(frames[:, :3, axis])
    across /= np.linalg.norm(across, axis=1, keepdims=True)
    offsets = np.linspace(8, -8, 65)
    sample_ras = centers[None, :, :] + offsets[:, None, None] * across[None, :, :]
    ijk = sample_ras @ ras_to_ijk[:3, :3].T + ras_to_ijk[:3, 3]
    intensities = map_coordinates(
        ct, ijk[..., ::-1].transpose(2, 0, 1), output=np.float32, order=1, cval=-1000
    )
    bend_rows = np.interp(columns, distance, bend[:, axis] / 0.25)
    # Keep the opening animation inside the strip's viewport, even on a long route.
    return distance, intensities, bend_rows / max(1, np.max(np.abs(bend_rows)) / 12)


class UnfoldPanel(qt.QObject):
    def __init__(self, slider, scroll, choose):
        super().__init__()
        self.slider = slider
        self.scroll = scroll
        self.distance = None
        self.widget = qt.QWidget()
        self.widget.setObjectName("unfoldPanel")
        self.widget.setStyleSheet("QWidget#unfoldPanel {background:#111d27;}")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(14, 6, 14, 9)
        layout.setSpacing(3)

        self.scene = qt.QGraphicsScene(self.widget)
        self.view = qt.QGraphicsView(self.scene, self.widget)
        self.view.setFrameShape(qt.QFrame.NoFrame)
        self.view.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.view.setFixedHeight(113)
        self.view.setBackgroundBrush(qt.QBrush(qt.QColor("#111d27")))
        self.view.setRenderHint(qt.QPainter.SmoothPixmapTransform)
        self.view.setAccessibleName("Straightened CT along the selected route")
        self.view.setToolTip("Drag, scroll, or use the arrow keys to move along the route")
        self.view.setFocusPolicy(qt.Qt.StrongFocus)
        self.view.installEventFilter(self)
        self.viewport = self.view.viewport()
        self.viewport.setCursor(qt.Qt.CrossCursor)
        self.viewport.installEventFilter(self)
        layout.addWidget(self.view)
        self.labels = qt.QHBoxLayout()
        self.labels.addWidget(qt.QLabel("Pulmonary trunk"))
        self.labels.addStretch()
        self.readout = qt.QLabel()
        self.readout.setAlignment(qt.Qt.AlignCenter)
        self.labels.addWidget(self.readout)
        self.labels.addStretch()
        self.route_end = qt.QLabel()
        self.labels.addWidget(self.route_end)
        layout.addLayout(self.labels)
        self.fork = qt.QWidget()
        options = qt.QHBoxLayout(self.fork)
        options.setContentsMargins(0, 2, 0, 0)
        options.addStretch()
        self.choices = ()
        self.choice_buttons = []
        for index in range(2):
            button = qt.QPushButton()
            button.connect("clicked()", lambda i=index: choose(self.choices[i]))
            options.addWidget(button)
            self.choice_buttons.append(button)
        options.addStretch()
        self.fork.hide()
        layout.addWidget(self.fork)
        self.image = self.scene.addPixmap(qt.QPixmap())
        self.image.setTransformationMode(qt.Qt.SmoothTransformation)
        pen = qt.QPen(qt.QColor("#f5ffb3"))
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.cursor = self.scene.addLine(0, 0, 0, 0, pen)
        self.cursor.setZValue(2)
        self.active_range = self.scene.addLine(0, 0, 0, 0, pen)
        self.active_range.setZValue(1)
        self.tick_pen = qt.QPen(qt.QColor("#a5b7c5"))
        self.tick_pen.setCosmetic(True)
        self.ticks = []
        self.animation = qt.QVariantAnimation(self.widget)
        self.animation.setDuration(520)
        self.animation.setStartValue(1.0)
        self.animation.setEndValue(0.0)
        self.animation.setEasingCurve(qt.QEasingCurve(qt.QEasingCurve.OutCubic))
        self.animation.connect("valueChanged(QVariant)", self.render)
        self.widget.hide()

    def open(self, distance, intensities, bend_rows, starts, window, level, end, animate):
        self.animation.stop()
        self.distance = distance
        self.route_end.setText(end)
        self.boundaries = distance[np.r_[starts, len(distance) - 1]]
        self.intensities = intensities
        self.bend_rows = bend_rows
        # Leave room for the curved strip as it straightens into the center of the frame.
        self.padding = int(np.ceil(np.max(np.abs(bend_rows)))) + 2
        rows, self.columns = intensities.shape
        self.height = rows + 2 * self.padding
        self.row_grid, self.column_grid = np.indices((self.height, self.columns), dtype=float)
        self.scene.setSceneRect(-0.5, 0, self.columns, self.height)
        while len(self.ticks) < len(starts) - 1:
            tick = self.scene.addLine(0, 0, 0, 0, self.tick_pen)
            tick.setZValue(1)
            self.ticks.append(tick)
        for index, tick in enumerate(self.ticks):
            tick.setVisible(index < len(starts) - 1)
            if index < len(starts) - 1:
                x = self.column_at(self.boundaries[index + 1])
                tick.setLine(x, 3, x, 9)
        self.amount = float(animate)
        self.set_window(window, level)
        self.widget.show()
        self.fit_view()
        self.view.setFocus()
        if animate:
            self.animation.start()

    def close(self):
        self.animation.stop()
        self.set_choices(())
        self.widget.hide()
        self.distance = None
        self.image.setPixmap(qt.QPixmap())

    def set_window(self, window, level):
        if self.distance is not None:
            self.window = window
            self.level = level
            self.render(self.amount)

    def render(self, amount):
        self.amount = float(amount)
        y = self.row_grid - self.padding + self.amount * self.bend_rows
        pixels = map_coordinates(
            self.intensities, [y, self.column_grid], order=1, cval=-1000
        )
        gray = np.clip((pixels - self.level) / self.window + 0.5, 0, 1) * 255
        # VTK image coordinates run bottom-up and Qt image coordinates run top-down.
        data = vtk.vtkImageData()
        data.SetDimensions(self.columns, self.height, 1)
        data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        vtk_to_numpy(data.GetPointData().GetScalars())[:] = gray[::-1].reshape(-1, 1)
        image = qt.QImage()
        slicer.qMRMLUtils().vtkImageDataToQImage(data, image)
        self.image.setPixmap(qt.QPixmap.fromImage(image))
        self.image.setPos(-0.5, 0)

    def column_at(self, mm):
        return mm / self.distance[-1] * (self.columns - 1)

    def set_choices(self, choices):
        if choices == self.choices:
            return
        self.choices = choices
        for button, record in zip(self.choice_buttons, choices):
            button.setText(record.get("short", record["name"]))
            button.setAccessibleName(record["name"])
        self.fork.setVisible(bool(choices))

    def set_position(self, index, section):
        mm = self.distance[index]
        if self.choices:
            text = "Choose a branch"
        elif index == 0:
            text = "Start of mapped route"
        elif index == len(self.distance) - 1:
            text = "End of mapped route"
        else:
            text = f"Drag to scrub   ·   {mm:.0f} / {self.distance[-1]:.0f} mm"
        self.readout.setText(text)
        x = self.column_at(mm)
        self.cursor.setLine(x, 0, x, self.height)
        left, right = self.column_at(self.boundaries[section:section + 2])
        self.active_range.setLine(left, 3, right, 3)

    def fit_view(self):
        if self.distance is not None:
            self.view.fitInView(self.scene.sceneRect, qt.Qt.KeepAspectRatio)
            left = self.view.mapFromScene(qt.QPointF(0, 0)).x()
            right = self.view.mapFromScene(qt.QPointF(self.columns - 1, 0)).x()
            self.labels.setContentsMargins(left, 0, max(0, self.viewport.width - right), 0)

    def eventFilter(self, obj, event):
        if self.distance is None:
            return False
        kind = event.type()
        if obj == self.view and kind == qt.QEvent.KeyPress:
            positions = {
                qt.Qt.Key_Left: self.slider.value - 1,
                qt.Qt.Key_Right: self.slider.value + 1,
                qt.Qt.Key_Home: 0,
                qt.Qt.Key_End: self.slider.maximum,
            }
            if event.key() in positions:
                self.slider.setValue(positions[event.key()])
                return True
        if obj != self.viewport:
            return False
        if kind == qt.QEvent.Resize:
            self.fit_view()
        elif kind == qt.QEvent.Wheel and event.modifiers() == qt.Qt.NoModifier:
            self.scroll(event.angleDelta().y(), event.pixelDelta().y())
            return True
        elif ((kind == qt.QEvent.MouseButtonPress and event.button() == qt.Qt.LeftButton)
              or (kind == qt.QEvent.MouseMove and event.buttons() & qt.Qt.LeftButton)):
            self.view.setFocus()
            x = self.view.mapToScene(event.pos()).x()
            mm = x / (self.columns - 1) * self.distance[-1]
            self.slider.setValue(int(np.argmin(np.abs(self.distance - mm))))
            return True
        return False
