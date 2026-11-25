"""Keep two CT scans at the same physical offset from their saved locations."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qt
import slicer

from grid import load_grid
from lung_cutaway import (
    PLANES, SPOTLIGHT_RADIUS, WINDOWS, LungCutaway, draw_crosshair,
    draw_orientation, image_data, sample_plane, windowed_rgb,
)


@dataclass
class ComparisonScan:
    date: str
    ct: np.ndarray
    ras_to_ijk: np.ndarray
    anchor: np.ndarray
    axial_spacing: float

    @classmethod
    def load(cls, folder, record):
        filename = record["file"]
        if Path(filename).name != filename:
            raise ValueError("Comparison volume must be stored with the pair")
        ct, affine, _ = load_grid(folder / filename)
        ras_to_ijk = np.linalg.inv(affine)
        anchor = np.asarray(record["anchorRAS"], dtype=float)
        ijk = ras_to_ijk[:3, :3] @ anchor + ras_to_ijk[:3, 3]
        if ct.ndim != 3 or not np.all((0 <= ijk) & (ijk <= np.array(ct.shape[::-1]) - 1)):
            raise ValueError("Comparison location is outside its source scan")
        axial_spacing = abs(affine[2, 2]) if np.allclose(affine[2, :2], 0) else 0.0
        return cls(record["date"], ct, ras_to_ijk, anchor, axial_spacing)


class SpotlightComparison(qt.QObject):
    def __init__(self, back):
        super().__init__()
        self.scans = []
        self.pixmaps = []
        self.scroll_remainder = 0.0
        self.widget = qt.QWidget()
        self.widget.setObjectName("shell")
        layout = qt.QVBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header, row = LungCutaway.bar()
        layout.addWidget(header)
        exit_button = qt.QPushButton("‹ Back")
        exit_button.connect("clicked()", back)
        row.addWidget(exit_button)
        title = qt.QLabel("Spotlight comparison")
        title.setObjectName("title")
        row.addWidget(title)
        row.addStretch()
        self.orientation = qt.QComboBox()
        self.orientation.setAccessibleName("Comparison plane")
        self.orientation.addItems(list(PLANES))
        self.orientation.connect("currentIndexChanged(int)", self.change_plane)
        row.addWidget(self.orientation)
        self.window = qt.QComboBox()
        self.window.setAccessibleName("CT window for both scans")
        self.window.addItems(list(WINDOWS))
        self.window.connect("currentIndexChanged(int)", self.update_views)
        row.addWidget(self.window)

        panels = qt.QHBoxLayout()
        panels.setSpacing(2)
        layout.addLayout(panels, 1)
        self.titles = []
        self.panes = []
        for _ in range(2):
            column = qt.QVBoxLayout()
            column.setSpacing(0)
            panels.addLayout(column, 1)
            title = qt.QLabel()
            title.setObjectName("location")
            title.setAlignment(qt.Qt.AlignCenter)
            title.setFixedHeight(35)
            column.addWidget(title)
            label = qt.QLabel()
            label.setStyleSheet("background: #080f16")
            label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Ignored)
            label.setAlignment(qt.Qt.AlignCenter)
            label.installEventFilter(self)
            column.addWidget(label, 1)
            self.titles.append(title)
            self.panes.append(label)

        footer, row = LungCutaway.bar()
        footer.setFixedHeight(45)
        layout.addWidget(footer)
        hint = qt.QLabel("Scroll either scan")
        hint.setToolTip("Each scan starts at its saved location. Both move by the same distance.")
        row.addWidget(hint)
        self.slider = qt.QSlider(qt.Qt.Horizontal)
        self.slider.setAccessibleName("Linked slice offset from the saved locations")
        self.slider.connect("valueChanged(int)", self.update_views)
        row.addWidget(self.slider, 1)
        self.readout = qt.QLabel()
        self.readout.setMinimumWidth(72)
        self.readout.setAlignment(qt.Qt.AlignCenter)
        row.addWidget(self.readout)
        reset = qt.QPushButton("Reset")
        reset.setToolTip("Return to the saved comparison slices")
        reset.connect("clicked()", self.reset)
        row.addWidget(reset)

    def open(self, manifest, series_uid):
        source = json.loads(manifest.read_text())
        if source["version"] != 1 or source["seriesUID"] != series_uid or len(source["scans"]) != 2:
            raise ValueError("Comparison does not match this study")
        scans = [ComparisonScan.load(manifest.parent, record) for record in source["scans"]]
        first, second = (scan.axial_spacing for scan in scans)
        self.axial_step = first if first > 0 and np.isclose(first, second) else 1.0
        self.orientation.setCurrentIndex(list(PLANES).index("Axial"))
        self.window.setCurrentIndex(0)
        self.scans = scans
        for scan, title, pane in zip(scans, self.titles, self.panes):
            title.setText(scan.date)
            pane.setAccessibleName(f"{scan.date} CT. Scroll to move both scans.")
        self.change_plane()

    def clear(self):
        self.scans = []
        self.pixmaps = []
        for pane in self.panes:
            pane.clear()

    def reset(self):
        self.slider.setValue(0)

    def change_plane(self, *_):
        if not self.scans:
            return
        plane = self.orientation.currentText
        u, v = np.asarray(PLANES[plane], dtype=float)
        self.u, self.v = u, v
        self.normal = -np.cross(u, v)
        self.slice_step = self.axial_step if plane == "Axial" else 1.0
        limit = int(SPOTLIGHT_RADIUS / self.slice_step)
        self.scroll_remainder = 0.0
        self.slider.blockSignals(True)
        self.slider.setRange(-limit, limit)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.update_views()

    @property
    def offset(self):
        return self.slider.value * self.slice_step

    def step(self, amount):
        self.slider.setValue(self.slider.value + amount)

    def update_views(self, *_):
        if not self.scans:
            return
        offset = self.offset
        self.readout.setText(f"{offset:+.1f} mm" if offset else "0.0 mm")
        self.pixmaps = [self.render(scan, offset) for scan in self.scans]
        for pane, pixmap in zip(self.panes, self.pixmaps):
            LungCutaway.fit_slice(pane, pixmap)

    def render(self, scan, offset):
        center = scan.anchor + offset * self.normal
        size = 201
        intensities = sample_plane(
            scan.ct, scan.ras_to_ijk, center, self.u, self.v,
            (SPOTLIGHT_RADIUS, SPOTLIGHT_RADIUS), (size, size), -2048,
        )
        rgb = windowed_rgb(intensities, self.window.currentText)
        image = qt.QImage()
        slicer.qMRMLUtils().vtkImageDataToQImage(image_data(rgb[None]), image)
        painter = qt.QPainter(image)
        draw_orientation(painter, self.u, size, True)
        midpoint = (size - 1) / 2
        draw_crosshair(painter, midpoint, midpoint)
        painter.end()
        return qt.QPixmap.fromImage(image)

    def eventFilter(self, obj, event):
        if not self.pixmaps:
            return False
        if event.type() == qt.QEvent.Resize:
            LungCutaway.fit_slice(obj, self.pixmaps[self.panes.index(obj)])
        elif event.type() == qt.QEvent.Wheel and event.modifiers() == qt.Qt.NoModifier:
            pixels = event.pixelDelta().y()
            self.scroll_remainder += pixels / 20 if pixels else event.angleDelta().y() / 120
            steps = int(self.scroll_remainder)
            if steps:
                self.scroll_remainder -= steps
                self.step(steps)
            return True
        return False
