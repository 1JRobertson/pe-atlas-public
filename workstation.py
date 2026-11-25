"""Dedicated CT review window running on the 3D Slicer engine."""

import faulthandler
import json
import os
import sys
import traceback
from html import escape
from pathlib import Path

import numpy as np
import qt
import slicer
import vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from artery_landmarks import ArteryMap, VesselLabels, course_frames
from artery_picking import ArteryPicker
from branch_unfold import UnfoldPanel, explore_routes, mapped_route, sample_strip
from lung_cutaway import LungCutaway
from spotlight_comparison import SpotlightComparison

SUPPORT = Path(
    os.environ.get(
        "PE_ATLAS_HOME", Path.home() / "Library/Application Support/PE Atlas Public"
    )
)
SUPPORT.mkdir(parents=True, exist_ok=True)
LOG = SUPPORT / "workstation.log"
crash_log = LOG.open("a")
faulthandler.enable(file=crash_log)
LAYOUT = 763
LAYOUT_XML = """<layout type="horizontal" split="true">
<item splitSize="540"><view class="vtkMRMLViewNode" singletontag="1"/></item>
<item splitSize="460"><layout type="vertical" split="true">
<item><view class="vtkMRMLSliceNode" singletontag="Red"><property name="orientation" action="default">Axial</property></view></item>
<item><view class="vtkMRMLSliceNode" singletontag="Green"><property name="orientation" action="default">Coronal</property></view></item>
</layout></item></layout>"""
STYLE = """
QMainWindow,QWidget#shell {background:#101820;color:#eff3f5;}
QWidget#controls,QWidget#footer,QWidget#navigation {background:#111d27;}
QLabel {color:#d4e1e9;font-size:13px;}
QLabel#title {font-size:22px;font-weight:600;color:#f4f8fb;}
QLabel#location {font-size:16px;font-weight:600;color:#e6eff6;}
QLabel#muted {color:#a5b7c5;font-size:12px;}
QLabel#subsegmentalReview {background:#13242e;color:#edf4f8;border:1px solid #405561;border-radius:6px;padding:5px 8px;}
QPushButton,QComboBox,QSpinBox {font-size:13px;color:#e6eff6;background:#233440;border:1px solid #40515e;border-radius:5px;padding:5px 9px;min-height:19px;}
QPushButton:checked {background:#126354;border-color:#4bc8ad;color:white;}
QPushButton:hover,QComboBox:hover {border-color:#83a5b8;}
QPushButton:disabled {color:#78848d;}
QPushButton#explore {background:#18483f;border-color:#3b8d7a;color:#d2ffef;padding:6px 12px;}
QPushButton#explore:hover {background:#225f52;border-color:#68cdb4;}
QPushButton#tools {font-size:18px;font-weight:600;padding:2px 9px;}
QPushButton#tools::menu-indicator,QPushButton#dropdown::menu-indicator {image:none;width:0;}
QMenu {background:#1a2935;color:#e6eff6;border:1px solid #40515e;padding:4px;}
QMenu::item {padding:6px 32px 6px 25px;}
QMenu::item:selected {background:#285e62;}
QMenu::item:disabled {color:#8c9aa3;}
QMenu::separator {height:1px;background:#40515e;margin:4px 8px;}
QComboBox QAbstractItemView {background:#1c2b36;color:#edf4f8;selection-background-color:#285e62;}
QSlider::groove:horizontal {height:4px;background:#41515d;border-radius:2px;}
QSlider::handle:horizontal {background:#62d4bc;width:12px;margin:-4px 0;border-radius:6px;}
QToolTip {color:#fff;background:#263744;}
"""


def log(message):
    with LOG.open("a") as f:
        print(message, file=f, flush=True)


def label(text, name=None):
    w = qt.QLabel(text)
    if name:
        w.setObjectName(name)
    return w


def bar(name):
    w = qt.QWidget()
    w.setObjectName(name)
    row = qt.QHBoxLayout(w)
    row.setContentsMargins(14, 7, 14, 7)
    row.setSpacing(8)
    return w, row


def connect_button(row, text, callback, tip="", check=False):
    b = qt.QPushButton(text)
    b.setCheckable(check)
    b.setToolTip(tip)
    b.connect("clicked()", callback)
    row.addWidget(b)
    return b


def menu_action(menu, text, callback, *, check=False):
    action = menu.addAction(text.replace("&", "&&"))
    action.setCheckable(check)
    action.connect("triggered()", callback)
    return action


def dropdown(row, text):
    button = qt.QPushButton(f"{text} ▾")
    button.setObjectName("dropdown")
    button.setAccessibleName(text)
    button.setEnabled(False)
    menu = qt.QMenu(button)
    button.setMenu(menu)
    row.addWidget(button)
    return button, menu


class NativeEvents(qt.QObject):
    def __init__(self, owner):
        qt.QObject.__init__(self)
        self.owner = owner

    def eventFilter(self, obj, event):
        a = self.owner
        t = event.type()
        if obj == a.axial_view and t == qt.QEvent.Resize:
            a.position_subsegmental_review(event.size().width())
        if obj == a.window and t == qt.QEvent.Close:
            if not a.save():
                event.ignore()
                return True
            qt.QTimer.singleShot(50, slicer.app.quit)
        if obj != a.window:
            if t == qt.QEvent.Wheel and event.modifiers() == qt.Qt.NoModifier:
                angle, pixels = event.angleDelta().y(), event.pixelDelta().y()
                if (angle or pixels) and (a.branch_steps is not None or a.start_route_scroll()):
                    a.scroll_branch(angle, pixels)
                    return True
            if a.branch_steps is not None and t in (
                qt.QEvent.MouseButtonPress, qt.QEvent.MouseButtonRelease,
                qt.QEvent.MouseButtonDblClick, qt.QEvent.MouseMove,
                qt.QEvent.KeyPress, qt.QEvent.Wheel,
            ):
                return obj == a.peer_view or a.peer_view.isAncestorOf(obj)
        return False


class Workstation:
    def __init__(self):
        self.ready = False
        self.dirty = False
        self.root = None
        self.layer_choices = {}
        self.case_nodes = []
        self.obs = None
        self.worker = None
        self.jumping = False
        self.branch_steps = None
        self.branch_marker_renderer = None
        self.unfold_route = None
        self.active_route = None
        self.followed_guide = None
        self.route_section = None
        self.free_scroll = False
        self.cutaway = None
        self.comparison = None
        self.window = qt.QMainWindow()
        self.window.setWindowTitle("PE Atlas")
        self.window.resize(1360, 850)
        self.window.setMinimumSize(920, 590)
        root = qt.QWidget()
        root.setObjectName("shell")
        self.pages = qt.QStackedWidget()
        self.study_page = root
        self.pages.addWidget(root)
        self.window.setCentralWidget(self.pages)
        col = qt.QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        self.review_bars = []
        w, row = bar("controls")
        self.review_bars.append(w)
        col.addWidget(w)
        self.cases = qt.QComboBox()
        self.cases.setAccessibleName("Study")
        row.addWidget(self.cases)
        self.landmarks_button, self.anatomy_menu = dropdown(row, "Landmarks")
        self.regions_button, self.regions = dropdown(row, "Review locations")
        row.addSpacing(8)
        self.hint = label("Click the artery tree", "location")
        self.hint.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        row.addWidget(self.hint, 1)
        self.unfold_button = connect_button(
            row,
            "Explore artery",
            self.start_unfold,
            "Follow the selected artery or choose a branch at a fork",
        )
        self.unfold_button.setObjectName("explore")
        self.unfold_button.hide()
        self.cutaway_button = connect_button(
            row, "Lung Cutaway", self.start_cutaway, "Move a CT cut through the lungs"
        )
        self.cutaway_button.setEnabled(False)
        more = qt.QPushButton("⋯")
        more.setObjectName("tools")
        more.setAccessibleName("More tools")
        more.setToolTip("More tools")
        row.addWidget(more)
        self.tools = qt.QMenu(more)
        more.setMenu(self.tools)
        self.setup_tools()
        self.unfold_header, row = bar("navigation")
        col.addWidget(self.unfold_header)
        self.unfold_exit = connect_button(row, "‹ Back", self.exit_unfold, "Back to the study")
        row.addSpacing(6)
        self.unfold_location = label("", "title")
        row.addWidget(self.unfold_location)
        row.addStretch()
        row.addWidget(label("Axial CT ↑  ·  Cross-section ↓", "muted"))
        row.addSpacing(10)
        self.unfold_image = label("")
        row.addWidget(self.unfold_image)
        self.unfold_header.hide()
        viewbox = qt.QWidget()
        col.addWidget(viewbox, 1)
        self.manager = slicer.app.layoutManager()
        if not self.manager:
            self.manager = slicer.qSlicerLayoutManager()
            slicer.app.setLayoutManager(self.manager)
        self.manager.setViewport(viewbox)
        self.manager.setMRMLScene(slicer.mrmlScene)
        layout = self.manager.layoutLogic().GetLayoutNode()
        layout.AddLayoutDescription(LAYOUT, LAYOUT_XML)
        self.manager.setLayout(LAYOUT)
        self.three = self.manager.threeDWidget(0)
        self.view = self.three.threeDView()
        self.slice = self.manager.sliceWidget("Red")
        self.axial_view = self.slice.sliceView()
        self.subsegmental_review = qt.QLabel(self.axial_view)
        self.subsegmental_review.setObjectName("subsegmentalReview")
        self.subsegmental_review.setTextFormat(qt.Qt.RichText)
        self.subsegmental_review.setWordWrap(True)
        self.subsegmental_review.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
        self.subsegmental_review.hide()
        self.visible_subsegmental_review = None
        self.sn = self.slice.mrmlSliceNode()
        self.peer = self.manager.sliceWidget("Green")
        self.peer_view = self.peer.sliceView()
        self.peer.sliceController().hide()
        self.peer_view.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
        self.coronal = self.peer.mrmlSliceNode()
        self.coronal.SetOrientationToCoronal()
        self.three.threeDController().hide()
        self.slice.sliceController().hide()
        vn = self.three.mrmlViewNode()
        # Use VTK's order-independent transparency without repeated depth passes.
        vn.SetUseDepthPeeling(0)
        vn.SetBackgroundColor(0.035, 0.065, 0.093)
        vn.SetBackgroundColor2(0.075, 0.115, 0.15)
        vn.SetOrientationMarkerSize(vn.OrientationMarkerSizeSmall)
        vn.SetBoxVisible(False)
        vn.SetAxisLabelsVisible(False)
        vn.SetOrientationMarkerType(vn.OrientationMarkerTypeCube)
        w, row = bar("controls")
        self.edit_bar = w
        col.addWidget(w)
        w.hide()
        self.buttons = {}
        self.edit_modes = qt.QButtonGroup(self.window)
        self.edit_modes.setExclusive(True)
        for mode, text in [("Paint", "Draw"), ("Erase", "Erase")]:
            b = connect_button(row, text, lambda m=mode: self.mode(m), check=True)
            self.edit_modes.addButton(b)
            self.buttons[mode] = b
        row.addWidget(label("Brush", "muted"))
        self.brush = qt.QSpinBox()
        self.brush.setRange(1, 15)
        self.brush.value = 3
        self.brush.setSuffix(" mm")
        row.addWidget(self.brush)
        self.undo = connect_button(row, "Undo", lambda: self.editor.undo())
        self.redo = connect_button(row, "Redo", lambda: self.editor.redo())
        row.addStretch()
        connect_button(row, "Done", lambda: self.mode(""))
        w, row = bar("footer")
        self.review_bars.append(w)
        col.addWidget(w)
        self.slice_label = label("Axial CT")
        row.addWidget(self.slice_label)
        row.addSpacing(6)
        self.slider = qt.QSlider(qt.Qt.Horizontal)
        self.slider.setAccessibleName("Axial slice")
        row.addWidget(self.slider, 1)
        row.addSpacing(10)
        self.unfold_panel = UnfoldPanel(self.slider, self.scroll_branch, self.choose_explore_branch)
        col.insertWidget(col.indexOf(viewbox) + 1, self.unfold_panel.widget)
        self.status = label("Loading study…", "muted")
        row.addWidget(self.status)
        self.shortcuts = []
        for key, callback in (
            ("A", lambda: self.step_position(-1)),
            ("D", lambda: self.step_position(1)),
            ("R", self.overview),
            ("Escape", self.exit_interaction),
        ):
            shortcut = qt.QShortcut(qt.QKeySequence(key), self.window)
            shortcut.connect("activated()", callback)
            self.shortcuts.append(shortcut)
        self.window.setStyleSheet(STYLE)
        self.editor = slicer.qMRMLSegmentEditorWidget()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.editor.setMaximumNumberOfUndoStates(12)
        self.editor.setUndoEnabled(True)
        self.timer = qt.QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(650)
        self.timer.connect("timeout()", self.save)
        self.branch_render = qt.QTimer(self.window)
        self.branch_render.setSingleShot(True)
        self.branch_render.connect("timeout()", self.render_branch_views)
        slicer.app.connect("aboutToQuit()", self.save)
        self.slider.connect("valueChanged(int)", self.slide)
        self.brush.connect("valueChanged(int)", self.set_brush)
        self.cases.connect("currentIndexChanged(int)", self.switch)
        self.events = NativeEvents(self)
        self.event_widgets = [self.window, self.axial_view]
        self.event_widgets.extend(
            widget for view in (self.axial_view, self.peer_view)
            for widget in slicer.util.findChildren(view)
            if widget.isWidgetType() and widget != self.subsegmental_review
        )
        for widget in self.event_widgets:
            widget.installEventFilter(self.events)
        slicer.app.connect("aboutToQuit()", self.remove_event_filters)
        self.interactor = self.view.interactorStyle().GetInteractor()
        self.press3d = None
        self.interactor.AddObserver("LeftButtonPressEvent", self.press3, 1.1)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self.release3, 1.1)
        self.view.interactorStyle().AddObserver(
            "EndInteractionEvent", self.release3, 1.1
        )
        self.ct_interactor = self.slice.sliceView().interactorStyle().GetInteractor()
        self.axial_press = None
        self.ct_interactor.AddObserver("LeftButtonPressEvent", self.press_axial, 1.1)
        self.ct_interactor.AddObserver(
            "LeftButtonReleaseEvent", self.release_axial, 1.1
        )
        self.sn.AddObserver(vtk.vtkCommand.ModifiedEvent, self.slice_moved)
        for view in (self.view, self.axial_view, self.peer_view):
            self.configure_presentation(view)
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        # Let Slicer's queued startup slice offsets settle before centering the CT.
        qt.QTimer.singleShot(0, self.load_choices)
        self.refresh = qt.QTimer()
        self.refresh.setInterval(4000)
        self.refresh.connect("timeout()", self.load_choices)
        self.refresh.start()

    @staticmethod
    def configure_presentation(view):
        if sys.platform == "darwin" and os.uname().machine == "arm64":
            widget = view.VTKWidget()
            fmt = widget.format()
            # Let macOS compose the latest completed frame without blocking Qt on a refresh.
            fmt.setSwapInterval(0)
            widget.setFormat(fmt)

    def setup_tools(self):
        self.follow_button = menu_action(self.tools, "Follow selected branch", self.toggle_follow, check=True)
        self.follow_button.setEnabled(False)
        self.edit_action = menu_action(self.tools, "Edit outlines", lambda: self.mode("Paint"))
        self.tools.addSeparator()
        display = self.tools.addMenu("Display")
        self.layer = display.addMenu("Review layer")
        self.layer.setToolTipsVisible(True)
        self.layer_group = qt.QActionGroup(self.layer)
        self.layer_actions = []
        for index, name in enumerate(("Detector", "Branch review")):
            action = menu_action(self.layer, name, lambda i=index: self.change_layer(i), check=True)
            self.layer_group.addAction(action)
            self.layer_actions.append(action)
        windows = display.addMenu("CT window")
        self.window_group = qt.QActionGroup(windows)
        angio = menu_action(windows, "Angiogram", self.windowing, check=True)
        self.lung_window = menu_action(windows, "Lung", self.windowing, check=True)
        self.window_group.addAction(angio)
        self.window_group.addAction(self.lung_window)
        angio.setChecked(True)
        self.guides_button = menu_action(display, "Artery guides", self.toggle_guides, check=True)
        self.overlay = menu_action(display, "Axial outlines", self.toggle_overlay, check=True)
        self.uncertain_button = menu_action(display, "Uncertain extent (amber)", self.toggle_uncertain, check=True)
        self.reference_button = menu_action(display, "Expert outlines (violet)", self.toggle_reference, check=True)
        self.reference_button.setVisible(False)
        menu_action(self.tools, "Reset 3D view", self.overview)
        self.analyze = menu_action(self.tools, "Analyze CT", lambda: self.run_detector(self.root))

    def remove_event_filters(self):
        for widget in self.event_widgets:
            widget.removeEventFilter(self.events)

    def load_choices(self):
        if self.worker:
            return
        base = SUPPORT / "Cases"
        valid = sorted(
            [
                f.parent
                for f in base.glob("*/source.json")
                if (f.parent / "ct.nii.gz").exists()
                and (f.parent / "vessels/lung_arteries.nii.gz").exists()
            ],
            key=lambda p: p.name,
        )
        if getattr(self, "choices", None) == valid:
            return
        self.choices = valid
        self.cases.blockSignals(True)
        self.cases.clear()
        for p in valid:
            self.cases.addItem(json.loads((p / "source.json").read_text())["title"])
        self.cases.setCurrentIndex(valid.index(self.root) if self.root in valid else 0)
        self.cases.blockSignals(False)
        if valid and self.root is None:
            self.switch(0)
        if not valid:
            self.status.setText("No prepared studies in " + str(base))

    def switch(self, index):
        if not (0 <= index < len(self.choices)):
            return
        dest = self.choices[index]
        if dest == self.root:
            return
        if self.root and not self.save():
            self.cases.blockSignals(True)
            self.cases.setCurrentIndex(self.choices.index(self.root))
            self.cases.blockSignals(False)
            return
        manifest = dest / "candidates.json"
        try:
            data = json.loads(manifest.read_text())
            current = (
                data.get("version") == 2 and (dest / data["evidenceFile"]).is_file()
            )
        except (OSError, ValueError, KeyError):
            current = False
        if not current or not all(
            (dest / k).exists()
            for k in ("candidates.json", "automatic.nii.gz", "artery-ras.vtp")
        ):
            self.run_detector(dest)
            return
        try:
            self.load(dest)
        except Exception:
            self.ready = False
            log(traceback.format_exc())
            self.status.setText("Case could not be opened. See workstation.log.")

    def run_detector(self, dest):
        if not dest or self.worker or not self.save():
            return
        contents = Path(qt.QCoreApplication.applicationFilePath()).parent.parent
        self.worker = qt.QProcess(self.window)
        self.analysis_case = dest
        self.analyze.setEnabled(False)
        self.cases.setEnabled(False)
        self.status.setText("Analyzing CT locally…")
        self.worker.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.worker.connect("finished(int,QProcess::ExitStatus)", self.analysis_done)
        self.worker.connect(
            "errorOccurred(QProcess::ProcessError)", self.analysis_error
        )
        self.worker.start(
            str(contents / "bin/PythonSlicer"),
            [str(Path(__file__).resolve().parent / "tools/prepare_case.py"), str(dest)],
        )

    def analysis_error(self, error):
        if self.worker and error == qt.QProcess.FailedToStart:
            self.analysis_done(1, None)

    def analysis_done(self, code, state):
        if not self.worker:
            return
        worker = self.worker
        try:
            if code:
                log(str(worker.readAllStandardOutput().data()))
                self.status.setText("Analysis failed — see workstation.log")
                return
            if self.save():
                self.load(self.analysis_case, reviewed=False)
        except Exception:
            log(traceback.format_exc())
            self.status.setText("Could not open analysis results")
        finally:
            self.worker = None
            worker.deleteLater()
            self.analyze.setEnabled(True)
            self.cases.setEnabled(True)

    def toggle_reference(self):
        if not self.ready:
            return
        if self.reference is None and self.reference_button.checked:
            src = slicer.util.loadLabelVolume(
                str(self.root / "reference.nii.gz"), {"show": False}
            )
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            node.CreateDefaultDisplayNodes()
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                src, node
            )
            self.remove_owned(src)
            self.reference = node
            self.case_nodes.append(node)
            sg = node.GetSegmentation()
            if sg.GetNumberOfSegments():
                sg.GetNthSegment(0).SetColor(0.78, 0.48, 1.0)
            node.CreateClosedSurfaceRepresentation()
            d = node.GetDisplayNode()
            self.limit_contours_to_axial(d)
            d.SetOpacity2DFill(0)
            d.SetOpacity2DOutline(1)
            d.SetSliceIntersectionThickness(2)
            d.SetOpacity3D(0.48)
            slicer.util.setSliceViewerLayers(background=self.vol, label=None)
        if self.reference:
            self.reference.GetDisplayNode().SetVisibility(self.reference_button.checked)

    def limit_contours_to_axial(self, display):
        # Preserve an unobscured second CT plane at the same point.
        display.AddViewNodeID(self.sn.GetID())
        display.AddViewNodeID(self.three.mrmlViewNode().GetID())

    def toggle_uncertain(self):
        if self.ready:
            display = self.uncertain.GetDisplayNode()
            display.SetVisibility(self.uncertain_button.checked)
            display.SetVisibility2D(self.overlay.checked)

    def from_tier(self, array, value, name, color):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        node.CreateDefaultDisplayNodes()
        node.SetReferenceImageGeometryParameterFromVolumeNode(self.vol)
        ident = node.GetSegmentation().AddEmptySegment(name, name, color)
        mask = array == value
        if mask.any():
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask.astype(np.uint8), node, ident, self.evidence_volume
            )
        node.CreateClosedSurfaceRepresentation()
        return node

    def remove_owned(self, node):
        if not node or not node.GetScene():
            return
        associated = []
        if node.IsA("vtkMRMLDisplayableNode"):
            associated.extend(
                node.GetNthDisplayNode(i) for i in range(node.GetNumberOfDisplayNodes())
            )
        if node.IsA("vtkMRMLStorableNode"):
            associated.append(node.GetStorageNode())
        slicer.mrmlScene.RemoveNode(node)
        for extra in associated:
            if extra and extra.GetScene():
                slicer.mrmlScene.RemoveNode(extra)

    def change_layer(self, index):
        if self.ready and bool(index) != self.reviewed:
            if self.save():
                self.load(self.root, reviewed=bool(index), reuse_study=True)
            else:
                self.layer_actions[int(self.reviewed)].setChecked(True)

    def load(self, dest, reviewed=None, *, reuse_study=False):
        self.exit_comparison()
        self.exit_cutaway()
        if self.cutaway is not None and not reuse_study:
            self.cutaway.release_volume()
        self.leave_branch()
        self.show_subsegmental_review(None)
        self.edit_bar.hide()
        self.free_scroll = False
        self.ready = False
        self.cutaway_button.setEnabled(False)
        self.editor.setActiveEffectByName(None)
        self.editor.setSegmentationNode(None)
        if self.obs:
            self.seg.GetSegmentation().RemoveObserver(self.obs)
            self.obs = None
        self.editor.setMRMLSegmentEditorNode(None)
        retained = [self.vol, self.art] if reuse_study else []
        for n in self.case_nodes:
            if n not in retained:
                self.remove_owned(n)
        self.case_nodes = retained
        self.root = dest
        self.reference = None
        self.reference_button.setChecked(False)
        self.reference_button.setVisible((dest / "reference.nii.gz").exists())
        manifest = json.loads((dest / "candidates.json").read_text())
        if not reuse_study:
            self.load_study(dest)
        review_file = dest / "image-review.json"
        review = None
        if review_file.exists():
            try:
                candidate = json.loads(review_file.read_text())
                filename = candidate["evidenceFile"]
                if (
                    candidate["version"] != 1
                    or candidate["seriesUID"] != self.meta["seriesUID"]
                    or candidate["dimensions"] != list(self.vol.GetImageData().GetDimensions())
                    or not np.allclose(candidate["nativeIJKToRAS"], self.aff, rtol=0, atol=0.001)
                    or Path(filename).name != filename
                    or not (dest / filename).is_file()
                ):
                    raise ValueError("Branch review does not match the study")
                review = candidate
            except (OSError, ValueError, TypeError, KeyError):
                log(traceback.format_exc())
        if reviewed is None:
            reviewed = self.layer_choices.get(dest, True)
        self.reviewed = bool(review) and reviewed
        self.layer_choices[dest] = self.reviewed
        self.layer.menuAction().setVisible(bool(review))
        self.layer_actions[int(self.reviewed)].setChecked(True)
        self.layer_actions[1].setToolTip(review["review"] if review else "")
        if self.reviewed:
            manifest = review
        self.auto = manifest["regions"]
        self.edit_path = dest / (
            f"{Path(manifest['evidenceFile']).name.split('.')[0]}-edits.seg.nrrd"
            if self.reviewed else "review.seg.nrrd"
        )
        self.evidence_volume = slicer.util.loadLabelVolume(
            str(dest / manifest["evidenceFile"]), {"show": False}
        )
        tier_data = slicer.util.arrayFromVolume(self.evidence_volume)
        self.uncertain = self.from_tier(
            tier_data, 2, "Uncertain extent", (1.0, 0.63, 0.20)
        )
        self.case_nodes.append(self.uncertain)
        ud = self.uncertain.GetDisplayNode()
        self.limit_contours_to_axial(ud)
        ud.SetOpacity2DFill(0)
        ud.SetOpacity2DOutline(1)
        ud.SetOpacity3D(0.64)
        ud.SetVisibility(self.reviewed)
        self.uncertain_button.setChecked(self.reviewed)
        self.uncertain_button.setEnabled(bool(np.any(tier_data == 2)))
        red_name = "Suspected clot" if self.reviewed else "CT candidate"
        if self.edit_path.exists():
            self.seg = slicer.util.loadSegmentation(str(self.edit_path))
        else:
            self.seg = self.from_tier(tier_data, 1, red_name, (0.99, 0.105, 0.17))
        del tier_data
        self.remove_owned(self.evidence_volume)
        self.evidence_volume = None
        self.case_nodes.append(self.seg)
        sg = self.seg.GetSegmentation()
        if not sg.GetNumberOfSegments():
            sg.AddEmptySegment("Candidate", "CT candidate")
        self.sid = sg.GetNthSegmentID(0)
        seg = sg.GetSegment(self.sid)
        seg.SetName("Saved review" if self.edit_path.exists() else red_name)
        seg.SetColor(0.99, 0.105, 0.17)
        self.seg.SetReferenceImageGeometryParameterFromVolumeNode(self.vol)
        self.seg.CreateClosedSurfaceRepresentation()
        sd = self.seg.GetDisplayNode()
        self.limit_contours_to_axial(sd)
        sd.SetOpacity2DFill(0.08)
        sd.SetOpacity2DOutline(1)
        sd.SetOpacity3D(1)
        sd.SetSliceIntersectionThickness(2)
        self.overlay.setChecked(True)
        sd.SetVisibility2D(True)
        slicer.util.setSliceViewerLayers(label=None)
        self.ed = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.case_nodes.append(self.ed)
        self.editor.setMRMLSegmentEditorNode(self.ed)
        self.editor.setSegmentationNode(self.seg)
        self.editor.setSourceVolumeNode(self.vol)
        self.editor.setCurrentSegmentID(self.sid)
        self.editor.setupViewObservations()
        self.mark = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        self.case_nodes.append(self.mark)
        self.mark.CreateDefaultDisplayNodes()
        md = self.mark.GetDisplayNode()
        md.SetSelectedColor(0.20, 0.95, 0.79)
        md.SetPointLabelsVisibility(False)
        md.SetPropertiesLabelVisibility(False)
        md.SetGlyphTypeFromString("Cross2D")
        md.SetUseGlyphScale(False)
        md.SetGlyphSize(2.3)
        md.SetVisibility3D(True)
        # The unlabeled marker can use GPU depth testing without a Slicer depth-buffer readback.
        md.SetOccludedVisibility(True)
        md.SetOccludedOpacity(0.0)
        self.mark.SetLocked(True)
        self.branch_outline = None
        self.section_outline = None
        self.setup_anatomy()
        self.regions.clear()
        supported = sum(bool(r["supportedVoxels"]) for r in self.auto)
        self.regions_button.setEnabled(bool(self.auto))
        for i, r in enumerate(self.auto):
            k = r.get("evidenceIJK", r["centerIJK"])[2]
            n = self.meta["slices"][k]["instance"]
            name = r["name"] if self.reviewed else f"{i + 1} · {'Flagged' if r['supportedVoxels'] else 'Uncertain'}"
            menu_action(self.regions, f"{name} · Image {n}", lambda index=i + 1: self.select_region(index))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.meta["slices"]) - 1)
        self.slider.blockSignals(False)
        self.dirty = False
        self.ready = True
        self.cutaway_button.setEnabled(True)
        self.obs = sg.AddObserver(
            slicer.vtkSegmentation.SourceRepresentationModified, self.modified
        )
        self.overview()
        preferred = (
            next(
                (r for r in self.anatomy.guides if r["id"] == self.anatomy.preferred),
                None,
            )
            if self.anatomy
            else None
        )
        if self.reviewed and supported:
            self.select_region(next(i for i, r in enumerate(self.auto, 1) if r["supportedVoxels"]))
        elif preferred:
            self.jump(
                self.anatomy.world(preferred["focusIJK"]), 115, True, landmark=preferred
            )
        elif supported:
            self.select_region(
                next(i for i, r in enumerate(self.auto, 1) if r["supportedVoxels"])
            )
        else:
            self.jump(
                (
                    self.aff
                    @ np.array(
                        [
                            self.vol.GetImageData().GetDimensions()[0] / 2,
                            self.vol.GetImageData().GetDimensions()[1] / 2,
                            len(self.meta["slices"]) // 2,
                            1,
                        ]
                    )
                )[:3],
                260,
                False,
            )
        self.status.setText(
            "Local edits saved"
            if self.edit_path.exists()
            else "Research view"
        )
        log(f"Opened {dest.name}: {len(self.auto)} locations, branch review={self.reviewed}")

    def load_study(self, dest):
        self.meta = json.loads((dest / "source.json").read_text())
        self.aff = np.array(self.meta["nativeIJKToRAS"])
        self.inv = np.linalg.inv(self.aff)
        self.vol = slicer.util.loadVolume(str(dest / "ct.nii.gz"), {"show": False})
        self.case_nodes.append(self.vol)
        self.windowing()
        slicer.util.setSliceViewerLayers(background=self.vol)

        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(dest / "artery-ras.vtp"))
        reader.Update()
        self.art = slicer.modules.models.logic().AddModel(reader.GetOutput())
        self.case_nodes.append(self.art)
        self.artery_picker = ArteryPicker(self.art.GetPolyData())
        display = self.art.GetDisplayNode()
        display.SetScalarVisibility(False)
        display.SetColor(0.20, 0.55, 0.93)
        display.SetOpacity(0.34)
        display.SetAmbient(0.26)
        display.SetSpecular(0.20)
        display.SetVisibility2D(False)
        self.vessel_labels = VesselLabels(
            dest, self.aff, self.vol.GetImageData().GetDimensions()
        )

    def setup_anatomy(self):
        try:
            self.anatomy = ArteryMap(
                self.root, self.meta, self.vol.GetImageData().GetDimensions()
            )
        except (OSError, ValueError, KeyError, TypeError):
            self.anatomy = None
            log(traceback.format_exc())
        self.anatomy_label = None
        self.anatomy_labels = []
        self.map_nodes = []
        self.fork_models = []
        self.active_guide = None
        self.follow_button.setEnabled(False)
        self.unfold_route = None
        self.explore_routes = {}
        self.next_branches = ()
        if self.anatomy:
            if "artery" in self.vessel_labels.masks:
                self.unfold_route = mapped_route(self.anatomy)
                self.explore_routes = explore_routes(self.anatomy, self.vessel_labels)
        self.unfold_button.setVisible("right-basal" in self.explore_routes)
        if self.anatomy:
            for selected in (False, True):
                model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
                self.case_nodes.append(model)
                model.CreateDefaultDisplayNodes()
                model.SetHideFromEditors(True)
                display = model.GetDisplayNode()
                display.SetColor(
                    *((0.45, 1.0, 0.76) if selected else (0.34, 0.83, 0.79))
                )
                display.SetOpacity(0.95 if selected else 0.48)
                display.SetAmbient(0.65)
                display.SetScalarVisibility(False)
                display.SetVisibility2D(False)
                self.map_nodes.append(model)
            self.anatomy_label = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode"
            )
            self.case_nodes.append(self.anatomy_label)
            self.anatomy_label.CreateDefaultDisplayNodes()
            display = self.anatomy_label.GetDisplayNode()
            display.AddViewNodeID(self.three.mrmlViewNode().GetID())
            display.SetVisibility2D(False)
            display.SetPropertiesLabelVisibility(False)
            display.SetPointLabelsVisibility(True)
            display.SetSelectedColor(0.68, 1.0, 0.83)
            display.SetUseGlyphScale(False)
            display.SetGlyphSize(2.3)
            display.SetTextScale(3.25)
            display.SetOccludedVisibility(True)
            display.SetOccludedOpacity(0.8)
            self.anatomy_label.SetLocked(True)
            self.arterial_courses = [
                self.anatomy.paths[r["id"]]
                for r in self.anatomy.guides
                if r.get("vessel", "artery") == "artery"
            ]
            self.draw_courses(self.map_nodes[0], self.arterial_courses, 0.65)
        self.guides_button.setChecked(bool(self.anatomy and self.anatomy.records))
        self.populate_anatomy()

    def highlight_course(self, record, radius):
        key = (record["id"], radius)
        if key != self.highlighted_course:
            self.draw_courses(self.map_nodes[1], [self.anatomy.paths[record["id"]]], radius)
            self.highlighted_course = key

    def draw_courses(self, node, paths, radius):
        points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
        for path in paths:
            if len(path) < 2:
                continue
            lines.InsertNextCell(len(path))
            for xyz in path:
                lines.InsertCellPoint(points.InsertNextPoint(*xyz))
        mesh = vtk.vtkPolyData()
        if points.GetNumberOfPoints():
            source = vtk.vtkPolyData()
            source.SetPoints(points)
            source.SetLines(lines)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(source)
            tube.SetRadius(radius)
            tube.SetNumberOfSides(12)
            tube.CappingOn()
            tube.Update()
            mesh.ShallowCopy(tube.GetOutput())
        node.SetAndObservePolyData(mesh)

    def populate_anatomy(self):
        self.highlighted_course = None
        self.anatomy_menu.clear()
        self.menu_actions = [None]
        self.landmarks_button.setEnabled(self.anatomy is not None)
        if not self.anatomy:
            self.guides_button.setEnabled(False)
            return
        self.guides_button.setEnabled(bool(self.anatomy.records))
        for r in self.anatomy.records:
            menu_action(self.anatomy_menu, r["name"], lambda i=len(self.menu_actions): self.anatomy_action(i))
            self.menu_actions.append(r)
        if self.anatomy.records:
            self.anatomy_menu.addSeparator()
        menu_action(self.anatomy_menu, "Pin current location…", lambda i=len(self.menu_actions): self.anatomy_action(i))
        self.menu_actions.append("add")
        if self.anatomy.pins:
            menu_action(self.anatomy_menu, "Remove a pin…", lambda i=len(self.menu_actions): self.anatomy_action(i))
            self.menu_actions.append("remove")

    def anatomy_action(self, index):
        if not self.ready or not 0 <= index < len(self.menu_actions):
            return
        action = self.menu_actions[index]
        if isinstance(action, dict):
            if self.branch_steps is not None and self.can_follow(action):
                self.start_follow(action, self.anatomy.world(action["focusIJK"]))
                return
            self.guides_button.setChecked(True)
            self.jump(
                self.anatomy.world(action["focusIJK"]), 115, True, landmark=action
            )
            return
        if self.branch_steps is not None:
            self.leave_branch()
            self.show_anatomy(self.focus, self.active_guide)
        if action not in ("add", "remove"):
            return
        dialog = qt.QInputDialog(self.window)
        dialog.setWindowTitle("Artery landmarks")
        if action == "add":
            dialog.setLabelText("Name this location")
        else:
            dialog.setLabelText("Remove which pin?")
            dialog.setComboBoxItems(
                [f"{p['name']}  ·  {i + 1}" for i, p in enumerate(self.anatomy.pins)]
            )
        if not dialog.exec():
            self.show_anatomy(self.focus)
            return
        value = dialog.textValue().strip()
        if not value:
            return
        try:
            if action == "add":
                key = self.anatomy.edit_pin(
                    value, (self.inv @ np.r_[self.focus, 1])[:3]
                )
            else:
                n = int(value.rsplit("·", 1)[1]) - 1
                key = self.anatomy.edit_pin(remove_id=self.anatomy.pins[n]["id"])
            self.populate_anatomy()
            self.guides_button.setChecked(bool(self.anatomy.records))
            self.show_anatomy(
                self.focus, next((r for r in self.anatomy.pins if r["id"] == key), None)
            )
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(str(exc))

    def toggle_guides(self):
        if self.ready:
            self.show_anatomy(self.focus, self.active_guide)

    def can_follow(self, record):
        return (
            record is not None
            and record.get("vessel", "artery") == "artery"
            and "artery" in self.vessel_labels.masks
            and len(record["pathIJK"]) > 1
        )

    def toggle_follow(self):
        if not self.ready:
            return
        if self.follow_button.checked:
            if self.can_follow(self.active_guide):
                self.start_follow(self.active_guide, self.focus)
        else:
            self.exit_unfold()

    def on_mapped_route(self, record):
        return self.unfold_route is not None and record in self.unfold_route[0]

    @property
    def following_route(self):
        return self.active_route is not None

    def start_route_scroll(self):
        guide = self.active_guide if self.ready else None
        if (
            not self.free_scroll
            and self.edit_bar.isHidden()
            and self.can_follow(guide)
            and (self.on_mapped_route(guide) or guide.get("level") == "subsegmental")
        ):
            self.start_follow(guide, self.focus)
            return True
        return False

    def step_position(self, amount):
        if self.in_comparison:
            self.comparison.step(amount)
            return
        if self.in_cutaway:
            self.cutaway.step(amount)
            return
        if self.branch_steps is None:
            self.start_route_scroll()
        self.slider.setValue(self.slider.value + amount)

    def start_unfold(self):
        if not self.ready or not self.explore_routes or self.unfold_panel.distance is not None:
            return
        record = self.followed_guide or self.active_guide
        key = record["id"] if record else None
        if key == "main-split":
            key = "trunk"
        near = self.focus if self.branch_steps is None else self.branch_steps[self.slider.value]
        fallback = key not in self.explore_routes
        if fallback:
            key = "right-basal" if "right-basal" in self.explore_routes else "trunk"
        route, choices = self.explore_routes[key]
        record = next(r for r in route[0] if r["id"] == key)
        if fallback:
            near = self.anatomy.world(record["focusIJK"])
        self.start_follow(record, near, route)
        self.next_branches = choices
        for bar_widget in self.review_bars:
            bar_widget.hide()
        self.unfold_header.show()
        tree = self.art.GetDisplayNode()
        self.tree_opacity = tree.GetOpacity()
        tree.SetOpacity(0.17)
        self.render_explore(animate=True)

    def render_explore(self, *, animate):
        records, steps, starts = self.active_route
        ct = slicer.util.arrayFromVolume(self.vol)
        # The loaded volume may store slices in a different order from the DICOM grid.
        ras_to_ct = vtk.vtkMatrix4x4()
        self.vol.GetRASToIJKMatrix(ras_to_ct)
        distance, intensities, bend = sample_strip(
            ct, slicer.util.arrayFromVTKMatrix(ras_to_ct), self.branch_frames
        )
        display = self.vol.GetDisplayNode()
        end = "Main PA split" if records[-1]["id"] == "trunk" else records[-1]["name"]
        self.unfold_panel.open(
            distance, intensities, bend, starts, display.GetWindow(), display.GetLevel(),
            end, animate,
        )
        self.draw_courses(self.map_nodes[0], [steps], 0.80)
        self.show_branch_step(self.slider.value)

    def choose_explore_branch(self, record):
        if record not in self.unfold_panel.choices:
            return
        route, choices = self.explore_routes[record["id"]]
        steps = route[1]
        if not np.array_equal(steps[:len(self.branch_steps)], self.branch_steps):
            return
        self.branch_frames = course_frames(steps, self.branch_frames)
        self.branch_steps = steps
        self.active_route = route
        self.followed_guide = record
        self.next_branches = choices
        self.branch_wheel = 0.0
        self.slider.setMaximum(len(steps) - 1)
        self.render_explore(animate=False)

    def show_fork_choices(self, choices):
        if choices == self.unfold_panel.choices:
            return
        self.unfold_panel.set_choices(choices)
        while len(self.fork_models) < len(choices):
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            self.case_nodes.append(node)
            self.fork_models.append(node)
            node.CreateDefaultDisplayNodes()
            node.SetHideFromEditors(True)
            display = node.GetDisplayNode()
            display.SetVisibility2D(False)
            display.SetScalarVisibility(False)
            display.SetColor(0.38, 1, 0.80)
            display.SetAmbient(0.9)
        for index, node in enumerate(self.fork_models):
            node.GetDisplayNode().SetVisibility(index < len(choices))
            if index < len(choices):
                path = self.anatomy.paths[choices[index]["id"]]
                self.draw_courses(node, [path], 1.6)

    def exit_unfold(self):
        self.leave_branch()
        self.free_scroll = True
        self.show_anatomy(self.focus, self.active_guide)

    def exit_interaction(self):
        if self.in_comparison:
            self.exit_comparison()
        elif self.in_cutaway:
            if not self.cutaway.close_spotlight():
                self.exit_cutaway()
        elif self.branch_steps is not None:
            self.exit_unfold()
        elif not self.edit_bar.isHidden():
            self.mode("")

    def close_unfold(self):
        if self.unfold_panel.distance is not None:
            self.show_fork_choices(())
            self.unfold_panel.close()
            self.unfold_header.hide()
            for bar_widget in self.review_bars:
                bar_widget.show()
            self.art.GetDisplayNode().SetOpacity(self.tree_opacity)
            self.draw_courses(self.map_nodes[0], self.arterial_courses, 0.65)

    def start_follow(self, record, near, route=None):
        if route is None and self.on_mapped_route(record):
            route = self.unfold_route
        steps = route[1] if route is not None else self.anatomy.course_steps(record)
        if len(steps) < 2:
            self.follow_button.setChecked(self.branch_steps is not None)
            return
        self.close_unfold()
        self.follow_button.setChecked(True)
        self.active_route = route
        self.followed_guide = record
        self.route_section = None
        self.branch_steps = steps
        self.branch_frames = course_frames(steps)
        self.peer_view.setAttribute(qt.Qt.WA_TransparentForMouseEvents, False)
        self.branch_wheel = 0.0
        self.slider.setAccessibleName("Position along branch")
        self.edit_bar.hide()
        self.editor.setActiveEffectByName(None)
        self.guides_button.setChecked(True)
        marker = self.mark.GetDisplayNode()
        marker.SetVisibility2D(False)
        marker.SetVisibility3D(False)
        self.show_branch_marker()
        if self.branch_outline is None:
            self.branch_outline = self.make_branch_outline(self.sn)
            self.section_outline = self.make_branch_outline(self.coronal)
        distances = np.linalg.norm(steps - near, axis=1)
        step_slices = np.rint(steps @ self.inv[2, :3] + self.inv[2, 3])
        near_slice = np.rint(self.inv[2] @ np.r_[near, 1])
        index = int(np.lexsort((distances, np.abs(step_slices - near_slice)))[0])
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(steps) - 1)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.show_anatomy(self.focus, record)
        self.show_branch_step(index, width=90)

    def show_branch_marker(self):
        if self.branch_marker_renderer is not None:
            return
        window = self.view.renderWindow()
        scene = window.GetRenderers().GetFirstRenderer()
        layer = window.GetNumberOfLayers()
        renderer = vtk.vtkRenderer()
        renderer.SetLayer(layer)
        renderer.SetActiveCamera(scene.GetActiveCamera())
        renderer.SetPreserveColorBuffer(True)
        renderer.SetPreserveDepthBuffer(False)
        renderer.InteractiveOff()

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(2.0)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(12)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.96, 1.0, 0.70)
        actor.GetProperty().SetAmbient(0.68)
        actor.GetProperty().SetDiffuse(0.32)
        actor.PickableOff()
        renderer.AddActor(actor)
        window.SetNumberOfLayers(layer + 1)
        window.AddRenderer(renderer)
        self.branch_marker_renderer = renderer
        self.branch_marker = actor

    def make_branch_outline(self, view):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        self.case_nodes.append(node)
        node.SetHideFromEditors(True)
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.AddViewNodeID(view.GetID())
        display.SetVisibility3D(False)
        display.SetVisibility2D(True)
        display.SetSliceDisplayModeToProjection()
        display.SetSliceIntersectionThickness(2)
        display.SetScalarVisibility(False)
        display.SetColor(0.35, 1.0, 0.79)
        return node

    def leave_branch(self):
        if self.branch_steps is None:
            return
        self.branch_render.stop()
        self.close_unfold()
        self.branch_steps = None
        window = self.view.renderWindow()
        window.RemoveRenderer(self.branch_marker_renderer)
        window.SetNumberOfLayers(self.branch_marker_renderer.GetLayer())
        self.branch_marker_renderer = None
        self.branch_marker = None
        self.branch_frames = None
        self.active_route = None
        self.followed_guide = None
        self.next_branches = ()
        self.route_section = None
        self.follow_button.setChecked(False)
        self.unfold_button.setText("Explore artery")
        self.branch_outline.GetDisplayNode().SetVisibility(False)
        self.section_outline.GetDisplayNode().SetVisibility(False)
        self.peer_view.setAttribute(qt.Qt.WA_TransparentForMouseEvents, True)
        self.coronal.SetOrientationToCoronal()
        self.sync_coronal(self.focus, self.sn.GetFieldOfView()[0])
        marker = self.mark.GetDisplayNode()
        marker.SetVisibility2D(self.edit_bar.isHidden())
        marker.SetVisibility3D(True)
        self.mark.SetNthControlPointPosition(0, *self.focus)
        k = int(np.rint((self.inv @ np.r_[self.focus, 1])[2]))
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.meta["slices"]) - 1)
        self.slider.setValue(k)
        self.slider.blockSignals(False)
        self.slider.setToolTip("")
        self.slider.setAccessibleName("Axial slice")
        self.label_slice(k)

    def scroll_branch(self, angle, pixels):
        self.branch_wheel += pixels / 20 if pixels else angle / 120
        steps = int(self.branch_wheel)
        if steps:
            self.branch_wheel -= steps
            self.slider.setValue(self.slider.value + steps)

    def show_branch_step(self, index, width=None):
        point = self.branch_steps[index]
        self.branch_marker.SetPosition(*point)
        k = self.show_slice(point, width, mark=True)
        ijk = (self.inv @ np.r_[self.focus, 1])[:3]
        profile = self.vessel_labels.arterial_profile(ijk)
        self.draw_branch_outline(self.branch_outline, profile, self.aff, k)

        frame = self.branch_frames[index]
        if width:
            self.set_slice_width(self.coronal, 64)
        self.coronal.GetSliceToRAS().DeepCopy(frame.ravel())
        self.coronal.UpdateMatrices()
        spacing = np.linalg.norm(self.aff[:3, :3], axis=0).min() / 2
        radius = int(np.ceil(max(self.coronal.GetFieldOfView()[:2]) / (2 * spacing))) + 2
        grid_to_ras = frame.copy()
        grid_to_ras[:3, :2] *= spacing
        grid_to_ras[:3, 3] -= radius * grid_to_ras[:3, :2].sum(axis=1)
        section_profile = self.vessel_labels.section_profile(self.inv @ grid_to_ras, radius)
        self.draw_branch_outline(self.section_outline, section_profile, grid_to_ras)
        self.slider.setToolTip(f"Branch position {index + 1} / {len(self.branch_steps)}")
        if self.following_route:
            self.show_route_position(index, k)
        if self.unfold_panel.distance is None:
            route = self.explore_routes.get(self.followed_guide["id"])
            next_fork = index == self.slider.maximum and route and route[1]
            button_text = "Choose next branch" if next_fork else "Explore artery"
            if self.unfold_button.text != button_text:
                self.unfold_button.setText(button_text)
            self.hint.setToolTip(
                "Artery outline unavailable at this position."
                if profile is None and section_profile is None
                else "Scroll either CT view to follow the artery. Esc returns to axial scrolling."
            )

    def show_route_position(self, index, k):
        records, _, starts = self.active_route
        section = int(np.searchsorted(starts, index, side="right")) - 1
        record = records[section]
        if section != self.route_section:
            self.route_section = section
            self.active_guide = record
            self.hint.setText(record["name"])
            self.highlight_course(record, 1.45)
            self.show_subsegmental_review(record)
        if self.unfold_panel.distance is not None:
            choices = self.next_branches if index == self.slider.maximum else ()
            self.show_fork_choices(choices)
            self.unfold_location.setText(record["name"])
            self.unfold_panel.set_position(index, section)
            self.unfold_image.setText(f"Image {self.meta['slices'][k]['instance']}")
            self.unfold_image.setToolTip(self.slice_label.toolTip)

    def draw_branch_outline(self, node, profile, grid_to_ras, k=0):
        node.GetDisplayNode().SetVisibility(profile is not None)
        if profile is not None:
            mask, i, j = profile
            mask = np.pad(mask, 1).astype(np.uint8)
            image = vtk.vtkImageData()
            image.SetDimensions(mask.shape[1], mask.shape[0], 1)
            image.SetOrigin(i - 1, j - 1, k)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            vtk_to_numpy(image.GetPointData().GetScalars())[:] = mask.ravel()
            contour = vtk.vtkMarchingSquares()
            contour.SetInputData(image)
            contour.SetValue(0, 0.5)
            transform = vtk.vtkTransform()
            transform.SetMatrix(grid_to_ras.ravel())
            world = vtk.vtkTransformPolyDataFilter()
            world.SetTransform(transform)
            world.SetInputConnection(contour.GetOutputPort())
            world.Update()
            node.SetAndObservePolyData(world.GetOutput())

    def position_subsegmental_review(self, view_width):
        card = self.subsegmental_review
        if not card.isHidden():
            width = min(340, max(1, view_width - 20))
            card.setGeometry(10, 10, width, card.heightForWidth(width))

    def show_subsegmental_review(self, record):
        review = record.get("subsegmentalReview") if record else None
        fields = ("assessment", "finding", "source")
        if not isinstance(review, dict) or not all(
            isinstance(review.get(key), str) and review[key].strip() for key in fields
        ):
            review = None
        if review == self.visible_subsegmental_review:
            return
        self.visible_subsegmental_review = review
        card = self.subsegmental_review
        if not review:
            card.hide()
            return
        assessment, finding, source = (escape(review[key]) for key in fields)
        card.setText(
            f"<b>Subsegmental · {assessment}</b><br>{finding}"
            f'<br><span style="color:#b0c5d3">{source}</span>'
        )
        card.show()
        self.position_subsegmental_review(self.axial_view.width)
        card.raise_()

    def show_anatomy(self, ras, record=None):
        if not self.anatomy:
            self.show_subsegmental_review(None)
            if self.edit_bar.isHidden():
                self.hint.setText("Click the artery tree")
                self.hint.setToolTip("")
            return
        exact = (
            record is not None
            and np.linalg.norm(self.anatomy.world(record["focusIJK"]) - ras) < 1.5
        )
        kind = self.vessel_labels.at((self.inv @ np.r_[ras, 1])[:3])
        if record is None:
            record = (
                self.anatomy.nearby(ras, kind)
                if kind
                else next(
                    (
                        r
                        for r in self.anatomy.pins
                        if np.linalg.norm(self.anatomy.world(r["focusIJK"]) - ras)
                        <= r["radiusMm"]
                    ),
                    None,
                )
            )
        self.active_guide = record
        self.show_subsegmental_review(record if self.edit_bar.isHidden() else None)
        self.follow_button.setEnabled(self.can_follow(record))
        enabled = self.guides_button.checked
        self.map_nodes[0].GetDisplayNode().SetVisibility(enabled)
        self.map_nodes[1].GetDisplayNode().SetVisibility(enabled and bool(record))
        self.anatomy_label.GetDisplayNode().SetVisibility(enabled)
        labels = []
        fork = next((r for r in self.anatomy.guides if r["id"] == "main-split"), None)
        if enabled and fork and fork != record:
            labels.append((tuple(self.anatomy.world(fork["focusIJK"])), "Main PA split"))
        if enabled and record:
            self.map_nodes[1].GetDisplayNode().SetColor(
                *(
                    (0.78, 0.63, 1.0)
                    if record.get("vessel") == "vein"
                    else (0.45, 1.0, 0.76)
                )
            )
            self.highlight_course(record, 1.25)
            if self.branch_steps is None:
                labels.append((tuple(ras), record.get("short", record["name"])))
        self.update_anatomy_labels(labels)
        if self.edit_bar.isHidden():
            self.hint.setToolTip(record["description"] if exact else "")
            if record:
                prefix = "" if exact or self.branch_steps is not None else "Near "
                self.hint.setText(prefix + record["name"])
            elif kind:
                self.hint.setText(f"Pulmonary {kind} estimate")
            else:
                self.hint.setText("Click the artery tree")

    def update_anatomy_labels(self, labels):
        if self.anatomy_labels == labels:
            return
        marks = self.anatomy_label
        with slicer.util.NodeModify(marks):
            if [name for _, name in self.anatomy_labels] == [name for _, name in labels]:
                for index, (position, _) in enumerate(labels):
                    if self.anatomy_labels[index][0] != position:
                        marks.SetNthControlPointPosition(index, *position)
            else:
                marks.RemoveAllControlPoints()
                for position, name in labels:
                    marks.AddControlPoint(vtk.vtkVector3d(*position), name)
        self.anatomy_labels = labels

    def windowing(self, *args):
        if not getattr(self, "vol", None):
            return
        d = self.vol.GetDisplayNode()
        d.AutoWindowLevelOff()
        w, l = (1250, -560) if self.lung_window.checked else (680, 150)
        d.SetWindowLevel(w, l)
        self.unfold_panel.set_window(w, l)

    @property
    def in_cutaway(self):
        return self.cutaway is not None and self.pages.currentWidget() == self.cutaway.widget

    def start_cutaway(self):
        if not self.ready or self.in_cutaway:
            return
        old_status = self.status.text
        qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        try:
            if self.cutaway is None:
                self.cutaway = LungCutaway(self.exit_cutaway, self.start_comparison)
                self.configure_presentation(self.cutaway.scene)
                self.pages.addWidget(self.cutaway.widget)
            self.cutaway.set_volume(self.vol)
            self.cutaway.compare_button.setVisible(self.comparison_manifest.is_file())
            self.pages.setCurrentWidget(self.cutaway.widget)
            self.cutaway.open(self.focus)
            self.status.setText(old_status)
        except Exception:
            self.exit_cutaway()
            log(traceback.format_exc())
            self.status.setText("Could not open Lung Cutaway for this scan.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def exit_cutaway(self):
        if self.in_cutaway:
            self.pages.setCurrentWidget(self.study_page)

    @property
    def comparison_manifest(self):
        return self.root / "spotlight-comparison" / "source.json"

    @property
    def in_comparison(self):
        return self.comparison is not None and self.pages.currentWidget() == self.comparison.widget

    def start_comparison(self):
        if not self.in_cutaway:
            return
        qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
        try:
            if self.comparison is None:
                self.comparison = SpotlightComparison(self.exit_comparison)
                self.pages.addWidget(self.comparison.widget)
            self.comparison.open(self.comparison_manifest, self.meta["seriesUID"])
            self.pages.setCurrentWidget(self.comparison.widget)
        except Exception:
            if self.comparison is not None:
                self.comparison.clear()
            log(traceback.format_exc())
            qt.QMessageBox.warning(self.window, "Spotlight comparison", "Could not open the paired scans for this study.")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def exit_comparison(self):
        if self.in_comparison:
            self.pages.setCurrentWidget(self.cutaway.widget)
            self.comparison.clear()

    def overview(self):
        if not self.ready:
            return
        if self.in_comparison:
            self.comparison.reset()
            return
        if self.in_cutaway:
            self.cutaway.reset_camera()
            return
        b = self.art.GetPolyData().GetBounds()
        center = np.array([(b[i] + b[i + 1]) / 2 for i in (0, 2, 4)])
        span = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        vn = self.three.mrmlViewNode()
        cn = slicer.modules.cameras.logic().GetViewActiveCameraNode(vn)
        cam = cn.GetCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(*(center + np.array([0, span * 2.75, 0])))
        cam.SetViewUp(0, 0, 1)
        cam.ParallelProjectionOn()
        size = self.view.renderWindow().GetSize()
        aspect = max(0.4, size[0] / max(1, size[1]))
        cam.SetParallelScale(max(b[5] - b[4], (b[1] - b[0]) / aspect) * 0.56)
        cn.Modified()
        ren = self.view.renderWindow().GetRenderers().GetFirstRenderer()
        ren.ResetCameraClippingRange()
        self.view.scheduleRender()

    def select_region(self, i):
        if self.ready and 0 < i <= len(self.auto):
            region = self.auto[i - 1]
            if not region["supportedVoxels"]:
                self.uncertain_button.setChecked(True)
                self.toggle_uncertain()
            p = (self.aff @ np.r_[region.get("evidenceIJK", region["centerIJK"]), 1])[
                :3
            ]
            anatomy_id = region.get("anatomyId")
            guide = next(
                (r for r in self.anatomy.guides if r["id"] == anatomy_id), None
            ) if anatomy_id and self.anatomy else None
            if self.can_follow(guide):
                self.start_follow(guide, p)
                return
            self.jump(p, 110, True, landmark=guide)
            if self.reviewed and not self.active_guide and self.edit_bar.isHidden():
                self.hint.setText(region["name"])

    def jump(self, p, width=None, mark=False, landmark=None):
        with slicer.util.RenderBlocker():
            self.leave_branch()
            if mark:
                self.free_scroll = False
            self.show_slice(p, width, mark, landmark)

    def show_slice(self, p, width=None, mark=False, landmark=None):
        self.jumping = True
        ijk = self.inv @ np.r_[p, 1]
        k = int(np.clip(np.rint(ijk[2]), 0, len(self.meta["slices"]) - 1))
        # Snap only the through-plane coordinate to the original acquisition.
        q = (self.aff @ np.array([ijk[0], ijk[1], k, 1.0]))[:3]
        self.focus = q
        if self.branch_steps is None:
            self.slider.blockSignals(True)
            self.slider.value = k
            self.slider.blockSignals(False)
        m = self.sn.GetSliceToRAS()
        axes = self.aff[:3, :3]
        x = axes[:, 0] / np.linalg.norm(axes[:, 0])
        y = -axes[:, 1] / np.linalg.norm(axes[:, 1])
        for j, axis in enumerate((x, y, -np.cross(x, y), q)):
            for i in range(3):
                m.SetElement(i, j, float(axis[i]))
        if width:
            self.set_slice_width(self.sn, width)
        self.sn.UpdateMatrices()
        if self.branch_steps is None:
            self.sync_coronal(q, width)
        if mark:
            marker = q if self.branch_steps is None else p
            if self.mark.GetNumberOfControlPoints():
                self.mark.SetNthControlPointPosition(0, *marker)
            else:
                self.mark.AddControlPoint(vtk.vtkVector3d(*marker), "Source location")
        self.label_slice(k)
        self.window.setWindowTitle("PE Atlas — " + self.meta["title"])
        if self.branch_steps is None:
            self.show_anatomy(q, landmark)
        self.jumping = False
        return k

    def label_slice(self, k):
        ref = self.meta["slices"][k]
        view = " · Cross-section" if self.branch_steps is not None else ""
        self.slice_label.setText(f"Axial image {ref['instance']}{view}")
        self.slice_label.setToolTip(
            f"Series {self.meta['series']}\nSource DICOM SOP Instance UID\n{ref['uid']}\n"
            "Scroll the CT or press A / D to move."
        )

    @staticmethod
    def set_slice_width(node, width):
        fov = node.GetFieldOfView()
        node.SetFieldOfView(width, width * fov[1] / max(fov[0], 0.01), fov[2])

    def sync_coronal(self, point, width=None):
        if width:
            self.set_slice_width(self.coronal, width)
        self.coronal.JumpSliceByCentering(*point)

    def slice_moved(self, *unused):
        if not self.ready or self.jumping:
            return
        m = self.sn.GetSliceToRAS()
        point = np.array([m.GetElement(i, 3) for i in range(3)])
        k = int(
            np.clip(
                np.rint((self.inv @ np.r_[point, 1])[2]),
                0,
                len(self.meta["slices"]) - 1,
            )
        )
        if self.branch_steps is not None:
            if k != int(np.rint((self.inv @ np.r_[self.focus, 1])[2])):
                self.show_branch_step(self.slider.value)
            return
        if self.slider.value != k:
            with slicer.util.RenderBlocker():
                self.focus = point
                self.sync_coronal(point)
                self.show_anatomy(point)
                self.slider.blockSignals(True)
                self.slider.value = k
                self.slider.blockSignals(False)
                self.label_slice(k)

    def slide(self, k):
        if not self.ready:
            return
        if self.branch_steps is not None:
            # Commit the linked CT planes and marker before any view draws the new step.
            views = self.branch_views()
            enabled = [view for view in views if view.renderEnabled]
            for view in enabled:
                view.setRenderEnabled(False)
            try:
                self.show_branch_step(k)
            finally:
                for view in enabled:
                    view.setRenderEnabled(True)
            if not self.branch_render.isActive():
                self.branch_render.start(0)
            return
        ij = self.inv @ np.r_[self.focus, 1]
        ij[2] = k
        self.jump((self.aff @ ij)[:3])

    def branch_views(self):
        return self.axial_view, self.peer_view, self.view

    def render_branch_views(self):
        for view in self.branch_views():
            view.forceRender()

    def press_axial(self, *unused):
        self.axial_press = (
            self.ct_interactor.GetEventPosition()
            if self.ready and self.edit_bar.isHidden() and self.branch_steps is None
            else None
        )

    def release_axial(self, *unused):
        start, self.axial_press = self.axial_press, None
        if start is None or not self.ready or not self.edit_bar.isHidden():
            return
        x, y = self.ct_interactor.GetEventPosition()
        if max(abs(x - start[0]), abs(y - start[1])) > 3:
            return
        w, h = self.slice.sliceView().renderWindow().GetSize()
        dims = self.sn.GetDimensions()
        ras = self.sn.GetXYToRAS().MultiplyPoint(
            (x * dims[0] / w, y * dims[1] / h, 0, 1)
        )
        ijk = (self.inv @ ras)[:3]
        if np.any(ijk < 0) or np.any(
            ijk > np.asarray(self.vol.GetImageData().GetDimensions()) - 1
        ):
            return
        if self.vessel_labels.at(ijk) != "artery" or not self.pick_subsegmental(ras[:3], surface=True):
            self.jump(ras[:3], mark=True)

    def press3(self, *args):
        if self.ready:
            self.press3d = self.interactor.GetEventPosition()

    def release3(self, *args):
        if self.press3d is None:
            return
        xy = self.interactor.GetEventPosition()
        start = self.press3d
        self.press3d = None
        if max(abs(a - b) for a, b in zip(start, xy)) <= 4:
            qt.QTimer.singleShot(0, lambda: self.pick(*xy))

    def pick_subsegmental(self, point, *, surface=False):
        if not self.anatomy or not self.edit_bar.isHidden():
            return False
        point = np.asarray(point)
        if surface:
            hit = self.inv[:3, :3] @ point + self.inv[:3, 3]
            if self.vessel_labels.at(hit) == "vein":
                return False

            def supports(center):
                start = self.inv[:3, :3] @ center + self.inv[:3, 3]
                return (
                    self.vessel_labels.at(start) == "artery"
                    and self.vessel_labels.at((start + hit) / 2) == "artery"
                )

            match = self.anatomy.subsegmental_pick(point, margin=1.35, supports=supports)
        else:
            match = self.anatomy.subsegmental_pick(point)
        if match is None:
            return False
        record, center = match
        if self.unfold_panel.distance is not None:
            if record in self.unfold_panel.choices:
                self.choose_explore_branch(record)
            elif record in self.active_route[0]:
                distances = np.linalg.norm(self.branch_steps - center, axis=1)
                self.slider.setValue(int(np.argmin(distances)))
            elif record["id"] in self.explore_routes:
                self.jump(center, mark=True, landmark=record)
                self.start_unfold()
            else:
                return False
        else:
            if not self.can_follow(record):
                return False
            self.start_follow(record, center)
        return True

    def pick_guide(self, picker, renderer, manager, x, y):
        picker.InitializePickList()
        for node in self.map_nodes:
            actor = manager.GetActorByID(node.GetDisplayNode().GetID())
            if actor and actor.GetVisibility():
                picker.AddPickList(actor)
        if picker.Pick(x, y, 0, renderer):
            return np.asarray(picker.GetPickPosition())
        return None

    def pick(self, x, y):
        if not self.ready:
            return False
        ren = self.view.renderWindow().GetRenderers().GetFirstRenderer()
        dm = self.view.displayableManagerByClassName("vtkMRMLModelDisplayableManager")
        exploring = self.unfold_panel.distance is not None
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        if exploring:
            choices = self.unfold_panel.choices
            if choices:
                actors = {}
                for record, node in zip(choices, self.fork_models):
                    actor = dm.GetActorByID(node.GetDisplayNode().GetID())
                    if actor:
                        actors[actor] = record
                        picker.AddPickList(actor)
                if picker.Pick(x, y, 0, ren):
                    if np.linalg.norm(self.branch_steps[-1] - picker.GetPickPosition()) > 3:
                        self.choose_explore_branch(actors[picker.GetActor()])
                        return True
                    if self.pick_subsegmental(picker.GetPickPosition()):
                        return True
        guide_point = self.pick_guide(picker, ren, dm, x, y)
        if guide_point is not None:
            if exploring:
                distances = np.linalg.norm(self.branch_steps - guide_point, axis=1)
                self.slider.setValue(int(np.argmin(distances)))
                return True
            if self.pick_subsegmental(guide_point):
                return True
        actor = dm.GetActorByID(self.art.GetDisplayNode().GetID())
        if not actor:
            return False
        p = self.artery_picker.pick(ren, actor, x, y)
        if p is not None and self.pick_subsegmental(p, surface=True):
            return True
        if exploring:
            return False
        # Visible candidate overlays may extend beyond the artery surface.
        overlay_picker = vtk.vtkCellPicker()
        overlay_picker.SetTolerance(0.0006)
        overlay_picker.PickFromListOn()
        props = ren.GetActors()
        props.InitTraversal()
        for i in range(props.GetNumberOfItems()):
            a = props.GetNextActor()
            if a and a != actor and a.GetVisibility() and a.GetPickable():
                color = a.GetProperty().GetColor()
                is_red = color[0] > 0.7 and color[1] < 0.35 and color[2] < 0.36
                is_amber = color[0] > 0.8 and 0.5 < color[1] < 0.72 and color[2] < 0.32
                if is_red or (self.uncertain_button.checked and is_amber):
                    overlay_picker.AddPickList(a)
        overlay_hit = bool(overlay_picker.Pick(x, y, 0, ren))
        rp = np.array(overlay_picker.GetPickPosition())
        if overlay_hit and (
            p is None
            or np.linalg.norm(rp - p) < 10
            or np.linalg.norm(rp - ren.GetActiveCamera().GetPosition())
            < np.linalg.norm(p - ren.GetActiveCamera().GetPosition())
        ):
            p = rp
        elif p is None:
            return False
        self.jump(p, 110, True)
        return True

    def toggle_overlay(self):
        if self.ready:
            self.seg.GetDisplayNode().SetVisibility2D(self.overlay.checked)
            self.toggle_uncertain()

    def mode(self, name):
        if not self.ready:
            return
        self.leave_branch()
        self.edit_bar.setVisible(bool(name))
        if name:
            self.buttons[name].setChecked(True)
            self.overlay.setChecked(True)
            self.toggle_overlay()
        self.editor.setActiveEffectByName(name or None)
        self.set_brush()
        self.mark.GetDisplayNode().SetVisibility2D(not bool(name))
        if name:
            self.show_subsegmental_review(None)
            self.hint.setText("Editing outlines")
            self.hint.setToolTip("Draw on the axial CT. Edits save locally.")
        else:
            self.show_anatomy(self.focus, self.active_guide)

    def set_brush(self, *args):
        if not self.ready:
            return
        effect = self.editor.activeEffect()
        if effect:
            effect.setCommonParameter("BrushDiameterIsRelative", 0)
            effect.setCommonParameter("BrushAbsoluteDiameter", self.brush.value)
            effect.setCommonParameter("BrushSphere", 0)
            effect.setCommonParameter("EditIn3DViews", 0)

    def modified(self, *args):
        if self.ready:
            self.dirty = True
            self.status.setText("Saving edits…")
            self.timer.start()

    def save(self):
        self.timer.stop()
        if not self.dirty:
            return True
        try:
            tmp = self.root / "review-writing.seg.nrrd"
            if not slicer.util.saveNode(self.seg, str(tmp)):
                raise IOError("Could not write correction")
            os.replace(tmp, self.edit_path)
            self.dirty = False
            self.status.setText("Local edits saved")
            return True
        except Exception:
            log(traceback.format_exc())
            self.status.setText("Save failed — edits remain open")
            return False


try:
    log("Starting PE workspace")
    atlas = Workstation()
    import __main__

    __main__.peAtlas = atlas

except Exception:
    log(traceback.format_exc())
    qt.QMessageBox.critical(None, "PE Atlas", traceback.format_exc())
