"""Narrow artery ray picks to nearby triangles before VTK tests the surface."""

import numpy as np
import vtk


def world_point(renderer, x, y, z):
    renderer.SetDisplayPoint(x, y, z)
    renderer.DisplayToWorld()
    point = np.asarray(renderer.GetWorldPoint())
    return point[:3] / point[3]


class ArteryPicker:
    def __init__(self, mesh):
        self.mesh = mesh
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0008)
        self.picker.PickFromListOn()
        self.locator = None
        if mesh.GetNumberOfPolys() != mesh.GetNumberOfCells() or mesh.GetPolys().IsHomogeneous() != 3:
            return

        self.locator = vtk.vtkStaticCellLocator()
        self.locator.SetDataSet(mesh)
        self.locator.BuildLocator()
        self.cells_in_box = vtk.vtkIdList()
        self.candidates = vtk.vtkIdList()
        self.extract = vtk.vtkExtractCells()
        self.extract.SetInputData(mesh)
        self.extract.SetAssumeSortedAndUniqueIds(True)
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(self.extract.GetOutputPort())
        self.pick_actor = vtk.vtkActor()
        self.pick_actor.SetMapper(mapper)

    def pick(self, renderer, actor, x, y):
        camera = renderer.GetActiveCamera()
        mapper = actor.GetMapper()
        if (
            self.locator is not None
            and actor.GetVisibility()
            and actor.GetPickable()
            and actor.GetProperty().GetOpacity() > 0
            and actor.GetMatrix().IsIdentity()
            and mapper.GetInput() == self.mesh
            and not mapper.GetNumberOfClippingPlanes()
            and camera.GetParallelProjection()
            and not camera.GetUseExplicitProjectionTransformMatrix()
            and camera.GetModelTransformMatrix().IsIdentity()
            and camera.GetViewShear()[:2] == (0.0, 0.0)
        ):
            if not self.set_candidates(renderer, x, y):
                return None
            actor = self.pick_actor

        self.picker.InitializePickList()
        self.picker.AddPickList(actor)
        if self.picker.Pick(x, y, 0, renderer):
            return np.asarray(self.picker.GetPickPosition())
        return None

    def set_candidates(self, renderer, x, y):
        start = world_point(renderer, x, y, 0)
        end = world_point(renderer, x, y, 1)
        width, height = renderer.GetRenderWindow().GetSize()
        left, bottom, right, top = renderer.GetViewport()
        lower = world_point(renderer, width * left, height * bottom, 0.5)
        upper = world_point(renderer, width * right, height * top, 0.5)
        padding = np.linalg.norm(upper - lower) * self.picker.GetTolerance() + 1e-5

        # The expanded boxes cover the entire ray within VTK's pick tolerance.
        points = np.linspace(start, end, 13)
        selected = set()
        for first, last in zip(points[:-1], points[1:]):
            bounds = np.column_stack((np.minimum(first, last) - padding, np.maximum(first, last) + padding))
            self.locator.FindCellsWithinBounds(bounds.ravel(), self.cells_in_box)
            selected.update(self.cells_in_box.GetId(i) for i in range(self.cells_in_box.GetNumberOfIds()))
        if not selected:
            return False

        # VTK's tie-breaking for grazing hits depends on the original cell order.
        self.candidates.Reset()
        for ident in sorted(selected):
            self.candidates.InsertNextId(ident)
        self.extract.SetCellList(self.candidates)
        self.extract.Update()
        return True
