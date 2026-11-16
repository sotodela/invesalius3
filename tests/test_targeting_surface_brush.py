import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkQuad, vtkPointLocator

from invesalius.data.surface_brush import SurfaceDistanceBrush, picked_surface_point_id
from invesalius.data.visualization.efield_visualizer import EFieldVisualizer
from invesalius.navigation.efield_navigation import EFieldNavigation


def folded_ribbon():
    points = vtkPoints()
    # A hairpin surface: the first and final banks are 0.1 mm apart in space,
    # but reaching the final bank along the surface requires a 4.1 mm journey.
    for x, z in [(0,0),(1,0),(2,0),(2,.1),(1,.1),(0,.1)]:
        points.InsertNextPoint(x, 0, z)
        points.InsertNextPoint(x, .1, z)
    cells = vtkCellArray()
    for i in range(5):
        quad = vtkQuad()
        for j, point_id in enumerate([2*i, 2*i+1, 2*i+3, 2*i+2]):
            quad.GetPointIds().SetId(j, point_id)
        cells.InsertNextCell(quad)
    mesh = vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


class SurfaceBrushTests(unittest.TestCase):
    def test_does_not_jump_across_nearby_fold(self):
        brush = SurfaceDistanceBrush(folded_ribbon())
        selected = brush.points_within_radius(0, 1.15)
        self.assertEqual(selected, [0,1,2,3])
        self.assertNotIn(10, selected)
        self.assertIn(10, brush.points_within_radius(0, 4.3))

    def test_disconnected_nearby_vertex_is_never_selected(self):
        mesh = folded_ribbon()
        mesh.GetPoints().InsertNextPoint(0,0,.01)
        brush = SurfaceDistanceBrush(mesh)
        self.assertNotIn(12, brush.points_within_radius(0, 100))

    def test_zero_radius_and_invalid_arguments(self):
        brush = SurfaceDistanceBrush(folded_ribbon())
        self.assertEqual(brush.points_within_radius(0, 0), [0])
        for seed, radius in [(0,-1),(0,float('nan')),(0,float('inf')),(-1,1),(12,1)]:
            with self.subTest(seed=seed, radius=radius), self.assertRaises(ValueError):
                brush.points_within_radius(seed, radius)

    def test_cache_ignores_colors_and_rebuilds_when_geometry_changes(self):
        mesh = folded_ribbon()
        brush = SurfaceDistanceBrush(mesh)
        brush.points_within_radius(0,1)
        graph = brush._adjacency
        colors = vtkUnsignedCharArray()
        mesh.GetPointData().SetScalars(colors)
        brush.points_within_radius(0,1)
        self.assertIs(brush._adjacency, graph)
        mesh.GetPoints().SetPoint(2,3,0,0)
        mesh.GetPoints().Modified()
        brush.points_within_radius(0,1)
        self.assertIsNot(brush._adjacency, graph)

    def test_add_and_erase_use_same_patch_and_preserve_roi(self):
        mesh = folded_ribbon()
        state = SimpleNamespace(
            efield_mesh=mesh, avoidance_surface_brush=None, efield_selection_kind='avoidance',
            efield_avoidance_action='add', efield_targeting_area_radius=1.15,
            efield_avoidance_ids={10}, efield_targeting_area_ids={5},
            _UpdateEfieldTargetingAreaHighlight=Mock(),
        )
        publisher = Mock()
        with patch.dict(EFieldNavigation.ApplyEfieldAreaBrush.__globals__, {'Publisher':publisher}):
            EFieldNavigation.ApplyEfieldAreaBrush(state,0)
            self.assertEqual(state.efield_avoidance_ids, {0,1,2,3,10})
            state.efield_avoidance_action='erase'
            EFieldNavigation.ApplyEfieldAreaBrush(state,0)
        self.assertEqual(state.efield_avoidance_ids, {10})
        self.assertEqual(state.efield_targeting_area_ids, {5})
        self.assertEqual(state._UpdateEfieldTargetingAreaHighlight.call_count, 2)
        publisher.sendMessage.assert_called_with('Efield avoidance area selected', selected_ids=[10])

    def test_mesh_replacement_discards_old_brush(self):
        old_mesh = folded_ribbon()
        new_mesh = folded_ribbon()
        state = SimpleNamespace(
            efield_mesh=new_mesh, avoidance_surface_brush=SurfaceDistanceBrush(old_mesh),
            efield_selection_kind='avoidance', efield_avoidance_action='add',
            efield_targeting_area_radius=0, efield_avoidance_ids=set(),
            _UpdateEfieldTargetingAreaHighlight=Mock(),
        )
        with patch.dict(EFieldNavigation.ApplyEfieldAreaBrush.__globals__, {'Publisher':Mock()}):
            EFieldNavigation.ApplyEfieldAreaBrush(state,0)
        self.assertIs(state.avoidance_surface_brush.mesh, new_mesh)
        self.assertEqual(state.efield_avoidance_ids, {0})


    def test_click_seed_stays_on_picked_face_when_other_vertices_are_closer(self):
        mesh = folded_ribbon()
        mesh.GetPoints().InsertNextPoint(.5,.05,.01)
        locator = vtkPointLocator()
        locator.SetDataSet(mesh)
        locator.BuildLocator()
        self.assertEqual(locator.FindClosestPoint([.5,.05,0]), 12)
        self.assertIn(picked_surface_point_id(mesh, 0, [.5,.05,0], locator), [0,1,2,3])

    def test_erased_highlight_restores_underlying_field_color(self):
        mesh = folded_ribbon()
        state = SimpleNamespace(efield_mesh=mesh, efield_painting_token=object(),
                                Id_list=list(range(12)), e_field_norms=[.5]*12,
                                efield_selection_kind='avoidance', efield_avoidance_ids={0},
                                efield_targeting_area_ids={0})
        visualizer = EFieldVisualizer.__new__(EFieldVisualizer)
        state.visualizer = visualizer
        visualizer.efield = state
        visualizer.colors_init = vtkUnsignedCharArray()
        visualizer.efield_lut = vtkLookupTable()
        visualizer.efield_lut.SetTableRange(0,1)
        visualizer.efield_lut.Build()
        visualizer.RecolorEfieldActor = Mock()
        visualizer.Refresh = Mock()
        visualizer.RestoreEfieldBaseColors()
        original_color = visualizer.colors_init.GetTuple(0)
        EFieldNavigation._UpdateEfieldTargetingAreaHighlight(state)
        self.assertEqual(visualizer.colors_init.GetTuple(0), (255,90,40))
        state.efield_avoidance_ids.clear()
        EFieldNavigation._UpdateEfieldTargetingAreaHighlight(state)
        self.assertEqual(visualizer.colors_init.GetTuple(0), original_color)


if __name__ == '__main__':
    unittest.main()
