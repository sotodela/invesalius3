import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkCellLocator, vtkPolyData, vtkTriangle

from invesalius.navigation.efield_navigation import EFieldNavigation
from invesalius.data.visualization.efield_visualizer import EFieldVisualizer


class TargetingArrowTests(unittest.TestCase):
    def setUp(self):
        points = vtkPoints()
        for point in [(0,0,0),(10,0,0),(0,10,0),(10,10,0)]:
            points.InsertNextPoint(*point)
        triangles = vtkCellArray()
        for ids in [(0,1,2),(1,3,2)]:
            cell = vtkTriangle()
            for index, point_id in enumerate(ids):
                cell.GetPointIds().SetId(index, point_id)
            triangles.InsertNextCell(cell)
        mesh = vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(triangles)
        self.efield = EFieldNavigation.__new__(EFieldNavigation)
        self.efield.efield_mesh = mesh
        self.efield.radius_list = vtkIdList()
        self.efield.efield_painting_token = object()
        self.efield.locator_efield_cell = vtkCellLocator()
        self.efield.locator_efield_cell.SetDataSet(mesh)
        self.efield.locator_efield_cell.BuildLocator()
        visualizer = EFieldVisualizer.__new__(EFieldVisualizer)
        self.efield.visualizer = visualizer
        visualizer.efield = self.efield
        visualizer.ren = Mock()
        visualizer.actor_factory = Mock()
        visualizer.DrawVectors = Mock(return_value=Mock())
        visualizer.OnUpdateEfieldvis = Mock()
        for name in ['max_efield_vector','ball_max_vector','GoGEfieldVector','ball_GoGEfieldVector']:
            setattr(visualizer, name, None)

    def show(self, norms, ids, vectors=None):
        with patch.dict(EFieldNavigation.GetEfieldMaxMin.__globals__, {
            'wx': SimpleNamespace(CallAfter=lambda method,*args: method(*args)), 'Publisher':Mock(),
        }):
            self.efield.ShowTargetingEfieldResult(norms, ids, efield_vectors=vectors)

    def test_maximum_arrow_moves_and_uses_each_selected_results_vector(self):
        self.show([1,5,2], [3,1,0], [[1,0,0],[0,5,0],[0,0,2]])
        self.efield.visualizer.MaxEfieldActor()
        self.assertEqual(self.efield.Idmax, 1)
        np.testing.assert_allclose(self.efield.position_max, [10,0,0])
        np.testing.assert_allclose(self.efield.visualizer.DrawVectors.call_args.args[1], [0,5,0])
        self.show([6,1,2], [3,1,0], [[6,0,0],[0,1,0],[0,0,2]])
        self.efield.visualizer.MaxEfieldActor()
        np.testing.assert_allclose(self.efield.position_max, [10,10,0])
        np.testing.assert_allclose(self.efield.max_efield_array, [6,0,0])

    def test_full_reordered_roi_keeps_field_vectors_paired_with_mesh_ids(self):
        self.show([6,2,1,0], [3,1,2,0], [[6,0,0],[0,2,0],[0,0,1],[0,0,0]])
        self.assertEqual(self.efield.Idmax, 3)
        self.assertEqual(self.efield.e_field_col1, [6,0,0,0])
        self.assertEqual(self.efield.e_field_norms, [6,2,1,0])

    def test_centroid_uses_weights_at_selected_global_point_ids(self):
        self.efield.Id_list = [0,1,2,3]
        self.efield.e_field_norms = [100,2,0,6]
        result = self.efield.FindCenterofGravity([1,3], [[10,0,0],[10,10,0]])
        np.testing.assert_allclose(result, [10,7.5,0])

    def test_centroid_actor_recalculates_using_current_thresholded_field(self):
        self.show([1,5,2], [3,1,0], [[1,0,0],[0,5,0],[0,0,2]])
        self.efield.cell_id_indexes_above_threshold = [1]
        self.efield.positions_above_threshold = [[10,0,0]]
        self.efield.visualizer.CoGEfieldActor()
        np.testing.assert_allclose(self.efield.center_gravity_position, [10,0,0])
        self.show([6,1,2], [3,1,0], [[6,0,0],[0,1,0],[0,0,2]])
        self.efield.cell_id_indexes_above_threshold = [3]
        self.efield.positions_above_threshold = [[10,10,0]]
        self.efield.visualizer.CoGEfieldActor()
        np.testing.assert_allclose(self.efield.center_gravity_position, [10,10,0])

    def test_missing_vectors_cannot_reuse_previous_arrow(self):
        self.efield.max_efield_array = [99,99,99]
        self.show([1,5], [0,1])
        self.efield.visualizer.MaxEfieldActor()
        self.assertIsNone(self.efield.max_efield_array)
        self.assertIsNone(self.efield.position_max)
        self.efield.visualizer.DrawVectors.assert_not_called()

    def test_zero_field_has_no_arbitrary_maximum_arrow(self):
        self.show([0,0], [0,1], [[0,0,0],[0,0,0]])
        self.efield.visualizer.MaxEfieldActor()
        self.assertIsNone(self.efield.position_max)
        self.efield.visualizer.DrawVectors.assert_not_called()


if __name__ == '__main__':
    unittest.main()
