import unittest
from types import SimpleNamespace

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from invesalius.data.targeting import extract_targeting_roi_fields, map_targets_to_mesh


class TargetMappingTests(unittest.TestCase):
    def setUp(self):
        points = vtkPoints()
        for point in [(0, 0, 0), (10, 20, 30), (40, 50, 60)]:
            points.InsertNextPoint(*point)
        self.mesh = vtkPolyData()
        self.mesh.SetPoints(points)

    def test_marker_y_is_flipped_and_global_id_preserved(self):
        target = {"position": [10, -20, 30], "coordinate_frame": "invesalius"}
        result = map_targets_to_mesh([target], self.mesh, [2, 1])
        self.assertEqual(result[0]["point_index"], 1)
        self.assertEqual(result[0]["mesh_position"], [10, 20, 30])
        self.assertEqual(result[0]["mesh_distance_mm"], 0)
        self.assertNotIn("point_index", target)

    def test_mesh_frame_and_nearest_point(self):
        target = {"position": [40.1, 50, 60], "coordinate_frame": "efield_mesh"}
        result = map_targets_to_mesh([target], self.mesh, [2, 0])
        self.assertEqual(result[0]["point_index"], 2)
        self.assertAlmostEqual(result[0]["mesh_distance_mm"], 0.1)

    def test_outside_roi_is_rejected_instead_of_relocated(self):
        with self.assertRaisesRegex(ValueError, "outside the selected ROI"):
            map_targets_to_mesh([{"position": [10, -20, 30]}], self.mesh, [0, 2])

    def test_invalid_position_and_frame_are_rejected(self):
        for target in [{"position": [float("nan"), 0, 0]}, {"position": [0, 0]},
                       {"position": [0, 0, 0], "coordinate_frame": "unknown"}]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                map_targets_to_mesh([target], self.mesh, [0, 1, 2])

    def test_invalid_or_duplicate_roi_is_rejected(self):
        for roi in [[], [-1], [3], [1, 1]]:
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                map_targets_to_mesh([{"position": [0, 0, 0]}], self.mesh, roi)

    def test_missing_mesh_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Load the E-field"):
            map_targets_to_mesh([{"position": [0, 0, 0]}], None, [0])


class FieldExtractionTests(unittest.TestCase):
    def test_full_mesh_is_subset_in_roi_order(self):
        field = SimpleNamespace(column1=[1, 2, 3], column2=[0, 0, 0], column3=[0, 0, 0])
        result = extract_targeting_roi_fields(field, [2, 0], 3, 1)
        self.assertEqual(result[0]["column1"], [3, 1])
        self.assertEqual(result[0]["enorm"], [3, 1])

    def test_mtms_channels_remain_separate_in_roi_order(self):
        field = SimpleNamespace(column1=[1, 2, -1, 3], column2=[0]*4, column3=[0]*4)
        result = extract_targeting_roi_fields(field, [1, 0], 2, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["column1"], [2, 1])
        self.assertEqual(result[1]["column1"], [3, -1])
        self.assertEqual(result[1]["enorm"], [3, 1])

    def test_wrong_field_size_or_invalid_roi_is_rejected(self):
        field = SimpleNamespace(column1=[1], column2=[0], column3=[0])
        for ids, size, coils in [([0], 2, 1), ([1], 1, 1), ([0], 1, 2)]:
            with self.subTest(ids=ids, size=size, coils=coils), self.assertRaises(ValueError):
                extract_targeting_roi_fields(field, ids, size, coils)


if __name__ == "__main__":
    unittest.main()
