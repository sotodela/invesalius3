import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

from invesalius.data.targeting import extract_targeting_roi_fields, map_targets_to_mesh, recalculate_targeting_field


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


class DirectRecalculationTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock()
        self.api.set_dIperdt.return_value = True
        self.currents = [1., 1.]
        self.calls = []
        def set_currents(dIperdt):
            self.currents = list(dIperdt)
            self.calls.append(list(dIperdt))
            return True
        self.api.set_dIperdt.side_effect = set_currents
        self.basis = np.array([[[1., 0, 0], [2, 0, 0], [3, 0, 0]],
                               [[0., 1, 0], [0, 2, 0], [0, 3, 0]]])
        def calculate(**kwargs):
            vectors = self.basis * np.asarray(self.currents)[:, None, None]
            return SimpleNamespace(**{f"column{i+1}": vectors[:, :, i].reshape(-1).tolist() for i in range(3)})
        self.api.update_efield_vectorROI.side_effect = calculate
        self.area = {"dIperdt": [1, 1], "efields": []}
        for pose in range(2):
            for channel, field in enumerate(self.basis):
                self.area["efields"].append({
                    "coil_position_index": pose, "channel_index": channel,
                    "position": [pose, 0, 0], "orientation": [0, 0, 0], "T_rot": np.eye(3),
                    "roi_indices": [2, 0],
                    "efield": {f"column{i+1}": field[[2, 0], i].tolist() for i in range(3)},
                })

    def test_direct_currents_group_by_pose_and_restore_previous_settings(self):
        result = recalculate_targeting_field(self.area, [2, -1, 0, 3], self.api, 3, [7, 8])
        self.assertEqual(self.calls, [[2, -1], [0, 3], [7, 8]])
        self.assertEqual(self.api.update_efield_vectorROI.call_count, 2)
        self.assertEqual(self.api.update_efield_vectorROI.call_args.kwargs["id_list"], [2, 0])
        np.testing.assert_allclose(result["vectors"], [[6, 6, 0], [2, 2, 0]])
        self.assertTrue(result["validation"]["matches_reconstruction"])
        self.assertAlmostEqual(result["validation"]["relative_vector_error"], 0)

    def test_backend_discrepancy_is_reported_and_direct_field_is_returned(self):
        self.api.update_efield_vectorROI.side_effect = lambda **kw: SimpleNamespace(
            column1=[0]*6, column2=[0]*6, column3=[0]*6)
        result = recalculate_targeting_field(self.area, [2, -1, 0, 3], self.api, 3, [7, 8])
        self.assertFalse(result["validation"]["matches_reconstruction"])
        self.assertAlmostEqual(result["validation"]["relative_vector_error"], 1)
        np.testing.assert_allclose(result["norms"], [0, 0])

    def test_service_failure_restores_current_settings(self):
        self.api.update_efield_vectorROI.side_effect = RuntimeError("Service failed")
        with self.assertRaisesRegex(RuntimeError, "Service failed"):
            recalculate_targeting_field(self.area, [2, -1, 0, 3], self.api, 3, [7, 8])
        self.assertEqual(self.currents, [7, 8])

    def test_incomplete_channels_and_wrong_weight_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid targeting"):
            recalculate_targeting_field(self.area, [1], self.api, 3, [7, 8])
        self.area["efields"].pop()
        with self.assertRaisesRegex(ValueError, "Missing channel"):
            recalculate_targeting_field(self.area, [1]*3, self.api, 3, [7, 8])
        self.api.update_efield_vectorROI.assert_not_called()

    def test_zero_fields_have_no_misleading_relative_error(self):
        result = recalculate_targeting_field(self.area, [0]*4, self.api, 3, [7, 8])
        self.assertIsNone(result["validation"]["relative_vector_error"])
        self.assertTrue(result["validation"]["matches_reconstruction"])


if __name__ == "__main__":
    unittest.main()
