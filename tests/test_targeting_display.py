"""Navigation display ownership and stale-result regression checks."""

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vtkmodules.vtkCommonCore import vtkIdList

from invesalius.data.e_field import EFieldCalculationThread
from invesalius.navigation.navigation import Navigation
from invesalius.navigation.efield_navigation import EFieldNavigation


class TargetingDisplayTests(unittest.TestCase):
    def test_mode_switch_invalidates_and_drains_pending_live_results(self):
        state = SimpleNamespace(
            targeting_efield_active=False, e_field_revision=3,
            efield_queue=Mock(), e_field_norms_queue=Mock(), e_field_IDs_queue=Mock(),
        )
        state.MarkEfieldParametersChanged = lambda: setattr(state, "e_field_revision", state.e_field_revision + 1)
        with patch.dict(Navigation.SetTargetingEfieldMode.__globals__, {"Publisher": Mock()}):
            Navigation.SetTargetingEfieldMode(state, True)
            self.assertTrue(state.targeting_efield_active)
            Navigation.SetTargetingEfieldMode(state, False)
        self.assertFalse(state.targeting_efield_active)
        self.assertEqual(state.e_field_revision, 5)
        for pending in (state.efield_queue, state.e_field_norms_queue, state.e_field_IDs_queue):
            self.assertEqual(pending.clear.call_count, 2)

    def test_live_values_and_roi_updates_cannot_mutate_held_result(self):
        state = SimpleNamespace(targeting_display_active=True, efield_painting_token=object(),
                                e_field_norms=[12.3], Id_list=[42])
        EFieldNavigation.GetEnorm(state, None, False)
        EFieldNavigation.UpdateEfieldPointLocation(state, None, None, Mock())
        EFieldNavigation.UpdateEfieldPointLocationOffline(state, None, None, None)
        self.assertEqual(state.e_field_norms, [12.3])
        self.assertEqual(state.Id_list, [42])

    def test_return_to_live_resets_roi_cache_and_old_targeting_colors(self):
        state = SimpleNamespace(targeting_display_active=True, last_efield_cell_id=42,
                                e_field_norms=[12.3], Id_list=[42], radius_list=Mock(), visualizer=Mock())
        EFieldNavigation.SetTargetingDisplayMode(state, False)
        self.assertFalse(state.targeting_display_active)
        self.assertIsNone(state.last_efield_cell_id)
        self.assertEqual(state.Id_list, [])
        self.assertIsNone(state.e_field_norms)
        state.radius_list.Reset.assert_called_once()
        state.visualizer.ReturnToDefaultColorActor.assert_called_once()

    def _worker(self, active):
        inputs, outputs, ids = queue.Queue(), queue.Queue(), queue.Queue()
        roi = vtkIdList()
        roi.InsertNextId(0)
        ids.put(roi)
        inputs.put((None, [0]*6, 1))
        navigation = SimpleNamespace(e_field_loaded=True, e_field_revision=1,
                                     targeting_efield_active=active,
                                     e_field_calculation_lock=threading.RLock())
        api = Mock()
        event = Mock()
        worker = EFieldCalculationThread([inputs, outputs, ids], event, 0, api, None, False, navigation)
        return worker, navigation, api, event, outputs

    def test_worker_does_not_request_fields_while_result_is_held(self):
        worker, navigation, api, event, outputs = self._worker(True)
        event.is_set.side_effect = [False, True]
        worker.run()
        api.update_efield_vectorROIMax.assert_not_called()
        self.assertTrue(outputs.empty())

    def test_worker_discards_result_if_targeting_started_during_request(self):
        worker, navigation, api, event, outputs = self._worker(False)
        event.is_set.side_effect = [False, False, True]
        def result(**kwargs):
            navigation.targeting_efield_active = True
            return object()
        api.update_efield_vectorROIMax.side_effect = result
        with patch.dict(worker.run.__globals__, {"Get_coil_position": Mock(return_value=([], []))}):
            worker.run()
        api.update_efield_vectorROIMax.assert_called_once()
        self.assertTrue(outputs.empty())


if __name__ == "__main__":
    unittest.main()
