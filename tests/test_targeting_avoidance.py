import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPointLocator, vtkPolyData

from invesalius.gui.task_efield import InnerTaskPanel
from invesalius.navigation.efield_navigation import EFieldNavigation
from invesalius.data.targeting import resolve_target_avoidance


class AvoidanceSelectionTests(unittest.TestCase):
    def setUp(self):
        points = vtkPoints()
        for point in [(0, 0, 0), (10, 0, 0), (10.5, 0, 0)]:
            points.InsertNextPoint(*point)
        self.mesh = vtkPolyData()
        self.mesh.SetPoints(points)

    def test_radius_selection_uses_mesh_point_coordinates(self):
        locator = vtkPointLocator()
        locator.SetDataSet(self.mesh)
        locator.BuildLocator()
        state = SimpleNamespace(efield_mesh=self.mesh, locator_efield=locator, efield_targeting_area_radius=1.)
        self.assertEqual(sorted(EFieldNavigation._GetEfieldIdsAroundCell(state, 1)), [1, 2])

    def test_avoidance_selection_does_not_change_calculation_roi(self):
        state = SimpleNamespace(efield_mesh=None, efield_targeting_area_ids={0}, efield_avoidance_ids=set())
        publisher = Mock()
        with patch.dict(EFieldNavigation.SetEfieldAvoidanceArea.__globals__, {"Publisher": publisher}):
            EFieldNavigation.SetEfieldAvoidanceArea(state, [2])
        self.assertEqual(state.efield_targeting_area_ids, {0})
        self.assertEqual(state.efield_avoidance_ids, {2})
        publisher.sendMessage.assert_called_once_with("Efield avoidance area selected", selected_ids=[2])

    def _panel(self):
        api = Mock()
        api.update_efield_vectorROI.return_value = SimpleNamespace(column1=[1, 2, 3], column2=[0]*3, column3=[0]*3)
        navigation = SimpleNamespace(neuronavigation_api=api, SetTargetingEfieldMode=Mock(),
                                     MarkEfieldParametersChanged=Mock(), e_field_calculation_lock=threading.RLock())
        return SimpleNamespace(e_field_mesh=self.mesh, coil='coil.file', coil_set=False, ci=[], co=[],
                               navigation=navigation, GetUnitdIPerdt=lambda:[1], Send_dI_per_dt_to_report=Mock())

    def test_avoidance_outside_roi_is_included_in_field_calculation(self):
        state = self._panel()
        publisher = Mock()
        request = {"target_region":{"indices":[0]}, "target_points":[{"position":[0,0,0]}],
                   "coil_positions":[{"position":[0,0,0],"orientation":[0,0,0]}],
                   "avoidance_region":{"indices":[2],"strength":3.}}
        with patch.dict(InnerTaskPanel.OnRequestEfieldsForTargeting.__globals__, {
            "Publisher":publisher, "Get_coil_position":Mock(return_value=([],[0,0,0])),
            "imagedata_utils":SimpleNamespace(convert_invesalius_to_world=lambda **kw:([0,0,0],[0,0,0])),
        }):
            InnerTaskPanel.OnRequestEfieldsForTargeting(state, request)
        api = state.navigation.neuronavigation_api
        self.assertEqual(api.update_efield_vectorROI.call_args.kwargs['id_list'], [0,2])
        result = publisher.sendMessage.call_args.kwargs
        self.assertEqual(result['efields'][0]['roi_indices'], [0,2])
        self.assertEqual(result['targeting_request']['target_region']['indices'], [0])
        self.assertEqual(result['targeting_request']['avoidance_region']['indices'], [2])

    def test_overlapping_target_or_invalid_avoidance_is_rejected_before_service_call(self):
        for region in [{'indices':[0],'strength':1}, {'indices':[99],'strength':1}]:
            with self.subTest(region=region):
                state = self._panel()
                publisher = Mock()
                request = {'target_region':{'indices':[0]}, 'target_points':[{'position':[0,0,0]}],
                           'avoidance_region':region}
                with patch.dict(InnerTaskPanel.OnRequestEfieldsForTargeting.__globals__, {'Publisher':publisher}):
                    InnerTaskPanel.OnRequestEfieldsForTargeting(state, request)
                state.navigation.neuronavigation_api.update_efield_vectorROI.assert_not_called()
                self.assertIn('message', publisher.sendMessage.call_args.kwargs)


    def test_per_target_avoidance_resolves_independently_and_unions_required_samples(self):
        targets=[{'point_index':0,'avoidance_region':{'indices':[1],'strength':10}},
                 {'point_index':1,'avoidance_region':{'indices':[0],'strength':2}}]
        resolved,ids=resolve_target_avoidance(targets,{'indices':[0,1],'strength':100},[2],3)
        self.assertEqual(ids,[2,1,0])
        self.assertEqual(resolved[0]['avoidance_region']['strength'],10)
        self.assertEqual(resolved[1]['avoidance_region']['indices'],[0])
        # Avoiding a different target is allowed; only self-overlap is rejected.
        with self.assertRaises(ValueError):
            resolve_target_avoidance([{'point_index':0}],{'indices':[0],'strength':1},[0],3)


if __name__ == '__main__':
    unittest.main()
