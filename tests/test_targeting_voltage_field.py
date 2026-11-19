import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray, vtkTriangle
from invesalius.data.targeting import recalculate_voltage_field
from invesalius.gui import task_efield


class VoltageFieldTests(unittest.TestCase):
    def api(self):
        api = Mock()
        api.set_dIperdt.return_value = True
        api.update_efield_vectorROI.return_value = SimpleNamespace(column1=[1,2]*5, column2=[0]*10, column3=[0]*10)
        return api

    def test_whole_cortex_sum_and_restore_on_success(self):
        api = self.api()
        result = recalculate_voltage_field([1,2,3,4,5], api, 2, [0]*3, [1]*9, [0]*3, [7]*5)
        self.assertEqual(result['vectors'], [[5,0,0],[10,0,0]])
        self.assertEqual(api.set_dIperdt.call_args_list[0].kwargs['dIperdt'], [1,2,3,4,5])
        self.assertEqual(api.set_dIperdt.call_args_list[-1].kwargs['dIperdt'], [7]*5)
        self.assertEqual(api.update_efield_vectorROI.call_args.kwargs['id_list'], [0,1])

    def test_restore_on_calculation_and_shape_failure(self):
        for failure in ('exception', 'shape'):
            api = self.api()
            if failure == 'exception': api.update_efield_vectorROI.side_effect = RuntimeError('offline failure')
            else: api.update_efield_vectorROI.return_value = SimpleNamespace(column1=[],column2=[],column3=[])
            with self.assertRaises((RuntimeError, ValueError)):
                recalculate_voltage_field([1]*5, api, 2, [0]*3, [1]*9, [0]*3, [7]*5)
            self.assertEqual(api.set_dIperdt.call_args_list[-1].kwargs['dIperdt'], [7]*5)

    def test_core_checks_geometry_and_requests_all_points_without_device_calls(self):
        points = vtkPoints()
        for xyz in [(0,0,0),(1,0,0),(0,1,0)]: points.InsertNextPoint(*xyz)
        mesh = vtkPolyData(); mesh.SetPoints(points)
        triangle = vtkTriangle()
        for i in range(3): triangle.GetPointIds().SetId(i,i)
        cells = vtkCellArray(); cells.InsertNextCell(triangle); mesh.SetPolys(cells)
        api = self.api()
        api.update_efield_vectorROI.return_value = SimpleNamespace(column1=[1,2,3]*5,column2=[0]*15,column3=[0]*15)
        navigation = SimpleNamespace(neuronavigation_api=api, targeting_efield_active=False,
            e_field_calculation_lock=contextlib.nullcontext(), SetTargetingEfieldMode=Mock())
        state = SimpleNamespace(e_field_mesh=mesh, coil_set=True, coil='/models/coilset-5-v230908.bin',
            _reported_dIperdt=[7]*5, navigation=navigation)
        request = {'polarity_profile':'coilset_5_v230908','currents_a_per_s':[1]*5, 'target_patch_radius_mm':.5,
            'target':{'position':[0,0,0],'orientation':[0,0,0],'coordinate_frame':'invesalius','point_index':0,'world_position':[0,0,0]},
            'coil_position':{'position':[0,0,0],'orientation':[0,0,0],'world_position':[0,0,0],'world_orientation':[0,0,0]}}
        publisher = Mock()
        rotation = [0,-1,0,1,0,0,0,0,1]
        with patch.object(task_efield,'Publisher',publisher), patch.object(task_efield,'Get_coil_position',return_value=(rotation,[0]*3)), patch.object(task_efield.imagedata_utils,'convert_invesalius_to_world',return_value=([0]*3,[0]*3)):
            task_efield.InnerTaskPanel.OnValidateVoltageSettings(state, request)
        self.assertEqual(publisher.sendMessage.call_args.args[0], 'Voltage settings Efield recalculated')
        self.assertEqual(api.update_efield_vectorROI.call_args.kwargs['id_list'], [0,1,2])
        self.assertEqual(publisher.sendMessage.call_args.kwargs['result']['target_patch_ids'], [0])
        api.send_pulse.assert_not_called(); api.send_charge_or_discharge.assert_not_called()
        api.reset_mock(); request['target']['point_index'] = 2
        with patch.object(task_efield,'Publisher',publisher):
            task_efield.InnerTaskPanel.OnValidateVoltageSettings(state, request)
        self.assertEqual(publisher.sendMessage.call_args.args[0], 'Voltage settings validation failed')
        api.set_dIperdt.assert_not_called()

    def test_restoration_failure_is_an_error(self):
        api = self.api(); api.set_dIperdt.side_effect = [True, False]
        with self.assertRaisesRegex(ValueError, 'restore'):
            recalculate_voltage_field([1]*5, api, 2, [0]*3, [1]*9, [0]*3, [7]*5)
