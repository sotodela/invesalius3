import unittest
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkTriangle, vtkCellArray
from invesalius.data.targeting import prepare_optimizer_geometry


class LocalizationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.mesh=vtkPolyData();points=vtkPoints()
        for point in [(0,0,0),(3,0,0),(0,4,0)]:points.InsertNextPoint(*point)
        self.mesh.SetPoints(points)
        triangle=vtkTriangle()
        for i in range(3):triangle.GetPointIds().SetId(i,i)
        cells=vtkCellArray();cells.InsertNextCell(triangle);self.mesh.SetPolys(cells)
        self.targets=[{'position':[0,0,0],'direction':[1,0,0]}]

    def test_localization_forces_whole_cortex_with_surface_distance_snapshot(self):
        for roi in ([],[1]):
            targets,ids,distances=prepare_optimizer_geometry(self.targets,self.mesh,roi,{},
                {'method':'localization','radius_mm':5},[{}])
            self.assertEqual(ids,[0,1,2])
            self.assertEqual(targets[0]['point_index'],0)
            self.assertEqual(distances,[{0:0.,1:3.,2:4.}])

    def test_comparison_uses_whole_cortex_even_when_least_squares_is_selected(self):
        _,ids,_=prepare_optimizer_geometry(self.targets,self.mesh,[],{},
            {'method':'least_squares','compare':True},[{}])
        self.assertEqual(ids,[0,1,2])

    def test_least_squares_keeps_selected_roi(self):
        _,ids,_=prepare_optimizer_geometry(self.targets,self.mesh,[0,1],{},
            {'method':'least_squares'},[{}])
        self.assertEqual(ids,[0,1])

    def test_multiple_poses_and_invalid_settings_rejected(self):
        for settings,poses in [({'method':'localization'},[{},{}]),
                              ({'method':'unknown'},[{}]),
                              ({'method':'localization','radius_mm':-1},[{}])]:
            with self.assertRaises(ValueError):
                prepare_optimizer_geometry(self.targets,self.mesh,[0],{},settings,poses)
