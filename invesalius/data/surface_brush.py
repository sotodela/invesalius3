"""Radius selection measured along connected surface-mesh edges."""

import heapq

import numpy as np


class SurfaceDistanceBrush:
    """Bounded Dijkstra brush with cached, geometry-dependent adjacency.

    Distances approximate surface geodesics along mesh edges. Nearby but
    disconnected vertices are deliberately not welded: that could bridge sulci.
    Scalar/color changes do not invalidate the topology cache.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._signature = None
        self._adjacency = None

    def _update_adjacency(self):
        mesh = self.mesh
        points = mesh.GetPoints()
        signature = (
            points.GetMTime() if points is not None else None,
            mesh.GetPolys().GetMTime(), mesh.GetStrips().GetMTime(),
            mesh.GetLines().GetMTime(), mesh.GetVerts().GetMTime(),
            mesh.GetNumberOfPoints(), mesh.GetNumberOfCells(),
        )
        if signature == self._signature:
            return
        adjacency = [{} for _ in range(mesh.GetNumberOfPoints())]
        for cell_id in range(mesh.GetNumberOfCells()):
            cell = mesh.GetCell(cell_id)
            if cell.GetCellDimension() != 2:
                continue
            for edge_id in range(cell.GetNumberOfEdges()):
                edge = cell.GetEdge(edge_id)
                if edge.GetNumberOfPoints() != 2:
                    raise ValueError("Surface brush requires a linear polygon mesh")
                a, b = edge.GetPointId(0), edge.GetPointId(1)
                length = float(np.linalg.norm(np.asarray(mesh.GetPoint(a)) - mesh.GetPoint(b)))
                if not np.isfinite(length):
                    raise ValueError("Surface brush mesh contains nonfinite coordinates")
                adjacency[a][b] = length
                adjacency[b][a] = length
        self._adjacency = adjacency
        self._signature = signature

    def points_within_radius(self, seed, radius):
        if not np.isfinite(float(radius)):
            raise ValueError("Surface brush radius must be finite and nonnegative")
        return sorted(self.distances_from(seed, radius))

    def distances_from(self, seed, radius=float("inf")):
        """Mesh-edge distances in mesh units; disconnected vertices are absent."""
        radius = float(radius)
        if np.isnan(radius) or radius < 0:
            raise ValueError("Surface brush radius must be finite and nonnegative")
        if seed < 0 or seed >= self.mesh.GetNumberOfPoints():
            raise ValueError("Surface brush seed outside mesh")
        self._update_adjacency()
        distances = {int(seed): 0.0}
        pending = [(0.0, int(seed))]
        while pending:
            distance, point_id = heapq.heappop(pending)
            if distance > distances[point_id]:
                continue
            for neighbor, length in self._adjacency[point_id].items():
                candidate = distance + length
                if candidate <= radius and candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    heapq.heappush(pending, (candidate, neighbor))
        return distances


def picked_surface_point_id(picked_mesh, cell_id, position, original_locator):
    """Seed on the clicked face, rather than a nearby vertex on another bank.

    Display normals can duplicate/reorder vertices, so map the chosen face
    vertex's coordinates back to the original calculation mesh.
    """
    if picked_mesh is None or not 0 <= cell_id < picked_mesh.GetNumberOfCells():
        return -1
    cell = picked_mesh.GetCell(cell_id)
    if cell.GetNumberOfPoints() == 0:
        return -1
    position = np.asarray(position, dtype=float)
    vertex = min(
        (picked_mesh.GetPoint(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())),
        key=lambda point: float(np.linalg.norm(np.asarray(point) - position)),
    )
    return int(original_locator.FindClosestPoint(vertex))
