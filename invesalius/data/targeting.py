"""Map directed marker targets onto the E-field calculation mesh."""

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkPointLocator


def map_targets_to_mesh(targets, mesh, roi_ids):
    """Return copied targets with global mesh point IDs, preserving ROI order.

    The loaded E-field mesh uses display coordinates. Marker positions use
    InVesalius coordinates, whose Y axis is reversed relative to the display.
    Resolve against the whole mesh first so a restricted ROI cannot silently
    move a target to a different anatomical region.
    """
    if mesh is None or mesh.GetNumberOfPoints() == 0:
        raise ValueError("Load the E-field cortex mesh before preparing targeting")
    if not targets:
        raise ValueError("Select at least one directed target point")
    roi_ids = [int(value) for value in roi_ids]
    if not roi_ids or any(i < 0 or i >= mesh.GetNumberOfPoints() for i in roi_ids):
        raise ValueError("Select valid E-field mesh indices for the ROI")
    if len(set(roi_ids)) != len(roi_ids):
        raise ValueError("The E-field ROI contains duplicate mesh indices")
    roi_set = set(roi_ids)
    locator = vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()
    mapped = []
    for index, target in enumerate(targets, 1):
        label = target.get("label") or f"Target {index}"
        position = np.asarray(target.get("position", []), dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{label}: a finite 3D target position is required")
        frame = target.get("coordinate_frame", "invesalius")
        if frame == "invesalius":
            mesh_position = position * [1.0, -1.0, 1.0]
        elif frame == "efield_mesh":
            mesh_position = position
        else:
            raise ValueError(f"{label}: unsupported coordinate frame {frame!r}")
        point_id = int(locator.FindClosestPoint(mesh_position))
        if point_id not in roi_set:
            raise ValueError(
                f"{label}: nearest mesh point {point_id} is outside the selected ROI; "
                "include that point or use whole cortex"
            )
        mapped.append(dict(
            target,
            point_index=point_id,
            mesh_position=mesh_position.tolist(),
            mesh_point_position=list(mesh.GetPoint(point_id)),
            mesh_distance_mm=float(np.linalg.norm(mesh_position - mesh.GetPoint(point_id))),
        ))
    return mapped


def extract_targeting_roi_fields(efield, roi_ids, mesh_point_count, coil_count):
    """Normalize the current E-field service's full-mesh, coil-major vectors.

    Keep each physical channel as an independent unit-dI/dt basis field.
    The service's summed norms are not used for channel optimization.
    """
    if efield is None or mesh_point_count <= 0 or coil_count <= 0:
        raise ValueError("No E-field vectors were returned for targeting")
    columns = []
    expected = mesh_point_count * coil_count
    for name in ("column1", "column2", "column3"):
        values = np.asarray(getattr(efield, name, []), dtype=float)
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            raise ValueError(f"E-field {name}: expected {expected} finite full-mesh values")
        columns.append(values.reshape(coil_count, mesh_point_count))
    roi_ids = np.asarray(roi_ids, dtype=int)
    if np.any(roi_ids < 0) or np.any(roi_ids >= mesh_point_count):
        raise ValueError("ROI index outside the returned E-field mesh")
    vectors = np.stack(columns, axis=-1)[:, roi_ids, :]
    return [
        {
            "column1": channel[:, 0].tolist(),
            "column2": channel[:, 1].tolist(),
            "column3": channel[:, 2].tolist(),
            "enorm": np.linalg.norm(channel, axis=1).tolist(),
        }
        for channel in vectors
    ]
