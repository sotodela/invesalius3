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


def recalculate_targeting_field(area, weights, api, mesh_point_count, restore_currents):
    """Recompute selected poses with weighted currents and verify superposition.

    The caller must ensure the original coil/mesh configuration is still loaded.
    Restore the backend current settings even when a calculation fails.
    """
    entries = area.get("efields", [])
    basis_currents = np.asarray(area.get("dIperdt", []), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (not entries or weights.shape != (len(entries),) or
            not np.all(np.isfinite(weights)) or basis_currents.ndim != 1 or
            basis_currents.size == 0 or not np.all(np.isfinite(basis_currents))):
        raise ValueError("Invalid targeting channel weights or basis currents")
    ids = list(entries[0]["roi_indices"])
    expected = np.zeros((len(ids), 3))
    groups = {}
    for entry, weight in zip(entries, weights):
        if list(entry["roi_indices"]) != ids:
            raise ValueError("Targeting channels have different ROI orders")
        vectors = np.column_stack([entry["efield"][name] for name in ("column1", "column2", "column3")])
        if vectors.shape != expected.shape or not np.all(np.isfinite(vectors)):
            raise ValueError("Invalid saved channel fields")
        expected += weight * vectors
        pose_index = entry["coil_position_index"]
        group = groups.setdefault(pose_index, {"entry": entry, "weights": {}})
        channel = int(entry["channel_index"])
        if channel < 0 or channel >= basis_currents.size or channel in group["weights"]:
            raise ValueError("Invalid or duplicate targeting channel index")
        group["weights"][channel] = float(weight)
    for group in groups.values():
        if len(group["weights"]) != basis_currents.size:
            raise ValueError("Missing channel fields for a targeting coil position")
    direct = np.zeros_like(expected)
    currents_by_pose = []
    try:
        for pose_index, group in groups.items():
            entry = group["entry"]
            currents = [group["weights"][i] * float(basis_currents[i]) for i in range(basis_currents.size)]
            if not api.set_dIperdt(dIperdt=currents):
                raise ValueError("E-field service rejected the optimized channel currents")
            response = api.update_efield_vectorROI(
                position=entry["position"], orientation=entry["orientation"],
                T_rot=entry["T_rot"], id_list=ids,
            )
            fields = extract_targeting_roi_fields(response, ids, mesh_point_count, basis_currents.size)
            for field in fields:
                direct += np.column_stack([field[name] for name in ("column1", "column2", "column3")])
            currents_by_pose.append({"coil_position_index": pose_index, "dIperdt": currents})
    finally:
        if not api.set_dIperdt(dIperdt=list(restore_currents)):
            raise ValueError("Could not restore E-field backend current settings")
    error = direct - expected
    denominator = float(np.linalg.norm(expected))
    absolute_error = float(np.linalg.norm(error))
    rtol, atol = 1e-4, 1e-8
    return {
        "id_list": ids,
        "vectors": direct.tolist(),
        "norms": np.linalg.norm(direct, axis=1).tolist(),
        "validation": {
            "display_source": "direct_efield_service",
            "relative_vector_error": absolute_error / denominator if denominator else None,
            "absolute_vector_error": absolute_error,
            "max_point_vector_error": float(np.max(np.linalg.norm(error, axis=1))) if ids else 0.0,
            "matches_reconstruction": bool(np.allclose(direct, expected, rtol=rtol, atol=atol)),
            "relative_tolerance": rtol, "absolute_tolerance": atol,
            "currents_by_pose": currents_by_pose,
        },
    }


def resolve_target_avoidance(targets, shared_region, roi_ids, mesh_point_count):
    """Resolve each target's override and include every required field sample."""
    resolved = []
    calculation_ids = list(roi_ids)
    for target in targets:
        region = target.get("avoidance_region", shared_region)
        indices = sorted(set(int(value) for value in region.get("indices", [])))
        strength = float(region.get("strength", 0.0))
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("Avoidance strength must be finite and nonnegative")
        if any(index < 0 or index >= mesh_point_count for index in indices):
            raise ValueError("Avoidance point outside the E-field cortex mesh")
        if strength > 0 and target["point_index"] in indices:
            raise ValueError("A target lies inside its avoidance region; adjust that target's selection")
        resolved.append(dict(target, avoidance_region={"indices": indices, "strength": strength}))
        calculation_ids.extend(indices)
    return resolved, list(dict.fromkeys(calculation_ids))


def recalculate_voltage_field(currents, api, mesh_point_count, position, rotation, orientation, restore_currents):
    """Compute whole-cortex vectors from model dI/dt; restore simulation state."""
    currents = np.asarray(currents, dtype=float)
    restore = np.asarray(restore_currents, dtype=float)
    if (currents.shape != (5,) or not np.all(np.isfinite(currents)) or restore.ndim != 1
            or len(restore) not in (5, 6) or not np.all(np.isfinite(restore)) or mesh_point_count <= 0):
        raise ValueError('Five model currents and known previous simulation settings are required')
    ids = list(range(mesh_point_count))
    try:
        if not api.set_dIperdt(dIperdt=currents.tolist()):
            raise ValueError('E-field model rejected voltage-derived currents')
        response = api.update_efield_vectorROI(position=position, orientation=orientation, T_rot=rotation, id_list=ids)
        fields = extract_targeting_roi_fields(response, ids, mesh_point_count, 5)
        vectors = np.sum([np.column_stack([field[k] for k in ('column1', 'column2', 'column3')]) for field in fields], axis=0)
    finally:
        if not api.set_dIperdt(dIperdt=restore.tolist()):
            raise ValueError('Could not restore the previous E-field model currents')
    return {'id_list': ids, 'vectors': vectors.tolist(), 'norms': np.linalg.norm(vectors, axis=1).tolist()}


def prepare_optimizer_geometry(targets, mesh, roi_ids, avoidance, optimizer, coil_positions):
    """Map targets and snapshot surface distances for a numerical worker."""
    from invesalius.data.surface_brush import SurfaceDistanceBrush
    method = optimizer.get("method", "least_squares")
    radius = float(optimizer.get("radius_mm", 5.0))
    if method not in ("localization", "least_squares"):
        raise ValueError("Unknown targeting optimizer")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Localization tolerance must be finite and nonnegative")
    if method == "localization" or optimizer.get("compare", False):
        if mesh is None:
            raise ValueError("Load the E-field cortex mesh first")
        if len(coil_positions) != 1:
            raise ValueError("Localization requires one fixed coil position; select one coil marker")
        roi_ids = list(range(mesh.GetNumberOfPoints()))
    mapped = map_targets_to_mesh(targets, mesh, roi_ids)
    mapped, roi_ids = resolve_target_avoidance(mapped, avoidance, roi_ids, mesh.GetNumberOfPoints())
    brush = SurfaceDistanceBrush(mesh)
    distances = [brush.distances_from(target["point_index"]) for target in mapped]
    return mapped, roi_ids, distances


def capture_field_export_geometry(mesh):
    """Original mesh vertex IDs and polygon connectivity, without reordering."""
    offsets=[0];face_ids=[]
    for i in range(mesh.GetNumberOfCells()):
        cell=mesh.GetCell(i)
        if cell.GetCellDimension()!=2:
            continue
        face_ids.extend(int(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints()))
        offsets.append(len(face_ids))
    return {"mesh_points_mm":[list(mesh.GetPoint(i)) for i in range(mesh.GetNumberOfPoints())],
            "mesh_face_offsets":offsets,"mesh_face_point_ids":face_ids}
