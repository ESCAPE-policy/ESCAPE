from typing import List

import numpy as np
import trimesh
from pxr import UsdGeom


def prim_to_trimesh_local(prim) -> trimesh.Trimesh:
    """Convert a supported USD geometry prim to a local-frame trimesh."""
    tname = prim.GetTypeName()
    if tname == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        vertices = np.array(mesh.GetPointsAttr().Get())
        face_vertex_counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        face_vertex_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())

        faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:
                faces.append(face_vertex_indices[idx : idx + 3])
            elif count > 3:
                for i in range(1, count - 1):
                    faces.append(
                        [
                            face_vertex_indices[idx],
                            face_vertex_indices[idx + i],
                            face_vertex_indices[idx + i + 1],
                        ]
                    )
            idx += count
        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    if tname == "Cube":
        size_attr = prim.GetAttribute("size")
        size = size_attr.Get() if size_attr and size_attr.HasAuthoredValue() else 2.0
        return trimesh.creation.box(extents=(size, size, size))

    raise ValueError(f"Unsupported prim type for robot sampling: {tname}")


def get_local_transform_matrix_from_prim(prim) -> np.ndarray:
    """Read one prim's local transform matrix using column-major convention."""
    xformable = UsdGeom.Xformable(prim)
    result = xformable.GetLocalTransformation()
    gf_mat = result[0] if isinstance(result, tuple) else result
    mat = np.array([[gf_mat[i][j] for j in range(4)] for i in range(4)], dtype=np.float32)
    return mat.T


def compute_relative_transform_matrix(parent_prim, child_prim) -> np.ndarray:
    """Compute the static transform from parent prim frame to child prim frame."""
    parent_path = parent_prim.GetPath().pathString
    child_path = child_prim.GetPath().pathString
    if parent_path == child_path:
        return np.eye(4, dtype=np.float32)

    prefix = parent_path.rstrip("/") + "/"
    if not child_path.startswith(prefix):
        raise ValueError(f"{child_path} is not a descendant of {parent_path}")

    relative_parts = child_path[len(prefix) :].split("/")
    stage = child_prim.GetStage()
    current_path = parent_path
    transform = np.eye(4, dtype=np.float32)
    for child_name in relative_parts:
        current_path = current_path + "/" + child_name
        current_prim = stage.GetPrimAtPath(current_path)
        transform = transform @ get_local_transform_matrix_from_prim(current_prim)
    return transform


def iter_descendant_geometry_prims(root_prim) -> List:
    """Collect supported geometry prims below a root prim."""
    supported = {"Mesh", "Cube"}
    geometry_prims = []
    stack = list(root_prim.GetChildren())
    while stack:
        prim = stack.pop()
        stack.extend(list(prim.GetChildren()))
        if prim.GetTypeName() not in supported:
            continue
        geometry_prims.append(prim)
    return geometry_prims

