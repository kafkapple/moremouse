"""
Mouse Body Model

Articulated mesh model for C57BL/6 laboratory mouse based on MAMMAL.
Implements Linear Blend Skinning (LBS) for pose-dependent deformation.

Reference:
- MAMMAL mouse model: 140 joints, 13059 vertices
- MoReMouse paper: Section 3.1 (Gaussian Mouse Avatar)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def to_tensor(array, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy array to torch tensor if necessary."""
    if isinstance(array, torch.Tensor):
        return array.to(dtype)
    # Handle numpy.matrix (from scipy.sparse.todense()) - convert to ndarray first
    if hasattr(array, 'A'):  # numpy.matrix has .A attribute
        array = np.asarray(array)
    return torch.tensor(array, dtype=dtype)


class MouseBodyModel(nn.Module):
    """
    Articulated mouse body model with Linear Blend Skinning.

    Based on MAMMAL mouse model with:
    - 140 articulated joints
    - 13059 vertices
    - Euler angle rotation (ZYX convention)

    Args:
        model_path: Path to mouse.pkl file
        device: Torch device
        dtype: Data type for computations
    """

    # Model statistics
    NUM_JOINTS = 140
    NUM_VERTICES = 13059
    NUM_BONE_LENGTHS = 28

    def __init__(
        self,
        model_path: Union[str, Path],
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype

        # Load model data
        model_path = Path(model_path)
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

        # Register buffers (non-trainable parameters)
        self.register_buffer(
            "v_template",
            to_tensor(params['vertices'], dtype)
        )
        self.register_buffer(
            "t_pose_joints",
            to_tensor(params['t_pose_joints'], dtype)
        )

        # Load skinning weights - handle sparse matrix and shape variations
        skinning_weights = params['skinning_weights']
        if hasattr(skinning_weights, 'todense'):
            skinning_weights = np.asarray(skinning_weights.todense())
        else:
            skinning_weights = np.asarray(skinning_weights)

        # Get expected dimensions
        num_verts = params['vertices'].shape[0]
        num_joints = params['t_pose_joints'].shape[0]

        # Handle shape: should be [V, J] for einsum 'bjmn,vj->bvmn'
        if skinning_weights.shape == (num_joints, num_verts):
            # Transpose from [J, V] to [V, J]
            skinning_weights = skinning_weights.T
        elif skinning_weights.shape != (num_verts, num_joints):
            # Try to find correct interpretation
            if skinning_weights.shape[0] == num_verts:
                # Might have extra columns, slice to num_joints
                skinning_weights = skinning_weights[:, :num_joints]
            elif skinning_weights.shape[1] == num_verts:
                # Transposed with extra rows
                skinning_weights = skinning_weights[:num_joints, :].T
            else:
                raise ValueError(
                    f"Cannot interpret skinning_weights shape {skinning_weights.shape}. "
                    f"Expected ({num_verts}, {num_joints}) or ({num_joints}, {num_verts})"
                )

        self.register_buffer(
            "weights",
            to_tensor(skinning_weights, dtype)
        )
        self.register_buffer(
            "parent",
            to_tensor(params['parents'], dtype=torch.int64)
        )

        # Load bone length mapper
        bone_mapper_path = model_path.parent / "mouse_txt" / "bone_length_mapper.txt"
        if bone_mapper_path.exists():
            self.bone_length_mapper = np.loadtxt(bone_mapper_path, dtype=np.int64).squeeze()
        else:
            # Default identity mapper
            self.bone_length_mapper = np.arange(self.NUM_JOINTS, dtype=np.int64)

        # Faces
        self.faces = params['faces_vert']

        # Load reduced faces if available
        reduced_face_path = model_path.parent / "mouse_txt" / "reduced_face_7200.txt"
        if reduced_face_path.exists():
            self.faces_reduced = np.loadtxt(reduced_face_path, dtype=np.int64)
        else:
            self.faces_reduced = self.faces

        # Load UV coordinates and UV face indices
        uv_coords_path = model_path.parent / "mouse_txt" / "textures.txt"
        faces_tex_path = model_path.parent / "mouse_txt" / "faces_tex.txt"

        if uv_coords_path.exists() and faces_tex_path.exists():
            # UV coordinates: (N_uv, 2) in [0, 1]
            uv_coords = np.loadtxt(uv_coords_path, dtype=np.float32)
            self.register_buffer("uv_coords", to_tensor(uv_coords, dtype))

            # UV face indices: (N_faces, 3)
            faces_tex = np.loadtxt(faces_tex_path, dtype=np.int64)
            self.register_buffer("faces_tex", torch.tensor(faces_tex, dtype=torch.int64))

            # Compute UV-to-vertex mapping
            # Each UV vertex maps to a 3D mesh vertex via face correspondence
            uv_to_vert = self._compute_uv_to_vert_mapping(
                self.faces, faces_tex, len(uv_coords)
            )
            self.register_buffer("uv_to_vert", torch.tensor(uv_to_vert, dtype=torch.int64))

            self.num_uv_coords = len(uv_coords)
            self._has_uv = True
        else:
            self._has_uv = False
            self.num_uv_coords = 0

        # Keypoint mapper for 22 keypoints
        self.mapper = None
        mapper_path = model_path.parent / "keypoint22_mapper.json"
        if mapper_path.exists():
            with open(mapper_path, 'r') as f:
                data = json.load(f)
            self.mapper = data.get("mapper", [])

        # Model dimensions
        self.num_vertices = self.v_template.shape[0]
        self.num_joints = self.t_pose_joints.shape[0]

        # Cached outputs
        self._V_posed: Optional[torch.Tensor] = None
        self._J_posed: Optional[torch.Tensor] = None

        # Move to device
        self.to(self.device)

    @staticmethod
    def _compute_uv_to_vert_mapping(
        faces_vert: np.ndarray,
        faces_tex: np.ndarray,
        num_uv_coords: int,
    ) -> np.ndarray:
        """
        Compute mapping from UV vertices to 3D mesh vertices.

        For each UV vertex, find the corresponding 3D mesh vertex
        using face correspondence: faces_tex[f, i] -> faces_vert[f, i]

        Args:
            faces_vert: (N_faces, 3) 3D mesh face indices
            faces_tex: (N_faces, 3) UV face indices
            num_uv_coords: Number of UV coordinates

        Returns:
            uv_to_vert: (N_uv,) mapping from UV to vertex indices
        """
        uv_to_vert = np.zeros(num_uv_coords, dtype=np.int64)

        for f in range(len(faces_vert)):
            for i in range(3):
                uv_idx = faces_tex[f, i]
                vert_idx = faces_vert[f, i]
                uv_to_vert[uv_idx] = vert_idx

        return uv_to_vert

    def compute_vertex_normals(
        self,
        vertices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute vertex normals from mesh faces.

        Args:
            vertices: [B, V, 3] vertex positions

        Returns:
            normals: [B, V, 3] normalized vertex normals
        """
        B = vertices.shape[0]
        device = vertices.device
        dtype = vertices.dtype

        faces = torch.tensor(self.faces, dtype=torch.int64, device=device)

        # Get vertices of each face
        v0 = vertices[:, faces[:, 0]]  # [B, F, 3]
        v1 = vertices[:, faces[:, 1]]
        v2 = vertices[:, faces[:, 2]]

        # Compute face normals
        e1 = v1 - v0
        e2 = v2 - v0
        face_normals = torch.cross(e1, e2, dim=-1)  # [B, F, 3]

        # Accumulate face normals to vertices
        vertex_normals = torch.zeros(B, self.num_vertices, 3, dtype=dtype, device=device)

        for i in range(3):
            vertex_normals.scatter_add_(
                1,
                faces[:, i].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3),
                face_normals,
            )

        # Normalize
        vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=-1)

        return vertex_normals

    def compute_tangent_frames(
        self,
        vertices: torch.Tensor,
        normals: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute local tangent frames (normal, tangent, bitangent) for each vertex.

        Uses UV coordinates to define consistent tangent directions.

        Args:
            vertices: [B, V, 3] vertex positions
            normals: [B, V, 3] vertex normals (computed if None)

        Returns:
            normals: [B, V, 3] normal vectors
            tangents: [B, V, 3] tangent vectors (along U direction)
            bitangents: [B, V, 3] bitangent vectors (along V direction)
        """
        if normals is None:
            normals = self.compute_vertex_normals(vertices)

        B = vertices.shape[0]
        device = vertices.device
        dtype = vertices.dtype

        if not self._has_uv:
            # Fallback: construct arbitrary orthonormal frame
            up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
            tangents = torch.cross(normals, up.expand_as(normals), dim=-1)
            tangents = torch.nn.functional.normalize(tangents, dim=-1)
            bitangents = torch.cross(normals, tangents, dim=-1)
            return normals, tangents, bitangents

        faces = torch.tensor(self.faces, dtype=torch.int64, device=device)

        # Get UV coordinates
        uv = self.uv_coords  # [N_uv, 2]

        # Map UV to vertex indices
        uv_per_face = self.faces_tex  # [F, 3]
        vert_per_face = faces  # [F, 3]

        # Get positions and UVs per face
        v0 = vertices[:, vert_per_face[:, 0]]  # [B, F, 3]
        v1 = vertices[:, vert_per_face[:, 1]]
        v2 = vertices[:, vert_per_face[:, 2]]

        uv0 = uv[uv_per_face[:, 0]]  # [F, 2]
        uv1 = uv[uv_per_face[:, 1]]
        uv2 = uv[uv_per_face[:, 2]]

        # Compute edges
        e1 = v1 - v0  # [B, F, 3]
        e2 = v2 - v0

        duv1 = uv1 - uv0  # [F, 2]
        duv2 = uv2 - uv0

        # Compute tangent per face
        # T = (duv2.y * e1 - duv1.y * e2) / (duv1.x * duv2.y - duv2.x * duv1.y)
        denom = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        denom = denom.unsqueeze(0).unsqueeze(-1)  # [1, F, 1]
        denom = torch.clamp(denom, min=1e-8)  # Avoid division by zero

        face_tangents = (duv2[:, 1].unsqueeze(0).unsqueeze(-1) * e1 -
                        duv1[:, 1].unsqueeze(0).unsqueeze(-1) * e2) / denom

        # Accumulate tangents to vertices
        tangents = torch.zeros(B, self.num_vertices, 3, dtype=dtype, device=device)
        for i in range(3):
            tangents.scatter_add_(
                1,
                vert_per_face[:, i].unsqueeze(0).unsqueeze(-1).expand(B, -1, 3),
                face_tangents,
            )

        # Gram-Schmidt orthonormalization
        # tangent = normalize(tangent - dot(tangent, normal) * normal)
        tangents = tangents - (tangents * normals).sum(dim=-1, keepdim=True) * normals
        tangents = torch.nn.functional.normalize(tangents, dim=-1)

        # Bitangent = cross(normal, tangent)
        bitangents = torch.cross(normals, tangents, dim=-1)

        return normals, tangents, bitangents

    @property
    def has_uv(self) -> bool:
        """Check if UV coordinates are available."""
        return self._has_uv

    @staticmethod
    def euler_to_rotation_matrix(euler_angles: torch.Tensor) -> torch.Tensor:
        """
        Convert Euler angles (ZYX convention) to rotation matrices.

        Args:
            euler_angles: [N, 1, 3] tensor of Euler angles (z, y, x)

        Returns:
            Rotation matrices [N, 3, 3]
        """
        N = euler_angles.shape[0]
        device = euler_angles.device
        dtype = euler_angles.dtype

        z = euler_angles[:, 0, 0]
        y = euler_angles[:, 0, 1]
        x = euler_angles[:, 0, 2]

        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)

        # Rotation matrices
        zeros = torch.zeros(N, dtype=dtype, device=device)
        ones = torch.ones(N, dtype=dtype, device=device)

        Rx = torch.stack([
            ones, zeros, zeros,
            zeros, cx, -sx,
            zeros, sx, cx
        ], dim=1).reshape(N, 3, 3)

        Ry = torch.stack([
            cy, zeros, sy,
            zeros, ones, zeros,
            -sy, zeros, cy
        ], dim=1).reshape(N, 3, 3)

        Rz = torch.stack([
            cz, -sz, zeros,
            sz, cz, zeros,
            zeros, zeros, ones
        ], dim=1).reshape(N, 3, 3)

        # ZYX order
        R = torch.bmm(torch.bmm(Rz, Ry), Rx)
        return R

    @staticmethod
    def with_zeros(x: torch.Tensor) -> torch.Tensor:
        """Append [0, 0, 0, 1] row to [B, 3, 4] tensor -> [B, 4, 4]."""
        B = x.shape[0]
        bottom = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]],
            dtype=x.dtype,
            device=x.device
        ).expand(B, -1, -1)
        return torch.cat([x, bottom], dim=1)

    @staticmethod
    def pack(x: torch.Tensor) -> torch.Tensor:
        """Append zero columns to [B, J, 4, 1] -> [B, J, 4, 4]."""
        B, J = x.shape[0], x.shape[1]
        zeros = torch.zeros(B, J, 4, 3, dtype=x.dtype, device=x.device)
        return torch.cat([zeros, x], dim=3)

    def _compute_global_transforms(
        self,
        local_rotations: torch.Tensor,
        J: torch.Tensor,
        bone_lengths: torch.Tensor,
        center_bone_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global transformation matrices for all joints.

        Args:
            local_rotations: [B, J, 3, 3] local rotation matrices
            J: [B, J, 3] joint positions
            bone_lengths: [B, 28] bone length parameters
            center_bone_length: [B, 1] center bone scale

        Returns:
            G: [B, J, 4, 4] global transformation matrices
            J_final: [B, J, 3] transformed joint positions
        """
        B = local_rotations.shape[0]
        results = []

        # Root joint
        root_transform = self.with_zeros(
            torch.cat([
                local_rotations[:, 0],
                J[:, 0:1, :].transpose(1, 2)
            ], dim=2)
        )
        results.append(root_transform)

        # Child joints
        for i in range(1, self.num_joints):
            parent_idx = self.parent[i].item()
            bone_length_id = self.bone_length_mapper[i]

            # Compute bone offset
            bone_offset = J[:, i, :] - J[:, parent_idx, :]

            if i == 1:
                # Center bone uses center_bone_length
                bone_offset = bone_offset * center_bone_length
            elif bone_length_id >= 0:
                # Use specific bone length
                scale = bone_lengths[:, bone_length_id:bone_length_id + 1]
                bone_offset = bone_offset * scale

            # Local transform
            local_transform = self.with_zeros(
                torch.cat([
                    local_rotations[:, i],
                    bone_offset.unsqueeze(-1)
                ], dim=2)
            )

            # Global transform
            global_transform = torch.bmm(results[parent_idx], local_transform)
            results.append(global_transform)

        # Stack all transforms
        G = torch.stack(results, dim=1)  # [B, J, 4, 4]

        # Extract final joint positions
        J_final = G[:, :, :3, 3]

        # Remove joint rest pose offset
        J_homo = torch.cat([
            J,
            torch.zeros(B, self.num_joints, 1, dtype=self.dtype, device=self.device)
        ], dim=2)  # [B, J, 4]

        deformed_joint = torch.matmul(
            G,
            J_homo.unsqueeze(-1)
        )  # [B, J, 4, 1]

        G = G - self.pack(deformed_joint)

        return G, J_final

    def forward(
        self,
        pose: torch.Tensor,
        bone_lengths: torch.Tensor = None,
        center_bone_length: torch.Tensor = None,
        trans: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward LBS to compute posed vertices and joints.

        Args:
            pose: [B, J*3] Euler angles for all joints
            bone_lengths: [B, 28] bone length parameters (default: zeros)
            center_bone_length: [B, 1] center bone scale (default: ones)
            trans: [B, 3] global translation (default: zeros)
            scale: [B, 1] global scale (default: ones)

        Returns:
            V: [B, V, 3] posed vertices
            J: [B, J, 3] posed joint positions
        """
        B = pose.shape[0]
        device = pose.device

        # Defaults
        if bone_lengths is None:
            bone_lengths = torch.zeros(B, self.NUM_BONE_LENGTHS, dtype=self.dtype, device=device)
        if center_bone_length is None:
            center_bone_length = torch.ones(B, 1, dtype=self.dtype, device=device)
        if trans is None:
            trans = torch.zeros(B, 3, dtype=self.dtype, device=device)
        if scale is None:
            scale = torch.ones(B, 1, dtype=self.dtype, device=device)

        # Apply sigmoid to bone lengths for bounded scaling
        bone_lengths_scaled = torch.sigmoid(bone_lengths / 5) * 2

        # T-pose joints
        J = self.t_pose_joints.unsqueeze(0).expand(B, -1, -1)

        # Compute rotation matrices from pose
        pose_reshaped = pose.view(-1, 1, 3)  # [B*J, 1, 3]
        local_rotations = self.euler_to_rotation_matrix(pose_reshaped)
        local_rotations = local_rotations.view(B, self.num_joints, 3, 3)

        # Compute global transforms
        G, J_final = self._compute_global_transforms(
            local_rotations, J, bone_lengths_scaled, center_bone_length
        )

        # Skinning
        T = torch.einsum('bjmn,vj->bvmn', G, self.weights)  # [B, V, 4, 4]

        # Apply to vertices
        v_template = self.v_template.unsqueeze(0).expand(B, -1, -1)
        v_homo = torch.cat([
            v_template,
            torch.ones(B, self.num_vertices, 1, dtype=self.dtype, device=device)
        ], dim=2)  # [B, V, 4]

        V = torch.matmul(T, v_homo.unsqueeze(-1)).squeeze(-1)[:, :, :3]  # [B, V, 3]

        # Apply global scale and translation
        V = V * scale.view(B, 1, 1) + trans.view(B, 1, 3)
        J_final = J_final * scale.view(B, 1, 1) + trans.view(B, 1, 3)

        # Cache outputs
        self._V_posed = V
        self._J_posed = J_final

        return V, J_final

    def get_keypoints_22(self) -> torch.Tensor:
        """
        Get 22 keypoints from posed mesh.

        Returns:
            keypoints: [B, 22, 3] keypoint positions
        """
        if self._V_posed is None or self.mapper is None:
            raise RuntimeError("Call forward() first and ensure mapper is loaded")

        B = self._V_posed.shape[0]
        keypoints = torch.zeros(B, 22, 3, dtype=self.dtype, device=self.device)

        for item in self.mapper:
            kp_id = item["keypoint"]
            if kp_id >= 22:
                continue

            if item["type"] == "V":
                # Average of vertex indices
                keypoints[:, kp_id] = self._V_posed[:, item["ids"]].mean(dim=1)
            elif item["type"] == "J":
                # Average of joint indices
                keypoints[:, kp_id] = self._J_posed[:, item["ids"]].mean(dim=1)

        return keypoints

    def get_faces(self, reduced: bool = False) -> np.ndarray:
        """Get mesh faces."""
        return self.faces_reduced if reduced else self.faces

    @property
    def posed_vertices(self) -> Optional[torch.Tensor]:
        """Get cached posed vertices."""
        return self._V_posed

    @property
    def posed_joints(self) -> Optional[torch.Tensor]:
        """Get cached posed joints."""
        return self._J_posed


def load_mouse_model(
    model_dir: Union[str, Path],
    device: torch.device = None,
) -> MouseBodyModel:
    """
    Convenience function to load mouse model.

    Args:
        model_dir: Directory containing mouse.pkl
        device: Torch device

    Returns:
        MouseBodyModel instance
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "mouse.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Mouse model not found: {model_path}")

    return MouseBodyModel(model_path, device=device)
