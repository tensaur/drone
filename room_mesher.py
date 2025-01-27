import numpy as np
import open3d as o3d
from collections import deque
import matplotlib.pyplot as plt

def mesh(room_path):
    point_cloud = o3d.io.read_point_cloud(room_path)

    # Perform voxel downsampling
    voxel_size = 0.1  # Adjust this value to control the level of downsampling
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    # Convert downsampled point cloud to numpy array
    point_cloud = np.asarray(downsampled_pcd.points)

    # Size of each grid cell
    grid_resolution = 1

    # Calculate Grid Bounds
    min_bounds = point_cloud.min(axis=0)
    max_bounds = point_cloud.max(axis=0)

    # Create the Grid
    grid_indices = np.floor((point_cloud - min_bounds) / grid_resolution).astype(int)
    unique_cells = set(map(tuple, grid_indices))  # Occupied cells (the shell)

    def flood_fill_outside(shell_cells, grid_bounds):
        neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        queue = deque()
        visited = set()
        external_cells = set()

        min_bounds, max_bounds = grid_bounds

        for x in range(min_bounds[0], max_bounds[0] + 1):
            for y in range(min_bounds[1], max_bounds[1] + 1):
                for z in range(min_bounds[2], max_bounds[2] + 1):
                    if (
                        x == min_bounds[0]
                        or x == max_bounds[0]
                        or y == min_bounds[1]
                        or y == max_bounds[1]
                        or z == min_bounds[2]
                        or z == max_bounds[2]
                    ):
                        queue.append((x, y, z))

        while queue:
            current = queue.popleft()

            if current in visited:
                continue
            visited.add(current)

            if all(min_bounds <= np.array(current)) and all(np.array(current) <= max_bounds):
                if current not in shell_cells:
                    external_cells.add(current)
                    for neighbor in neighbors:
                        neighbor_cell = tuple(np.array(current) + np.array(neighbor))
                        if neighbor_cell not in visited:
                            queue.append(neighbor_cell)

        return external_cells

    # Calculate grid bounds for flood fill
    min_grid = np.floor((min_bounds - min_bounds) / grid_resolution).astype(int)
    max_grid = np.ceil((max_bounds - min_bounds) / grid_resolution).astype(int)
    grid_bounds = (min_grid, max_grid)

    # Perform flood-fill from the outside
    external_cells = flood_fill_outside(unique_cells, grid_bounds)

    # Identify interior cells (all cells minus external and shell)
    all_grid_cells = set(
        (x, y, z)
        for x in range(min_grid[0], max_grid[0] + 1)
        for y in range(min_grid[1], max_grid[1] + 1)
        for z in range(min_grid[2], max_grid[2] + 1)
    )
    interior_cells = all_grid_cells - external_cells - unique_cells


    final_scene_objects = []

    # Add interior cells to the scene
    for cell in interior_cells:
        cube = o3d.geometry.TriangleMesh.create_box(width=grid_resolution,
                                                    height=grid_resolution,
                                                    depth=grid_resolution)
        cell_center = np.array(cell) * grid_resolution + min_bounds
        cube.translate(cell_center)
        cube.paint_uniform_color([0, 1, 0])
        cube.compute_vertex_normals()
        final_scene_objects.append(cube)

    # Add the original point cloud to the scene
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    final_scene_objects.append(pcd)

    # Visualise the combined result
    o3d.visualization.draw_geometries(final_scene_objects,
                                    window_name="Point Cloud with Interior Cells",
                                    mesh_show_wireframe=True)

    # Initialize lists to accumulate vertices and faces
    all_vertices = []
    all_faces = []
    vertex_offset = 0  # To keep track of the current vertex index

    corner_offsets = np.array([
        [-0.5, -0.5, -0.5],  # Vertex 0
        [ 0.5, -0.5, -0.5],  # Vertex 1
        [ 0.5,  0.5, -0.5],  # Vertex 2
        [-0.5,  0.5, -0.5],  # Vertex 3
        [-0.5, -0.5,  0.5],  # Vertex 4
        [ 0.5, -0.5,  0.5],  # Vertex 5
        [ 0.5,  0.5,  0.5],  # Vertex 6
        [-0.5,  0.5,  0.5],  # Vertex 7
    ]) * grid_resolution  # Scale by grid resolution


    face_indices = [
        [0, 1, 2, 3],  # Front face
        [4, 5, 6, 7],  # Back face
        [0, 1, 5, 4],  # Bottom face
        [2, 3, 7, 6],  # Top face
        [0, 3, 7, 4],  # Left face
        [1, 5, 6, 2],  # Right face
    ]


    for cell in interior_cells:
        cell_center = np.array(cell) * grid_resolution + min_bounds
        corners = cell_center + corner_offsets
        all_vertices.extend(corners.tolist())

        for face in face_indices:
            quad = [
                vertex_offset + face[0],
                vertex_offset + face[1],
                vertex_offset + face[2],
                vertex_offset + face[3],
            ]
            all_faces.append(quad)

        vertex_offset += 8

    all_vertices = np.array(all_vertices)
    all_faces = np.array(all_faces)

    quads = np.array([[all_vertices[vertex_idx] for vertex_idx in face] for face in all_faces])

    reshaped_quads = quads.reshape(-1, 12)
    sorted_quads = np.sort(reshaped_quads, axis=1)

    d = {}

    for i, quad in enumerate(sorted_quads):
        d[tuple(quad.tolist())] = 0

    for quad in sorted_quads:
        if tuple(quad.tolist()) in d:
            d[tuple(quad.tolist())] += 1

    valid = []
    for key, val in d.items():
        if val == 1:
            valid.append(quads[sorted_quads.tolist().index(list(key))])

    valid = np.array(valid).reshape(-1, 4, 3)

    return valid
