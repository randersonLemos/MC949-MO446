import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import os


def draw_camera_frustum(ax, R, t, scale=0.1, depth_factor=3.0, color='r'):
    origin = t.ravel()
    w, h = scale, scale
    d = scale * depth_factor
    corners = np.array([
        [ w,  h, d],
        [ w, -h, d],
        [-w, -h, d],
        [-w,  h, d],
    ]).T
    world_corners = R @ corners + origin.reshape(3,1)
    for i in range(4):
        ax.plot([origin[0], world_corners[0,i]],
                [origin[1], world_corners[1,i]],
                [origin[2], world_corners[2,i]], c=color)
    for i in range(4):
        ax.plot([world_corners[0,i], world_corners[0,(i+1)%4]],
                [world_corners[1,i], world_corners[1,(i+1)%4]],
                [world_corners[2,i], world_corners[2,(i+1)%4]], c=color)
    z_axis = R[:,2] * d
    ax.quiver(origin[0], origin[1], origin[2],
              z_axis[0], z_axis[1], z_axis[2],
              color=color, arrow_length_ratio=0.2, linewidth=2)

class Plot:
    _figures_to_show = []

    @classmethod
    def plot_images_grid(cls, images, nrows=1, ncols=1, figsize=(12, 8), save_path=None, show=False):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = [axes] if nrows * ncols == 1 else axes.flatten()
        for ax, img in zip(axes, images):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.axis("off")
        for ax in axes[len(images):]:
            ax.axis("off")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)  # saved-only figures are closed
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def plot_cameras_frustum(cls, camera_poses, points3d=None, points3d_color=None,
                             scale=0.33, points3d_size=2.5, save_path=None, show=False):
        """
        Plot or save 3D visualization of camera frustums and points.

        If `save_path` is provided, multiple static images will be saved.
        If `show=True`, an interactive 3D figure will be displayed.
        """

        # --- Internal helper for static image saving ---
        def save_camera_views():
            elevations = np.linspace(-165, 15, 45)
            azimuth = 0
            roll = -90

            def apply_roll(ax, roll_deg):
                """Apply roll to the 3D axes projection matrix (for static render)."""
                roll_rad = np.deg2rad(roll_deg)
                R = np.array([
                    [np.cos(roll_rad), -np.sin(roll_rad), 0, 0],
                    [np.sin(roll_rad), np.cos(roll_rad), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                proj = ax.get_proj()
                ax.get_proj = lambda: R @ proj

            base_dir, base_file = os.path.split(save_path)
            base_name, ext = os.path.splitext(base_file)
            os.makedirs(base_dir or ".", exist_ok=True)

            for i, elev in enumerate(elevations, 1):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                colors = plt.cm.get_cmap("tab10", len(camera_poses))

                # plot camera frustums
                for j, (R, C) in enumerate(camera_poses):
                    color = colors(j)
                    draw_camera_frustum(ax, R, C, scale=scale, color=color)
                    ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {j + 1}")

                # plot 3D points
                if points3d is not None:
                    if points3d_color is not None:
                        colors_norm = points3d_color.astype(np.float32) / 255.0
                        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                                   c=colors_norm, s=points3d_size)
                    else:
                        ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                                   c="g", s=points3d_size)

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_box_aspect([1, 1, 1])
                ax.view_init(elev=elev, azim=azimuth)
                apply_roll(ax, roll)  # safe for static image

                plt.savefig(os.path.join(base_dir, f"{base_name}_{i:02d}{ext}"))
                plt.close(fig)

        # --- Internal helper for interactive visualization ---
        def show_camera_scene():
            azimuth = 0
            roll = -90  # only affects orientation, not projection

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            colors = plt.cm.get_cmap("tab10", len(camera_poses))

            # plot camera frustums
            for j, (R, C) in enumerate(camera_poses):
                color = colors(j)
                draw_camera_frustum(ax, R, C, scale=scale, color=color)
                ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {j + 1}")

            # plot 3D points
            if points3d is not None:
                if points3d_color is not None:
                    colors_norm = points3d_color.astype(np.float32) / 255.0
                    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                               c=colors_norm, s=points3d_size)
                else:
                    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                               c="g", s=points3d_size)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=0, azim=azimuth)

            # Apply roll safely by rotating the entire scene orientation
            ax.view_init(elev=0, azim=azimuth)
            ax.set_title("Interactive 3D Camera View")

            plt.ion()  # Enable interactivity
            plt.show(block=True)

            cls._figures_to_show.append(fig)
            return fig

        # --- Dispatch to the appropriate behavior ---
        if save_path:
            save_camera_views()

        if show:
            return show_camera_scene()

    @classmethod
    def plot_cameras_surface(cls, camera_poses, points3d=None, points3d_color=None,
                             scale=0.33, save_path=None, show=False):
        """
        Plot or save a 3D visualization of camera frustums and a triangulated surface
        reconstructed from 3D points.

        If `save_path` is provided, multiple static images will be saved at different elevations.
        If `show=True`, an interactive 3D figure will be displayed.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import Delaunay
        import numpy as np
        import os

        # ------------------------------------------------------------
        # Helper to apply roll to the 3D axes projection (matches frustum version)
        # ------------------------------------------------------------
        def apply_roll(ax, roll_deg):
            """Apply roll to the 3D axes projection matrix (camera rotation)."""
            roll_rad = np.deg2rad(roll_deg)
            R = np.array([
                [np.cos(roll_rad), -np.sin(roll_rad), 0, 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            proj = ax.get_proj()
            ax.get_proj = lambda: R @ proj

        # ------------------------------------------------------------
        # Save multiple surface views as static images
        # ------------------------------------------------------------
        def save_camera_surface_views():
            elevations = np.linspace(-165, 15, 45)
            azimuth = 0
            roll = -90

            base_dir, base_file = os.path.split(save_path)
            base_name, ext = os.path.splitext(base_file)
            os.makedirs(base_dir or ".", exist_ok=True)

            for i, elev in enumerate(elevations, 1):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                colors = plt.cm.get_cmap("tab10", len(camera_poses))

                # Plot cameras
                for j, (R, C) in enumerate(camera_poses):
                    color = colors(j)
                    draw_camera_frustum(ax, R, C, scale=scale, color=color)
                    ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {j + 1}")

                # Plot surface
                if points3d is not None and len(points3d) >= 3:
                    tri = Delaunay(points3d[:, :2])
                    face_color = "lightblue"
                    if points3d_color is not None and len(points3d_color) == len(points3d):
                        colors_norm = points3d_color.astype(np.float32) / 255.0
                        face_color = np.mean(colors_norm[tri.simplices], axis=1)
                    ax.plot_trisurf(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                                    triangles=tri.simplices, facecolor=face_color,
                                    linewidth=0.2, alpha=0.9)

                # Axes & view configuration
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_box_aspect([1, 1, 1])
                ax.view_init(elev=elev, azim=azimuth)
                apply_roll(ax, roll)  # ✅ ensure same camera roll as frustum version
                ax.legend()

                # Save image
                out_path = os.path.join(base_dir, f"{base_name}_{i:02d}{ext}")
                plt.savefig(out_path)
                plt.close(fig)

        # ------------------------------------------------------------
        # Interactive visualization
        # ------------------------------------------------------------
        def show_camera_surface_plot():
            azimuth = 0
            roll = -90

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            colors = plt.cm.get_cmap("tab10", len(camera_poses))

            # Plot cameras
            for i, (R, C) in enumerate(camera_poses):
                color = colors(i)
                draw_camera_frustum(ax, R, C, scale=scale, color=color)
                ax.scatter([], [], [], c=[color], marker="o", label=f"Camera {i + 1}")

            # Plot surface
            if points3d is not None and len(points3d) >= 3:
                tri = Delaunay(points3d[:, :2])
                face_color = "lightblue"
                if points3d_color is not None and len(points3d_color) == len(points3d):
                    colors_norm = points3d_color.astype(np.float32) / 255.0
                    face_color = np.mean(colors_norm[tri.simplices], axis=1)
                ax.plot_trisurf(points3d[:, 0], points3d[:, 1], points3d[:, 2],
                                triangles=tri.simplices, facecolor=face_color,
                                linewidth=0.2, alpha=0.9)

            # Axes setup
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])
            ax.legend()
            ax.view_init(elev=0, azim=azimuth)
            apply_roll(ax, roll)
            ax.set_title("Interactive 3D Camera Surface View")

            plt.ion()
            plt.show(block=True)

            cls._figures_to_show.append(fig)
            return fig

        # ------------------------------------------------------------
        # Dispatcher: decide between save and show
        # ------------------------------------------------------------
        if save_path:
            save_camera_surface_views()
        if show:
            fig = show_camera_surface_plot()
            return fig
        return None

    @classmethod
    def plot_cameras_surface_grid(cls, camera_poses, points3d, points3d_color=None,
                                  grid_size=100, alpha=0.9, save_path=None, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.get_cmap('tab10', len(camera_poses))
        for i, (R, C) in enumerate(camera_poses):
            color = colors(i)
            draw_camera_frustum(ax, R, C, scale=0.33, color=color)
            ax.scatter([], [], [], c=[color], marker='o', label=f'Camera {i+1}')
        x, y, z = points3d[:,0], points3d[:,1], points3d[:,2]
        grid_x, grid_y = np.meshgrid(np.linspace(x.min(), x.max(), grid_size),
                                     np.linspace(y.min(), y.max(), grid_size))
        grid_z = griddata((x,y), z, (grid_x, grid_y), method='cubic')
        ax.plot_surface(grid_x, grid_y, grid_z, alpha=alpha, cmap='viridis')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,1])
        ax.legend()
        ax.view_init(elev=20, azim=-60)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        if show:
            cls._figures_to_show.append(fig)
        return fig

    @classmethod
    def show_poisson_surface_plot(cls, camera_poses, points3d, points3d_color=None,
                                  save_path=None, show=False, scale_cameras=0.33):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import open3d as o3d

        # ---------------------- Poisson reconstruction ----------------------
        def reconstruct_poisson(points3d, points3d_color=None):
            pcd = o3d.geometry.PointCloud()
            points3d = np.asarray(points3d)
            pcd.points = o3d.utility.Vector3dVector(points3d)
            if points3d_color is not None:
                colors = np.asarray(points3d_color, dtype=np.float32) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=4)
            mesh.compute_vertex_normals()
            densities = np.asarray(densities)
            mask = densities >= np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(~mask)

            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            if len(mesh.vertex_colors) > 0:
                vertex_colors = np.asarray(mesh.vertex_colors)
                face_colors = vertex_colors[triangles].mean(axis=1)
            else:
                face_colors = np.ones((len(triangles), 3)) * [0.6, 0.8, 1.0]
            return vertices, triangles, face_colors

        # ---------------------- Plot helper ----------------------
        def plot_mesh(ax, vertices, triangles, face_colors):
            mesh_collection = Poly3DCollection(
                vertices[triangles],
                facecolors=face_colors,
                edgecolor='gray',
                linewidths=0.2
            )
            ax.add_collection3d(mesh_collection)

            # Plot camera centers
            if camera_poses is not None:
                colors = plt.cm.get_cmap('tab10', len(camera_poses))
                for i, (R, C) in enumerate(camera_poses):
                    ax.scatter(C[0], C[1], C[2], color=colors(i), s=30, label=f'Camera {i + 1}')

            # Center and scale
            max_range = np.ptp(vertices, axis=0).max() / 2.0
            mid = vertices.mean(axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1, 1, 1])
            ax.legend()

        # ---------------------- Apply roll helper ----------------------
        def apply_roll(ax, roll_deg):
            import numpy as np
            roll_rad = np.deg2rad(roll_deg)
            R = np.array([
                [np.cos(roll_rad), -np.sin(roll_rad), 0, 0],
                [np.sin(roll_rad), np.cos(roll_rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            proj = ax.get_proj()
            ax.get_proj = lambda: R @ proj

        # ---------------------- Save multiple views ----------------------
        def save_views(vertices, triangles, face_colors):
            elevations = np.linspace(-165, 15, 45)
            azimuth = 0
            roll = -90

            base_dir, base_file = os.path.split(save_path)
            base_name, ext = os.path.splitext(base_file)
            os.makedirs(base_dir or ".", exist_ok=True)

            for i, elev in enumerate(elevations, 1):
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                plot_mesh(ax, vertices, triangles, face_colors)
                ax.view_init(elev=elev, azim=azimuth)
                apply_roll(ax, roll)
                out_path = os.path.join(base_dir, f"{base_name}_{i:02d}{ext}")
                plt.savefig(out_path)
                plt.close(fig)

        # ---------------------- Interactive view ----------------------
        def show_interactive(vertices, triangles, face_colors):
            azimuth = 0
            roll = -90

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            plot_mesh(ax, vertices, triangles, face_colors)
            ax.view_init(elev=0, azim=azimuth)
            apply_roll(ax, roll)
            ax.set_title("Interactive 3D Poisson Surface")

            plt.ion()
            plt.show(block=True)
            cls._figures_to_show.append(fig)
            return fig

        # ---------------------- Main ----------------------
        vertices, triangles, face_colors = reconstruct_poisson(points3d, points3d_color)

        if save_path:
            save_views(vertices, triangles, face_colors)
        if show:
            fig = show_interactive(vertices, triangles, face_colors)
            return fig
        return None


    @classmethod
    def show(cls):
        """Show only figures passed with show=True. Blocks execution for interactivity."""
        if cls._figures_to_show:
            plt.show()  # This will block and allow interactive zoom/rotate
            cls._figures_to_show.clear()