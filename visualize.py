import open3d as o3d
import time


def visualize_with_open3d(vertices, faces):

        # o3d.visualization.webrtc_server.enable_webrtc()
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.3, 0.3, 0.3])

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        def visulaize_seq(vis):
            for i in range(len(vertices)):
                # Deform mesh vertices
                mesh.vertices = o3d.utility.Vector3dVector(vertices[i])
                vis.update_geometry(mesh)
                vis.update_renderer()
                vis.poll_events()
                # time.sleep(0.05)

        def end(vis):
            vis.destroy_window()
            exit()

        vis.register_key_callback(32, visulaize_seq) #space bar
        vis.register_key_callback(256, end)          #escape key
        vis.add_geometry(mesh)

        ctr = vis.get_view_control()
        ctr.set_lookat([0,0,0])
        ctr.set_front([0.5,-1.25,1])
        ctr.set_up([0,0,1])
        # ctr.set_zoom(0.5)
        
        vis.run() 
        vis.destroy_window()


