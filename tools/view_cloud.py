import open3d as o3d
pcd = o3d.io.read_point_cloud("output/latest_cloud.pcd")
print("extent:", pcd.get_max_bound() - pcd.get_min_bound())
print("mean:", pcd.get_center())