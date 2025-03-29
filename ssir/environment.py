import os
import random
from dataclasses import dataclass
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

from ssir import basestations as bs
from ssir import pathfinder as pf

# Predefined maps
map_list = [
    {
        "longitude_range": [15, 45],
        "latitude_range": [25, 45],
    },
    {
        "longitude_range": [118, 148],
        "latitude_range": [-20, 0],
    },
    {
        "longitude_range": [110, 140],
        "latitude_range": [23, 43],
    },
    {
        "longitude_range": [75, 105],
        "latitude_range": [2, 22],
    },
    {
        "longitude_range": [-85, -55],
        "latitude_range": [35, 55],
    },
    {
        "longitude_range": [35, 65],
        "latitude_range": [10, 30],
    },
    {
        "longitude_range": [-10, 20],
        "latitude_range": [45, 65],
    },
]


def generate_config(exp_index):
    map = map_list[exp_index % len(map_list)]
    area_size = (map["latitude_range"][1] - map["latitude_range"][0]) * (
        map["longitude_range"][1] - map["longitude_range"][0]
    )
    config = {
        "num_maritime_basestations": int(area_size / 8),
        "num_ground_basestations": int(area_size / 8),
        "num_haps_basestations": int(area_size / 70),
        "num_leo_basestations": int(area_size / 80),
        "num_users": int(area_size / 10),
        "random_seed": exp_index,
    }
    config.update(map)
    return config


class DataManager:
    def __init__(
        self,
        longitude_range: List[float],
        latitude_range: List[float],
        land_shp_path: str = "map/ne_10m_land/ne_10m_land.shp",
        lakes_shp_path: str = "map/ne_10m_lakes/ne_10m_lakes.shp",
        rivers_shp_path: str = "map/ne_10m_rivers_lake_centerlines/ne_10m_rivers_lake_centerlines.shp",
        maritime_shp_path: str = "map/ne_10m_ocean/ne_10m_ocean.shp",
        coastline_shp_path: str = "map/ne_10m_coastline/ne_10m_coastline.shp",
        target_crs: str = "EPSG:4326",
        num_maritime_basestations: int = 20,
        num_ground_basestations: int = 25,
        num_haps_basestations: int = 20,
        num_leo_basestations: int = 15,
        num_users: int = 30,
        random_seed: int = 42,
        **kwargs,
    ):
        # if master_dir arguemnt is provided, use the master_dir to load the shapefiles
        if "master_dir" in kwargs:
            master_dir = kwargs["master_dir"]
        else:
            master_dir = "."
        self.longitude_range = longitude_range
        self.latitude_range = latitude_range
        self.land_shp_path = land_shp_path
        self.lakes_shp_path = lakes_shp_path
        self.rivers_shp_path = rivers_shp_path
        self.maritime_shp_path = maritime_shp_path
        self.target_crs = target_crs
        self.num_maritime_basestations = num_maritime_basestations
        self.num_ground_basestations = num_ground_basestations
        self.num_haps_basestations = num_haps_basestations
        self.num_leo_basestations = num_leo_basestations
        self.num_users = num_users
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        ####################
        # Generate the geographical data
        ####################
        # Generate a bounding box for the area of interest
        self.bbox = box(
            minx=longitude_range[0],
            miny=latitude_range[0],
            maxx=longitude_range[1],
            maxy=latitude_range[1],
        )
        self.bbox_gdf = gpd.GeoDataFrame(geometry=[self.bbox], crs=target_crs)

        # Load the shapefiles for land, lakes, rivers, and maritime areas
        land = gpd.read_file(os.path.join(master_dir, land_shp_path))
        lakes = gpd.read_file(os.path.join(master_dir, lakes_shp_path))
        rivers = gpd.read_file(os.path.join(master_dir, rivers_shp_path))
        maritime = gpd.read_file(os.path.join(master_dir, maritime_shp_path))
        coastline = gpd.read_file(os.path.join(master_dir, coastline_shp_path))

        self.gdf_list = [land, lakes, rivers, maritime, coastline]
        for gdf in self.gdf_list:
            if gdf.crs != target_crs:
                gdf.to_crs(target_crs, inplace=True)
        self.gdf_list = [self.clip_to_bbox(gdf, self.bbox_gdf) for gdf in self.gdf_list]

        ####################
        # Generate the user and basestation data
        ####################
        # Filter the data to the bounding box
        source_basestation_point = self.generate_source_point(self.gdf_list[0], 2)
        maritime_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[3], num_maritime_basestations
        )
        lakes_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[1], int(num_maritime_basestations * 0.2)
        )
        # merge maritime and lakes basestations and sample with num_maritime_basestations
        maritime_basestations_points = (
            maritime_basestations_points + lakes_basestations_points
        )
        maritime_basestations_points = random.sample(
            maritime_basestations_points, num_maritime_basestations
        )
        ground_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[0],
            num_ground_basestations,
        )
        haps_basestations_points = self.generate_random_points_within_gdf(
            self.bbox_gdf, num_haps_basestations
        )
        leo_basestations_points = self.generate_random_points_within_gdf(
            self.bbox_gdf, num_leo_basestations
        )
        users_points = self.generate_random_points_within_gdf(self.bbox_gdf, num_users)
        node_point_list = [
            source_basestation_point,
            maritime_basestations_points,
            ground_basestations_points,
            haps_basestations_points,
            leo_basestations_points,
            users_points,
        ]

        self.node_gdf_list = [
            gpd.GeoDataFrame(geometry=node_points, crs=target_crs)
            for node_points in node_point_list
        ]

    def generate_random_points_within_gdf(
        self,
        target_gdf,
        num_points,
    ):
        """
        Generate random points within a given GeoDataFrame.

        Parameters:
            target_gdf (GeoDataFrame): GeoDataFrame of the area of interest.
            num_points (int): Number of random points to generate.

        Returns:
            List[Point]: List of generated random points.
        """
        # Prepare the target geometry for faster contains check
        merged_poly = unary_union(target_gdf.geometry)
        prepared_poly = prep(merged_poly)

        # Generate random points within the bounding box
        minx, miny, maxx, maxy = target_gdf.total_bounds
        factor = 10
        xs = np.random.uniform(minx, maxx, factor * num_points)
        ys = np.random.uniform(miny, maxy, factor * num_points)
        candidate_points = [Point(x, y) for x, y in zip(xs, ys)]
        inside_points = [
            point for point in candidate_points if prepared_poly.contains(point)
        ]
        # Return num_points if there are enough points inside the target_gdf
        if len(inside_points) >= num_points:
            return random.sample(inside_points, num_points)
        # Return all the points if there are not enough points inside the target_gdf
        else:
            return inside_points

    def generate_source_point(self, target_gdf, offset_length=3):
        # Prepare the target geometry for faster contains check
        merged_poly = unary_union(target_gdf.geometry)
        prepared_poly = prep(merged_poly)

        minx, miny, maxx, maxy = target_gdf.total_bounds

        xs = np.random.uniform(minx, maxx, 30)
        ys = np.random.uniform(miny, maxy, 30)
        points = [Point(x, y) for x, y in zip(xs, ys)]

        # Find the nearest point on the coastline
        near_coastline_points = [
            nearest_points(point, unary_union(self.gdf_list[-1].geometry))[1]
            for point in points
        ]

        candidate_points = []
        for point_geometry in near_coastline_points:
            x, y = point_geometry.x, point_geometry.y
            # Generate a random offset from the nearest point
            offset_lat = random.uniform(1, offset_length)
            offset_lon = random.uniform(1, offset_length)

            point_list = [
                Point(x + offset_lon, y),
                Point(x - offset_lon, y),
                Point(x, y + offset_lat),
                Point(x, y - offset_lat),
                Point(x + offset_lon, y + offset_lat),
                Point(x - offset_lon, y - offset_lat),
                Point(x + offset_lon, y - offset_lat),
                Point(x - offset_lon, y + offset_lat),
            ]
            candidate_points.extend(point_list)
        inside_points = [
            point for point in candidate_points if prepared_poly.contains(point)
        ]

        return random.sample(inside_points, 1)

    def clip_to_bbox(self, gdf, bbox):
        """

        Cut data to the bounding box using the clip function of GeoPandas
        """
        return gpd.clip(gdf, bbox)

    def generate_master_graph(self) -> bs.IABRelayGraph:
        graph = bs.IABRelayGraph(bs.environmental_variables)

        basestations_gdf_list = self.node_gdf_list[:-1]
        user_gdf_list = self.node_gdf_list[-1]
        basestations_type_list = [
            bs.BaseStationType.GROUND,  # Source
            bs.BaseStationType.MARITIME,
            bs.BaseStationType.GROUND,
            bs.BaseStationType.HAPS,
            bs.BaseStationType.LEO,
        ]
        basestations_altitude_list = [
            bs.environmental_variables.ground_basestations_altitude,
            bs.environmental_variables.maritime_basestations_altitude,
            bs.environmental_variables.ground_basestations_altitude,
            bs.environmental_variables.haps_basestations_altitude,
            bs.environmental_variables.leo_basestations_altitude,
        ]

        # Add basestations nodes
        node_id = 0
        for i, basestations_gdf in enumerate(basestations_gdf_list):
            for point in basestations_gdf["geometry"]:
                node = bs.BaseStation(
                    node_id,
                    np.array([point.x, point.y, basestations_altitude_list[i]]),
                    basestations_type_list[i],
                    isGeographic=True,
                )
                graph.add_node(node)
                node_id += 1

        # Add users nodes
        for i, point in enumerate(user_gdf_list["geometry"]):
            node = bs.User(node_id, np.array([point.x, point.y, 0]), isGeographic=True)
            graph.add_node(node)
            node_id += 1

        # Change the location of source node connected with at least 5 nodes.
        source = graph.nodes[0]
        while len(source.get_children()) < 5:
            # Reset the source graph edges
            graph.adjacency_list[0] = []
            source._children = []

            source_basestation_point = self.generate_source_point(self.gdf_list[0], 0)
            self.node_gdf_list[0] = gpd.GeoDataFrame(
                geometry=source_basestation_point, crs=self.target_crs
            )
            source._position = np.array(
                [source_basestation_point[0].x, source_basestation_point[0].y, 0]
            )
            graph.connect_reachable_nodes(target_node_id=0)

        graph.connect_reachable_nodes()
        costs, predecessors = pf.astar.a_star(graph, metric="distance")
        disconnected_uid_list = []
        for user in graph.users:
            path = pf.astar.get_shortest_path(predecessors, user.get_id())
            if path[0] == -1:
                disconnected_uid_list.append(user.get_id())

        for uid in disconnected_uid_list:
            graph.remove_node(uid)

        # Rearrange node id to be consecutive
        graph.reset()
        all_nodes = graph.nodes.values()
        graph.nodes = {}
        graph.users = []
        graph.basestations = []
        for i, node in enumerate(all_nodes):
            node._node_id = i
            graph.add_node(node)
        graph.connect_reachable_nodes()

        user_point_list = [user.get_position()[:2] for user in graph.users]
        self.node_gdf_list[-1] = gpd.GeoDataFrame(
            geometry=[Point(*point) for point in user_point_list],
            crs=self.target_crs,
        )

        return graph


@dataclass
class PlotManager:
    # Set plotting parameters
    fontsize = 16
    markersize = 5
    linewidth = 1
    gdf_color_list = [
        np.array([230, 255, 230]) / 255.0,  # land
        np.array([122, 213, 255]) / 255.0,  # lakes
        np.array([122, 213, 255]) / 255.0,  # rivers
        np.array([122, 213, 255]) / 255.0,  # maritime
        np.array([0, 0, 0]) / 255.0,  # coastline
    ]
    node_color_list = [
        "deeppink",  # source basestation
        "blue",  # maritime basestation
        "teal",  # ground basestation
        "dodgerblue",  # haps basestation
        "navy",  # leo basestation
        "red",  # users
    ]
    node_marker_list = ["*", "o", "^", "x", "s", "d"]
    node_label_list = [
        "Source Basestation",
        "Maritime Basestation",
        "Ground Basestation",
        "HAPS Basestation",
        "LEO Basestation",
        "Users",
    ]

    def plot_dm(
        self,
        dm: DataManager,
        graph_list: Optional[Union[bs.IABRelayGraph, List[bs.IABRelayGraph]]] = None,
        verbose: bool = False,
        verbose_id: int | List[int] | None = None,
        legend: bool = True,
    ):
        # Set the plotting parameters
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set the plot limits to the bounding box
        ax.set_xlim(*dm.longitude_range)
        ax.set_ylim(*dm.latitude_range)
        # Add title and legend
        ax.set_xlabel("Longitude", fontsize=self.fontsize)
        ax.set_ylabel("Latitude", fontsize=self.fontsize)
        ax.set_xticks(np.arange(dm.longitude_range[0], dm.longitude_range[1] + 1, 10))
        ax.set_yticks(np.arange(dm.latitude_range[0], dm.latitude_range[1] + 1, 5))

        # Check if the graph_list is a single graph or a list of graphs
        if isinstance(graph_list, bs.IABRelayGraph):
            graph_list = [graph_list]

        if isinstance(verbose_id, int):
            verbose_id = [verbose_id]

        # Select the nodes with parents or children
        # Users are added regardless of the connection to check the runtime error
        selected_node_ids = set()
        if graph_list is not None:
            for graph in graph_list:
                for node in graph.basestations:
                    if node.has_parent() or node.has_children():
                        selected_node_ids.add(node.get_id())
            for node in graph_list[0].users:
                selected_node_ids.add(node.get_id())

        # Plot the geographical data
        for gdf, color in zip(dm.gdf_list, self.gdf_color_list):
            if gdf.empty:
                continue
            gdf.plot(ax=ax, color=color, linewidth=0, zorder=1)

        node_id = 0
        for i, node_gdf in enumerate(dm.node_gdf_list):
            for idx, row in node_gdf.iterrows():
                if graph_list is None:
                    color = self.node_color_list[i]
                    markersize = (
                        self.markersize if node_id != 0 else self.markersize * 3
                    )
                else:
                    color = (
                        self.node_color_list[i]
                        if node_id in selected_node_ids
                        else "lightgray"
                    )
                    if node_id in selected_node_ids:
                        markersize = (
                            self.markersize if node_id != 0 else self.markersize * 3
                        )
                    else:
                        markersize = 3
                marker = self.node_marker_list[i]
                label = self.node_label_list[i] if idx == 0 else None
                ax.plot(
                    row["geometry"].x,  # type: ignore
                    row["geometry"].y,  # type: ignore
                    color=color,
                    marker=marker,
                    label=label,
                    linestyle="None",
                    markersize=markersize,
                    zorder=2,
                )

                if verbose and (verbose_id is None or node_id in verbose_id):
                    ax.text(
                        row["geometry"].x,  # type: ignore
                        row["geometry"].y,  # type: ignore
                        f"{node_id}",
                        fontsize=12,
                    )
                node_id += 1

        if graph_list is not None:
            color_list = ["black", "red", "blue", "green", "purple", "orange"]
            for i, graph in enumerate(graph_list):
                for edge in graph.edges:
                    start = graph.nodes[edge[0]].get_position()
                    end = graph.nodes[edge[1]].get_position()
                    plt.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        color=color_list[i],
                        linestyle="--",
                        linewidth=self.linewidth,
                    )
        if legend:
            ax.legend(loc="lower right", fontsize=8)


def main():
    import pathfinder as pf

    config = {
        "longitude_range": [-90, 10],
        "latitude_range": [20, 50],
        "num_maritime_basestations": 20,
        "num_ground_basestations": 25,
        "num_haps_basestations": 20,
        "num_leo_basestations": 15,
        "num_users": 10,
    }
    dm = DataManager(**config, master_dir="/home/hslyu/research/SSIR/scripts/")
    pm = PlotManager()
    graph = dm.generate_master_graph()

    c = dm.gdf_list[-1]

    # costs, predecessors = pf.astar.a_star(graph, metric="distance")
    # graph_astar_distance = pf.astar.get_solution_graph(graph, predecessors)
    #
    # costs, predecessors = pf.astar.a_star(graph, metric="hop")
    # graph_astar_hop = pf.astar.get_solution_graph(graph, predecessors)
    #
    # graph_list = [graph_astar_distance, graph_astar_hop]
    #
    # print(
    #     f"A* distance throughput: {graph_astar_distance.compute_network_throughput()}"
    # )
    # print(f"A* hop throughput: {graph_astar_hop.compute_network_throughput()}")


if __name__ == "__main__":
    main()
