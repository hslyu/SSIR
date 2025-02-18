import os
import random
from dataclasses import dataclass
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely.vectorized as sv
from shapely.geometry import Point, box
from shapely.ops import nearest_points

from ssir import basestations as bs
from ssir import pathfinder as pf

# Predefined maps
map_list = [
    {
        "latitude_range": [25, 55],
        "longitude_range": [-15, 65],
    },
    {
        "latitude_range": [-18, 12],
        "longitude_range": [90, 150],
    },
    {
        "latitude_range": [10, 40],
        "longitude_range": [100, 150],
    },
    {
        "latitude_range": [-8, 22],
        "longitude_range": [35, 105],
    },
    {
        "latitude_range": [22, 52],
        "longitude_range": [-83, 2],
    },
]


def generate_config(exp_index):
    random.seed(exp_index)
    config = {
        "num_maritime_basestations": random.randint(40, 50),
        "num_ground_basestations": random.randint(60, 80),
        "num_haps_basestations": random.randint(15, 20),
        "num_leo_basestations": random.randint(10, 15),
        "num_users": random.randint(40, 60),
        "random_seed": exp_index,
    }
    config.update(map_list[exp_index % len(map_list)])
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
        self.bbox_gdf = gpd.GeoDataFrame({"geometry": [self.bbox]}, crs=target_crs)

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
        source_basestation_point = self.generate_random_points_within_gdf(
            self.gdf_list[0], 1, near_coastline=True, offset_length=3, source=True
        )
        maritime_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[3], num_maritime_basestations
        )
        ground_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[0], num_ground_basestations, near_coastline=True
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
            gpd.GeoDataFrame({"geometry": node_points}, crs=target_crs)
            for node_points in node_point_list
        ]

    def generate_random_points_within_gdf(
        self,
        target_gdf,
        num_points,
        near_coastline=False,
        offset_length=1,
        source=False,
    ):
        """
        Generate random points within a given GeoDataFrame.

        Parameters:
            target_gdf (GeoDataFrame): GeoDataFrame of the area of interest.
            num_points (int): Number of random points to generate.

        Returns:
            List[Point]: List of generated random points.
        """
        minx, miny, maxx, maxy = target_gdf.total_bounds

        points = []
        attempts = 0
        max_attempts = num_points * 50  # Maximum number of attempts to generate points

        while len(points) < num_points and attempts < max_attempts:
            # Generate random points within the bounding box
            random_lon = random.uniform(minx * 1.01, maxx * 0.99)
            if source:
                random_lat = miny + (maxy - miny) * random.uniform(0.05, 0.95)
            else:
                random_lat = random.uniform(miny * 1.01, maxy * 0.99)
            point = Point(random_lon, random_lat)
            if target_gdf.contains(point).any():
                if near_coastline:
                    # Find the nearest point on the coastline
                    nearest_geometry = nearest_points(
                        point, self.gdf_list[-1].unary_union
                    )[1]

                    # Generate a random offset from the nearest point
                    offset_lat = random.uniform(-offset_length, offset_length)
                    offset_lon = random.uniform(-offset_length, offset_length)
                    new_point = Point(
                        nearest_geometry.x + offset_lon, nearest_geometry.y + offset_lat
                    )
                    if target_gdf.contains(new_point).any():
                        points.append(new_point)
                    else:
                        new_point = Point(
                            nearest_geometry.x - offset_lon,
                            nearest_geometry.y - offset_lat,
                        )
                        if target_gdf.contains(new_point).any():
                            points.append(new_point)
                else:
                    points.append(point)
            attempts += 1

        if len(points) < num_points:
            print(
                f"Warning: Only {len(points)} points were generated after {attempts} attempts."
            )

        return points

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
        # TODO(?): distinguish ground user, maritime user, and aeiral users
        for i, point in enumerate(user_gdf_list["geometry"]):
            node = bs.User(node_id, np.array([point.x, point.y, 0]), isGeographic=True)
            graph.add_node(node)
            node_id += 1

        graph.connect_reachable_nodes()
        costs, predecessors = pf.astar.a_star(graph, metric="hop")
        for user in graph.users:
            path = pf.astar.get_shortest_path(predecessors, user.get_id())
            if path[0] == -1:
                while True:
                    user._position = np.array(
                        [
                            random.uniform(*self.longitude_range),
                            random.uniform(*self.latitude_range),
                            0,
                        ]
                    )
                    graph.connect_reachable_nodes()
                    if user.has_parent():
                        break
                # graph.remove_node(user.get_id())

        source = graph.nodes[0]
        while len(source.get_children()) < 3:
            source_basestation_point = self.generate_random_points_within_gdf(
                self.gdf_list[0], 1, near_coastline=True, offset_length=3, source=True
            )
            source._position = np.array(
                [source_basestation_point[0].x, source_basestation_point[0].y, 0]
            )
            graph.connect_reachable_nodes(target_node_id=0)

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

        return graph


@dataclass
class PlotManager:
    # Set plotting parameters
    fontsize = 16
    markersize = 15
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
    ):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set the plot limits to the bounding box
        ax.set_xlim(*dm.longitude_range)
        ax.set_ylim(*dm.latitude_range)
        # Add title and legend
        ax.set_xlabel("Longitude", fontsize=self.fontsize)
        ax.set_ylabel("Latitude", fontsize=self.fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.fontsize)

        # Plot the geographical data
        for gdf, color in zip(dm.gdf_list, self.gdf_color_list):
            if gdf.empty:
                continue
            gdf.plot(ax=ax, color=color, linewidth=0, zorder=1)

        node_id = 0
        for i, node_gdf in enumerate(dm.node_gdf_list):
            color = self.node_color_list[i]
            node_gdf.plot(
                ax=ax,
                color=color,
                marker=self.node_marker_list[i],
                markersize=self.markersize if i != 0 else self.markersize * 10,
                label=self.node_label_list[i],
                zorder=2,
            )

            # print node id on the plot
            if verbose:
                for idx, row in node_gdf.iterrows():
                    ax.text(
                        row["geometry"].x - 1.1,
                        row["geometry"].y - 1.6,
                        f"{node_id}",
                        fontsize=12,
                    )
                    node_id += 1

        if graph_list is None:
            return

        if isinstance(graph_list, bs.IABRelayGraph):
            graph_list = [graph_list]

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
        ax.legend(loc="lower left")


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
