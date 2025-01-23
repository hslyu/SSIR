import random
from dataclasses import dataclass
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, box

import basestations as bs


class DataManager:
    def __init__(
        self,
        langitude_range: List[float],
        longitude_range: List[float],
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
        self.langitude_range = langitude_range
        self.longitude_range = longitude_range
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
            minx=langitude_range[0],
            miny=longitude_range[0],
            maxx=langitude_range[1],
            maxy=longitude_range[1],
        )
        self.bbox_gdf = gpd.GeoDataFrame({"geometry": [self.bbox]}, crs=target_crs)

        # Load the shapefiles for land, lakes, rivers, and maritime areas
        land = gpd.read_file(land_shp_path)
        lakes = gpd.read_file(lakes_shp_path)
        rivers = gpd.read_file(rivers_shp_path)
        maritime = gpd.read_file(maritime_shp_path)
        coastline = gpd.read_file(coastline_shp_path)

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
            self.gdf_list[0], 1
        )
        maritime_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[3], num_maritime_basestations
        )
        ground_basestations_points = self.generate_random_points_within_gdf(
            self.gdf_list[0], num_ground_basestations
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

    def generate_random_points_within_gdf(self, target_gdf, num_points):
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
            random_lon = random.uniform(minx, maxx)
            random_lat = random.uniform(miny, maxy)
            point = Point(random_lon, random_lat)
            # Check if the generated point is within the target GeoDataFrame
            if target_gdf.contains(point).any():
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
        # WARNING: BASESTATION order should be the same as the order of BaseStationType
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
        return graph


@dataclass
class PlotManager:
    # Set plotting parameters
    fontsize = 16
    markersize = 15
    linewidth = 1
    gdf_color_list = [
        np.array([240, 255, 240]) / 255.0,  # land
        np.array([212, 238, 255]) / 255.0,  # lakes
        np.array([212, 238, 255]) / 255.0,  # rivers
        np.array([212, 238, 255]) / 255.0,  # maritime
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
    ):
        fig, ax = plt.subplots(figsize=(12, 12))
        for gdf, color in zip(dm.gdf_list, self.gdf_color_list):
            if gdf.empty:
                continue
            gdf.plot(ax=ax, color=color, zorder=1)

        for i, node_gdf in enumerate(dm.node_gdf_list):
            node_gdf.plot(
                ax=ax,
                color=self.node_color_list[i],
                marker=self.node_marker_list[i],
                markersize=self.markersize if i != 0 else self.markersize * 10,
                label=self.node_label_list[i],
                zorder=2,
            )
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

        # Set the plot limits to the bounding box
        ax.set_xlim(*dm.langitude_range)
        ax.set_ylim(*dm.longitude_range)
        # Add title and legend
        ax.set_xlabel("Longitude", fontsize=self.fontsize)
        ax.set_ylabel("Latitude", fontsize=self.fontsize)
        ax.tick_params(axis="both", which="major", labelsize=self.fontsize)
        ax.legend(loc="lower left")


def main():
    import pathfinder as pf

    config = {
        "langitude_range": [-90, 10],
        "longitude_range": [20, 50],
        "num_maritime_basestations": 20,
        "num_ground_basestations": 25,
        "num_haps_basestations": 20,
        "num_leo_basestations": 15,
        "num_users": 10,
    }
    dm = DataManager(**config)
    pm = PlotManager()
    graph = dm.generate_master_graph()

    costs, predecessors = pf.a_star(graph, metric="distance")
    graph_astar_distance = pf.get_solution_graph(graph, predecessors)

    costs, predecessors = pf.a_star(graph, metric="hop")
    graph_astar_hop = pf.get_solution_graph(graph, predecessors)

    graph_list = [graph_astar_distance, graph_astar_hop]

    print(
        f"A* distance throughput: {graph_astar_distance.compute_network_throughput()}"
    )
    print(f"A* hop throughput: {graph_astar_hop.compute_network_throughput()}")


if __name__ == "__main__":
    main()
