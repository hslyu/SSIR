import csv
import datetime as dt
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple, Union, Dict

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from matplotlib.transforms import Affine2D
from matplotlib.path import Path 

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
        "latitude_range": [35, 55],
    },
]


def generate_config(exp_index):
    map = map_list[exp_index % len(map_list)]
    area_size = (map["latitude_range"][1] - map["latitude_range"][0]) * (
        map["longitude_range"][1] - map["longitude_range"][0]
    )
    config = {
        "num_maritime_basestations": int(area_size / 4),
        "num_ground_basestations": int(area_size / 4),
        "num_haps_basestations": int(area_size / 50),
        "num_leo_basestations": int(area_size / 60),
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
        # Optional coordinate lists (lat, lon[, alt])
        source_coords: Optional[List[Tuple[float, float, float]]] = None,
        ground_coords: Optional[List[Tuple[float, float, float]]] = None,
        maritime_coords: Optional[List[Tuple[float, float, float]]] = None,
        haps_coords: Optional[List[Tuple[float, float, float]]] = None,
        leo_coords: Optional[List[Tuple[float, float, float]]] = None,
        user_coords: Optional[List[Tuple[float, float, float]]] = None,
        # File paths and CRS
        land_shp_path: str = "map/ne_10m_land/ne_10m_land.shp",
        lakes_shp_path: str = "map/ne_10m_lakes/ne_10m_lakes.shp",
        rivers_shp_path: str = "map/ne_10m_rivers_lake_centerlines/ne_10m_rivers_lake_centerlines.shp",
        maritime_shp_path: str = "map/ne_10m_ocean/ne_10m_ocean.shp",
        coastline_shp_path: str = "map/ne_10m_coastline/ne_10m_coastline.shp",
        target_crs: str = "EPSG:4326",
        # Numbers of random instances (ignored if coords list given)
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

        # ── Helper: build point/alt lists ──────────────────────────────────
        def _prepare(
            coords: Optional[List[Tuple[float, float, float]]],
            default_alt: float,
            generator,
            *args,
        ):
            """
            Return (points, altitude_list).
            If coords is None → use generator to make points and uniform alt list.
            If coords is provided → convert to Point objects and extract per‑node alt.
            """
            if coords is not None:
                pts, alts = [], []
                for tup in coords:
                    if len(tup) == 3:
                        lat, lon, alt = tup
                    elif len(tup) == 2:
                        lat, lon = tup
                        alt = default_alt
                    else:
                        raise ValueError("Each coordinate must be (lat, lon[, alt]).")
                    pt = Point(lon, lat)
                    if self.bbox.contains(pt):
                        pts.append(pt)
                        alts.append(alt)
                return pts, alts
            # Random generation path
            pts = generator(*args)
            alts = [default_alt] * len(pts)
            return pts, alts

        env = bs.environmental_variables
        # Prepare every category
        source_pts, source_alts = _prepare(
            source_coords,
            env.ground_basestations_altitude,
            self.generate_source_point,
            self.gdf_list[0],
            2,
        )
        maritime_pts, maritime_alts = _prepare(
            maritime_coords,
            env.maritime_basestations_altitude,
            self.generate_random_points_within_gdf,
            self.gdf_list[3],
            num_maritime_basestations,
        )
        # 20% of maritime BSs can be on lakes (only when random)
        if maritime_coords is None:
            lakes_pts = self.generate_random_points_within_gdf(
                self.gdf_list[1],
                int(num_maritime_basestations * 0.2),
            )
            maritime_pts += lakes_pts
            maritime_alts += [env.maritime_basestations_altitude] * len(lakes_pts)
            maritime_samples = random.sample(
                list(zip(maritime_pts, maritime_alts)),
                min(num_maritime_basestations, len(maritime_pts)),
            )
            maritime_pts, maritime_alts = map(list, zip(*maritime_samples))

        ground_pts, ground_alts = _prepare(
            ground_coords,
            env.ground_basestations_altitude,
            self.generate_random_points_within_gdf,
            self.gdf_list[0],
            num_ground_basestations,
        )

        haps_pts, haps_alts = _prepare(
            haps_coords,
            env.haps_basestations_altitude,
            self.generate_random_points_within_gdf,
            self.bbox_gdf,
            num_haps_basestations,
        )
        leo_pts, leo_alts = _prepare(
            leo_coords,
            env.leo_basestations_altitude,
            self.generate_random_points_within_gdf,
            self.bbox_gdf,
            num_leo_basestations,
        )
        users_pts, users_alts = _prepare(
            user_coords,
            0.0,
            self.generate_random_points_within_gdf,
            self.bbox_gdf,
            num_users,
            50,
        )

        # Store for later use
        self.node_points = [
            source_pts,
            maritime_pts,
            ground_pts,
            haps_pts,
            leo_pts,
            users_pts,
        ]
        self.node_alts = [
            source_alts,
            maritime_alts,
            ground_alts,
            haps_alts,
            leo_alts,
            users_alts,
        ]
        self.node_gdf_list = [
            gpd.GeoDataFrame(geometry=pts, crs=target_crs) for pts in self.node_points
        ]

    def generate_random_points_within_gdf(
        self,
        target_gdf,
        num_points,
        factor=10,
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

    def generate_source_point(self, target_gdf, offset_length=5):
        # Prepare the target geometry for faster contains check
        merged_poly = unary_union(target_gdf.geometry)
        prepared_poly = prep(merged_poly)

        minx, miny, maxx, maxy = target_gdf.total_bounds
        xrange = maxx - minx
        yrange = maxy - miny

        xs = np.random.uniform(minx + 0.2 * xrange, maxx - 0.2 * xrange, 30)
        ys = np.random.uniform(miny + 0.2 * yrange, maxy - 0.2 * yrange, 30)
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

        bstype_order = [
            bs.BaseStationType.GROUND,  # Source
            bs.BaseStationType.MARITIME,
            bs.BaseStationType.GROUND,
            bs.BaseStationType.HAPS,
            bs.BaseStationType.LEO,
        ]

        # Add basestation nodes
        node_id = 0
        for idx in range(5):  # first five lists are basestations
            pts, alts = self.node_points[idx], self.node_alts[idx]
            bstype = bstype_order[idx]
            for p, alt in zip(pts, alts):
                graph.add_node(
                    bs.BaseStation(
                        node_id,
                        np.array([p.x, p.y, alt]),
                        bstype,
                        isGeographic=True,
                    )
                )
                node_id += 1

        # Add user nodes
        for p, alt in zip(self.node_points[-1], self.node_alts[-1]):
            graph.add_node(
                bs.User(node_id, np.array([p.x, p.y, alt]), isGeographic=True)
            )
            node_id += 1

        # Connect the nodes
        graph.connect_reachable_nodes()

        # Change the location of source node connected with at least 5 nodes.
        source = graph.nodes[0]
        while True:
            count = 0
            for node in source.get_children():
                if (
                    node.basestation_type.name == bs.BaseStationType.HAPS.name
                    or node.basestation_type.name == bs.BaseStationType.LEO.name
                ):
                    count += 1
            if count > 1:
                break

            graph.reset()
            for node in graph.basestations:
                if (
                    node.basestation_type.name == bs.BaseStationType.HAPS.name
                    or node.basestation_type.name == bs.BaseStationType.LEO.name
                ):
                    # Re-position the node
                    node._position[0] = random.uniform(
                        self.longitude_range[0], self.longitude_range[1]
                    )
                    node._position[1] = random.uniform(
                        self.latitude_range[0], self.latitude_range[1]
                    )
            graph.connect_reachable_nodes()

            # # Remove the edges connected to the source node
            # for node in source.get_children():
            #     graph.remove_edge(source.get_id(), node.get_id())
            #
            # # Generate a new source node
            # source_basestation_point = self.generate_source_point(self.gdf_list[0], 0)
            # self.node_gdf_list[0] = gpd.GeoDataFrame(
            #     geometry=source_basestation_point, crs=self.target_crs
            # )
            # # Update the source node position
            # source._position = np.array(
            #     [source_basestation_point[0].x, source_basestation_point[0].y, 0]
            # )
            # # Add the source node to the graph
            # graph.connect_reachable_nodes(target_node_id=0)

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
    
# --------------------------------------------------------------------------- #
def _center_bbox(p: Path) -> Path:
    """Translate path so that its bounding-box centre becomes the origin."""
    verts = p.vertices.copy()
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts -= centre
    return Path(verts, p.codes)


def load_svg_path(fp: str,
                  *,
                  rotate_deg: float = 0.0,
                  normalise: bool = True,
                  y_offset: float = 0.0) -> Path:
    """
    Load the *first* <path> element from an SVG, centre it, optionally rotate
    and scale so that the maximum absolute extent equals 1.  Suitable for use
    as a matplotlib marker.
    """
    _, attrs = svg2paths(fp)
    p = parse_path(attrs[0]["d"])
    p = _center_bbox(p)

    if normalise:
        p.vertices /= np.abs(p.vertices).max()

    if rotate_deg:
        p = p.transformed(Affine2D().rotate_deg(rotate_deg))

    p = p.transformed(Affine2D().scale(-1, 1))
    return p


def build_markers(icon_dir: str) -> Dict[str, Tuple[Path, Path]]:
    """
    Pre-load all markers used in PlotManager.
    Returns:
        {
          "ground":   (circle_marker, ground_silhouette),
          "maritime": (circle_marker, maritime_silhouette),
          ...
          "source":   (source_marker, None)          # no outer overlay
        }
    """
    circle = load_svg_path(f"{icon_dir}/circle.svg")  # outer ring (black)
    source = load_svg_path(f"{icon_dir}/source.svg", rotate_deg=180)  # source BS (star)

    silhouette_map = {
        "ground":   load_svg_path(f"{icon_dir}/ground.svg", rotate_deg=180),
        "maritime": load_svg_path(f"{icon_dir}/maritime.svg", rotate_deg=180),
        "haps":     load_svg_path(f"{icon_dir}/haps.svg", rotate_deg=180),
        "leo":      load_svg_path(f"{icon_dir}/leo.svg", rotate_deg=180),
        "users":    load_svg_path(f"{icon_dir}/user.svg", rotate_deg=180),
    }

    markers = {k: (circle, v) for k, v in silhouette_map.items()}
    markers["source"] = (None, source)   # composite not needed
    return markers

@dataclass
class PlotManager:
    # Set plotting parameters
    fontsize = 16
    markersize = 8
    linewidth = 1
    gdf_color_list = [
        np.array([230, 255, 230]) / 255.0,  # land
        "#C5F0FF", # lakes
        "#C5F0FF", # rivers
        "#C5F0FF", # maritime
        np.array([0, 0, 0]) / 255.0,  # coastline
    ]
    node_color_list = [
        "deeppink",  # source basestation
        "#1360D5",  # maritime basestation
        "#004d4d",  # ground basestation
        "#0930AA",  # haps basestation
        "#000080",  # leo basestation
        "#1C90FF",  # users
    ]
    # node_color_list = [
    #     "#7D0A0A",
    #     "#050C9C",
    #     "#EA7300",
    #     "#3572EF",
    #     "#3ABEF9",
    #     "#BF3131",
    # ]
    node_marker_list = ["*", "o", "^", "P", "s", "d"]
    node_label_list = [
        "Source Basestation",
        "Maritime Basestation",
        "Ground Basestation",
        "HAPS Basestation",
        "LEO Basestation",
        "Users",
    ]

    # ----------------------------------------------------------------------- #
    # runtime-initialised attributes                                          #
    # ----------------------------------------------------------------------- #
    icon_dir: str = "icons"         # where the SVGs live
    _markers: Dict[str, Tuple[Path, Optional[Path]]] = None  # set in __post_init__

    def __post_init__(self):
        """Pre-load all SVG markers once."""
        self._markers = build_markers(self.icon_dir)

    # ----------------------------------------------------------------------- #
    #   Internal helper – draw a single node                                  #
    # ----------------------------------------------------------------------- #
    def _draw_node(self, ax, x, y, node_type: str,
                   face_color: str, ms: float, z: int, label: Optional[str]):
        """
        Draw either:
          • composite marker = outer black circle + white silhouette
          • single marker (for 'source')
        """
        outer, inner = self._markers[node_type]

        ax.plot(x, y, marker=inner, ls="None", markeredgewidth=0, markeredgecolor="none",
                color=face_color, markersize=ms, zorder=z, label=label)
        return


    def plot(
        self,
        dm: DataManager,
        graph: Optional[bs.IABRelayGraph] = None,
        verbose: bool = False,
        verbose_id: Optional[Union[int, List[int]]] = None,
        legend: bool = True,
        save_path: str = "",
        plot: bool = True,
        draw_bbox: List[float] = None,
    ):
        """
        Plot geographic map and, if provided, a single IABRelayGraph:
        - background layers
        - nodes (highlighted if in graph)
        - dashed edges weighted by bandwidth share
        - longest-hop path in crimson
        - stats box (kbps) at top-right, left-aligned
        - non-source nodes: Tx/Jam pies
        - source node: Tx/Jam mini bars
        - circle the BS with minimum throughput
        """

        # 1) Setup figure & axes
        fig, ax = plt.subplots(figsize=(8, 8))
        lon_min, lon_max = dm.longitude_range; lat_min, lat_max = dm.latitude_range
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude", fontsize=self.fontsize)
        ax.set_ylabel("Latitude", fontsize=self.fontsize)
        ax.set_xticks(np.arange(lon_min, lon_max + 1, 8))
        ax.set_yticks(np.arange(lat_min, lat_max + 1, 5))

        # 2) Plot background
        for gdf, col in zip(dm.gdf_list, self.gdf_color_list):
            if not gdf.empty: gdf.plot(ax=ax, color=col, linewidth=0, zorder=1)

        # 3) Highlight logic
        highlighted, longest_hop_edges = set(), set()
        min_bs_node = None
        if graph:
            graph.compute_hops()
            highlighted |= {bsn.get_id() for bsn in graph.basestations if bsn.has_parent() or bsn.has_children()}
            highlighted |= {u.get_id() for u in graph.users}
            if graph.users:
                u_max = max(graph.users, key=lambda u: u.hops)
                cur = u_max
                while cur.has_parent():
                    p = cur.get_parent()[0]; longest_hop_edges.add((p.get_id(), cur.get_id())); cur = p
            if len(graph.basestations) > 1:
                min_bs_node = min(graph.basestations[1:], key=lambda b: b.compute_throughput())

        # 4) Plot nodes
        vid_list = ([verbose_id] if isinstance(verbose_id, int) else verbose_id) or []
        node_types = ["source", "maritime", "ground", "haps", "leo", "users"]
        nid = 0
        for layer, ng in enumerate(dm.node_gdf_list):
            node_type = node_types[layer]
            color_node = self.node_color_list[layer]
            for idx, row in ng.iterrows():
                x, y = row.geometry.x, row.geometry.y  # type: ignore
                hi = (graph is None or nid in highlighted)
                c = color_node if hi else "lightgray"
                s = self.markersize*(2 if nid==0 else 1) if hi else 5
                if node_type == "users":
                    s = self.markersize * 2
                if nid == 0:
                    s = self.markersize * 3
                zorder = 6 if hi else 1
                label = self.node_label_list[layer] if idx == 0 else None
                if node_type == "source":
                    self._draw_node(ax, x, y+0.36, node_type, c, s, zorder, label)
                else:
                    self._draw_node(ax, x, y, node_type, c, s, zorder, label)
                if min_bs_node and nid == min_bs_node.get_id():
                    ax.add_patch(mpatches.Circle((x+0.025, y), 0.4, fill=True, facecolor="#FFE000", zorder=1))
                if verbose or nid in vid_list:
                    ax.text(x, y-0.2, str(nid), fontsize=8, zorder=6)
                nid += 1

        # 5) If graph, draw stats, edges, pies/bars
        if graph:
            # 5.1 stats
            tput = graph.compute_network_throughput()/1e3
            hops = np.mean([u.hops for u in graph.users]) if graph.users else 0
            # dlist = [(graph.nodes[u].get_distance(graph.nodes[v]), np.log2(1+graph.nodes[u]._compute_snr(graph.nodes[v])))
            #         for u, v in graph.edges if isinstance(graph.nodes[u], bs.BaseStation)]
            dlist = []
            for u, v in graph.edges:
                if isinstance(graph.nodes[u], bs.BaseStation) and u != 0:
                    dist = graph.nodes[u].get_distance(graph.nodes[v])
                    se = np.log2(1 + graph.nodes[u]._compute_snr(graph.nodes[v])) * 1e3
                    dlist.append((dist, se))
            dist_avg = np.mean([d for d, _ in dlist]) if dlist else 0
            se_avg   = np.mean([se for _, se in dlist]) if dlist else 0
            dx, dy = (lon_max-lon_min)*0.02, (lat_max-lat_min)*0.02
            stats = (f"Throughput (min): {tput:6.1f} kbps\n"
                    f"Avg hops         : {hops:5.2f}\n"
                    f"Avg link dist    : {dist_avg:5.1f} km\n"
                    f"Avg spectral η   : {se_avg:5.2f} b/s/Hz")
            ax.text(lon_max-dx, lat_max-dy, stats, ha="right", va="top",
                    multialignment="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8), zorder=6)

            # 5.2 denom map
            denom = {bsn: sum(self._link_bw_weight(bsn, ch) for ch in bsn.get_children())
                    for bsn in graph.basestations if bsn.has_children()}

            # 5.3 edges
            for u, v in graph.edges:
                pu, pv = graph.nodes[u], graph.nodes[v]
                if not isinstance(pu, bs.BaseStation): continue
                num = self._link_bw_weight(pu, pv); den = denom.get(pu, 1); ratio = num/den if den>0 else 0
                lw = 0.8 + 1.5*ratio
                if isinstance(pv, bs.User) and pv.hops==1: lw=0.5
                col = "black"
                # col = "#FF0000" if (u, v) in longest_hop_edges else "black"
                ls = (0, (3, 1, 1, 1)) if (u, v) in longest_hop_edges else "-"
                x0, y0, _ = pu.get_position(); x1, y1, _ = pv.get_position()
                ax.plot([x0, x1], [y0, y1], ls=ls, color=col, linewidth=lw, zorder=4)
                if (u, v) in longest_hop_edges:
                    ax.plot([x0, x1], [y0, y1], ls="-", color="#FFE000", linewidth=7.5, alpha=0.8, zorder=3)

            # 5.4 pies/bars
            for bsn in graph.basestations:
                nid_ = bsn.get_id()
                if nid_ != 0 and nid_ in highlighted:
                    # increase width_frac, height_frac, offset_frac, and zorder
                    self._draw_power_bar(
                        ax,
                        bsn,
                        dm,
                        width_frac=0.01,      # was 0.01 → wider
                        height_frac=0.03,     # was 0.03 → taller
                        offset_frac=(0.02, 0.00),  # move further away from node
                        zorder=10             # draw above nodes/edges
                    )
            
            # 5.5 draw bbox
            if draw_bbox is not None:
                lon_min, lon_max = draw_bbox[0], draw_bbox[1]
                lat_min, lat_max = draw_bbox[2], draw_bbox[3]
                ax.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max-lon_min, lat_max-lat_min,
                                                fill=False, edgecolor="red", lw=1.5, zorder=10))

        # 6) legend & finalize
        if legend: ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        # save as eps
        if save_path != "": plt.savefig(save_path, dpi=300, bbox_inches="tight", format="eps")
        if plot: plt.show()

    # ── helper: 링크별 BW-weight 계산 ─────────────────────────────────────
    def _link_bw_weight(self, parent: bs.BaseStation, child: bs.AbstractNode) -> float:
        """
        Return relative bandwidth share (0~1) of link parent→child.
        Denominator = Σ_{child∈parent.children} (sum_hops / spectral_eff).
        """
        # Ensure power densities are up-to-date
        parent._set_transmission_and_jamming_power_density()

        snr = parent._compute_snr(child)
        se = np.log2(1 + snr)

        # sum_hops 계산 방식은 compute_throughput과 동일
        if isinstance(child, bs.User):
            hops_sum = child.hops
        else:  # child is BaseStation
            hops_sum = sum(u.hops for u in child.connected_user)

        return hops_sum / (se + 1e-9)  # numerator

    def _draw_power_bar(
        self,
        ax,
        node: bs.BaseStation,
        dm: DataManager,
        width_frac: float = 0.01,
        height_frac: float = 0.03,
        offset_frac: Tuple[float, float] = (0.02, 0.02),
        zorder: int = 5,
    ):
        """
        Draw two vertical bars next to the source BS:
        - green: transmit power / max_transmit
        - red  : jamming power  / max_jamming
        Bars normalized so max height = height_frac * lat_range.
        """
        # update Tx/Jam densities
        node._set_transmission_and_jamming_power_density()
        tx_lin   = bs.dB_to_linear(node.transmission_power_density)
        jam_lin  = bs.dB_to_linear(node.jamming_power_density)
        cfg      = node.basestation_type.config
        total_lin= bs.dB_to_linear(cfg.power_capacity)/(cfg.bandwidth*1e6)
        tx_max   = total_lin * cfg.minimum_transit_power_ratio
        jam_max  = total_lin * (1 - cfg.minimum_transit_power_ratio)
        f_tx     = tx_lin/tx_max   if tx_max>0 else 0
        f_jam    = jam_lin/jam_max if jam_max>0 else 0

        lon_min, lon_max = dm.longitude_range
        lat_min, lat_max = dm.latitude_range
        bar_w    = (lon_max-lon_min)*width_frac
        bar_hmax = (lat_max-lat_min)*height_frac

        x0, y0,_= node.get_position()
        dx, dy   = (lon_max-lon_min)*offset_frac[0], (lat_max-lat_min)*offset_frac[1]
        x0 += dx; y0 += dy

        # transmit bar
        rect_tx = mpatches.Rectangle(
            (x0, y0), bar_w, f_tx*bar_hmax,
            facecolor="green", edgecolor="w", lw=0.8, zorder=zorder
        )
        # jamming bar just to the right
        rect_jm = mpatches.Rectangle(
            (x0+bar_w*1.2, y0), bar_w, f_jam*bar_hmax,
            facecolor="red", edgecolor="w", lw=0.8, zorder=zorder
        )
        ax.add_patch(rect_tx)
        ax.add_patch(rect_jm)

def load_leo_positions(csv_path, latitude_range=[-90, 90], longitude_range=[-180, 180]):
    """
    Load slant observations from CSV and compute
    (longitude, latitude, altitude) for each entry in one function.

    Parameters:
      - csv_path: path to CSV with columns 'distance','elevation','azimuth'
      - observer_lat_deg: observer latitude in degrees
      - observer_lon_deg: observer longitude in degrees

    Returns:
      - List of tuples: (lon_deg, lat_deg, alt_km)
    """
    positions = []

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat_deg = float(row["lat"])
            lon_deg = float(row["lon"])
            alt_km = float(row["alt"])
            if (
                latitude_range[0] <= lat_deg <= latitude_range[1]
                and longitude_range[0] <= lon_deg <= longitude_range[1]
            ):
                positions.append((lat_deg, lon_deg, alt_km))

    return positions


def load_haps_positions(
    csv_path: str, when: dt.datetime, tol: timedelta = timedelta(hours=1.5)
) -> list[tuple[float, float, float]]:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    mask = (df["timestamp"] >= when - tol) & (df["timestamp"] <= when + tol)
    pos_df = df.loc[mask, ["callsign", "timestamp", "lat", "lon", "alt"]]

    positions: list[tuple[float, float, float]] = list(
        pos_df[["lat", "lon", "alt"]].itertuples(index=False, name=None)
    )
    
    # Remove duplicates within 0.15 deg
    non_duplicate_positions = []
    for lat, lon, alt in positions:
        if any(
            abs(lat - pos[0]) < 0.15 and abs(lon - pos[1]) < 0.15
            for pos in non_duplicate_positions
        ):
            continue
        non_duplicate_positions.append((lat, lon))
        
    return non_duplicate_positions


def load_ground_positions(
    csv_path, latitude_range=[-90, 90], longitude_range=[-180, 180], duplicate_tol=0.15
):
    """
    Extract (lat, lon) positions of LTE base stations from OpenCelliD CSV.

    Parameters
    ----------
    csv_path : str
        Path to the OpenCelliD-formatted CSV file.

    Returns
    -------
    List[Tuple[float, float]]
        List of (latitude, longitude) tuples for LTE radio type entries.
    """
    positions = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            if (
                latitude_range[0] <= lat <= latitude_range[1]
                and longitude_range[0] <= lon <= longitude_range[1]
            ):
                # skip the entry if there are another bs within 0.1 deg
                if any(
                    abs(lat - pos[0]) < duplicate_tol and abs(lon - pos[1]) < duplicate_tol
                    for pos in positions
                ):
                    continue
                positions.append((lat, lon))
    return positions


def load_maritime_positions(csv_path):
    """
    Extract (lat, lon) positions of maritime base stations from OpenCelliD CSV.

    Parameters
    ----------
    csv_path : str
        Path to the OpenCelliD-formatted CSV file.

    Returns
    -------
    List[Tuple[float, float]]
        List of (latitude, longitude) tuples for LTE radio type entries.
    """
    positions = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            positions.append((lat, lon))
    return positions


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
