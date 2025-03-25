#!/usr/bin/env python3
import json
import math
import os
import pickle
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch
from numpy.typing import NDArray
from torch_geometric.data import Data


@dataclass
class EnvironmentalVariables:
    noise_power_density: float = -174
    SPSC_probability: float = 0.9999
    maritime_basestations_altitude: float = 0.0
    ground_basestations_altitude: float = 0.25
    haps_basestations_altitude: float = 22
    leo_basestations_altitude: float = 500


environmental_variables = EnvironmentalVariables()


def dB_to_linear(dB: float) -> float:
    return 10 ** (dB / 10)


def linear_to_dB(linear: float) -> float:
    return 10 * np.log10(linear)


@dataclass
class BaseStationConfig:
    power_capacity: float  # in dBm
    minimum_transit_power_ratio: float  # dimensionless
    carrier_frequency: float  # in GHz
    bandwidth: float  # in MHz
    transmit_antenna_gain: float  # in dBi
    receive_antenna_gain: float  # in dBi
    antenna_gain_to_noise_temperature: float  # in dB
    pathloss_exponent: float  # dimensionless
    eavesdropper_density: float  # in m^-2
    maximum_link_distance: float  # in km


class BaseStationType(Enum):
    MARITIME = BaseStationConfig(
        power_capacity=30,  # in dBm
        minimum_transit_power_ratio=0.8,  # dimensionless
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.9,  # dimensionless
        eavesdropper_density=1e-3,  # in km^-2
        maximum_link_distance=150,  # in km
    )
    GROUND = BaseStationConfig(
        power_capacity=30,  # in dBm
        minimum_transit_power_ratio=0.8,  # dimensionless
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.8,  # dimensionless
        eavesdropper_density=1e-3,  # in km^-2
        maximum_link_distance=150,  # in km
    )
    HAPS = BaseStationConfig(
        power_capacity=30,  # in dBm
        minimum_transit_power_ratio=0.8,  # dimensionless
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.5,  # dimensionless
        eavesdropper_density=3e-4,  # in km^-2
        maximum_link_distance=500,  # in km
    )
    LEO = BaseStationConfig(
        power_capacity=21.5,  # in dBm
        minimum_transit_power_ratio=0.8,  # dimensionless
        carrier_frequency=20,  # in GHz
        bandwidth=400,  # in MHz
        transmit_antenna_gain=38.5,  # in dBi
        receive_antenna_gain=38.5,  # in dBi
        antenna_gain_to_noise_temperature=13,  # in dB
        pathloss_exponent=2.5,  # dimensionless
        eavesdropper_density=5e-5,  # in km^-2
        maximum_link_distance=900,  # in km
    )

    @property
    def config(self) -> BaseStationConfig:
        """Returns the configuration for the base station type."""
        return self.value

    def __repr__(self):
        return f"BaseStationType({self.name})"


class AbstractNode(ABC):
    """An abstract node class."""

    def __init__(self, node_id: int, position: NDArray, isGeographic: bool = True):
        self._node_id = node_id
        self._position = position
        self._parent: List[AbstractNode] = []
        self._children: List[AbstractNode] = []
        self._isGeographic = isGeographic

    def get_position(self) -> NDArray:
        return self._position

    def get_distance(self, node: "AbstractNode"):
        """
        Calculate the distance between this node and another node.

        Returns:
            Distance in kilometers if _isGeographic is True,
            otherwise in local metric units.
        """
        if self._isGeographic:
            lat1, lon1, alt1 = self.get_position()
            lat2, lon2, alt2 = node.get_position()

            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            # Haversine formula for great-circle distance
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            # Horizontal distance on the Earth's surface
            R = 6371.01  # Earth's radius in kilometers
            horizontal_distance = R * c

            # Vertical distance (altitude difference)
            altitude_difference = alt2 - alt1

            # 3D distance calculation using Pythagoras' theorem
            return math.sqrt(horizontal_distance**2 + altitude_difference**2)
        else:
            # Use Euclidean distance for local metric systems
            return float(np.linalg.norm(self.get_position() - node.get_position()))

    def get_id(self) -> int:
        return self._node_id

    def get_parent(self) -> List["AbstractNode"]:
        return self._parent

    def get_children(self) -> List["AbstractNode"]:
        return list(self._children)

    def has_children(self) -> bool:
        return len(self._children) > 0

    def has_parent(self) -> bool:
        return len(self._parent) > 0

    def add_child_link(self, node: "AbstractNode"):
        if node not in self._children:
            self._children.append(node)

    def add_parent_link(self, node: "AbstractNode"):
        if node not in self._parent:
            self._parent.append(node)

    def remove_child_link(self, node: "AbstractNode"):
        """
        Removes a child node from the children list.
        """
        if node in self._children:
            self._children.remove(node)

    def remove_parent_link(self, node: "AbstractNode"):
        """
        Removes a parent node from the parent list.
        """
        if node in self._parent:
            self._parent.remove(node)


class BaseStation(AbstractNode):
    def __init__(
        self,
        node_id: int,
        position: NDArray,
        basestation_type: BaseStationType,
        isGeographic: bool = True,
    ):
        super().__init__(
            node_id=node_id,
            position=position,
            isGeographic=isGeographic,
        )
        self.basestation_type = basestation_type
        self.connected_user: List[User] = []
        self.transmission_power_density: float = 0.0
        self.jamming_power_density: float = 0.0
        self.throughput: float = 0.0

    def __repr__(self):
        # return f"BaseStation(node_id={self._node_id}, position={self._position}, type={self.basestation_type})"
        return f"BS{self._node_id}"

    def compute_throughput(self) -> float:
        """
        Compute the throughput of the base station.
        Implementation of the Equation (??) in the paper.
        """
        if not self.has_children():
            return float("inf")
        self._set_transmission_and_jamming_power_density()
        denominator = 1e-8
        for node in self.get_children():
            snr = self._compute_snr(node)
            spectral_efficiency = np.log2(1 + snr)
            if isinstance(node, User):
                sum_hops = node.hops
            elif isinstance(node, BaseStation):
                sum_hops = sum([user.hops for user in node.connected_user])
            else:
                raise ValueError("Unsupported node type.")
            denominator += sum_hops / spectral_efficiency

        self.throughput = self.basestation_type.config.bandwidth * 1e6 / denominator
        return self.throughput

    def _compute_snr(self, node, in_dB: bool = False) -> float:
        """
        Computes SNR (in dB) at a given distance (meters) using a log-distance pathloss model.
        Assumes that power_capacity in BaseStationConfig is in dBm, carrier_frequency is in GHz,
        bandwidth is in MHz, antenna gains are in dBi, and pathloss_exponent is dimensionless.

        SNR(dB) = RxPower(dBm) - NoisePower(dBm)
        RxPower(dBm) = TxPower(dBm) + TxGain(dBi) + RxGain(dBi) - Pathloss(dB)
        Pathloss(dB) = Pathloss(1m) + 10*alpha*log10(d/1m)
        NoisePower(dBm) = -174 + 10*log10(BW_Hz)
        """
        distance_m = self.get_distance(node) * 1e3
        config = self.basestation_type.config

        # Physical constants and configuration parameters
        c = 3.0e8  # speed of light (m/s)
        freq_ghz = config.carrier_frequency
        freq_hz = freq_ghz * 1e9
        wavelength_m = c / freq_hz
        reference_distance_m = 1.0

        # Calculate path loss at reference distance d0=1m
        # Pathloss(1m) = 20*log10(4*pi * 1m / lambda)
        pathloss_1m = 20.0 * np.log10(
            (4.0 * np.pi * reference_distance_m) / wavelength_m
        )

        # Calculate path loss at distance d(m) [dB]
        pathloss_d = pathloss_1m + 10.0 * config.pathloss_exponent * np.log10(
            distance_m
        )

        # Transmit power [dBm], transmit/receive antenna gains [dBi]
        bw_hz = self.basestation_type.config.bandwidth * 1e6
        tx_power_density_dbm = self.transmission_power_density
        tx_gain_db = config.transmit_antenna_gain
        rx_gain_db = config.receive_antenna_gain

        # Received power [dBm]
        rx_power_dbm = tx_power_density_dbm + tx_gain_db + rx_gain_db - pathloss_d

        # Calculate noise power [dBm]
        # Thermal noise = -174 dBm/Hz
        noise_power_density_dbm = (
            environmental_variables.noise_power_density
            + config.antenna_gain_to_noise_temperature
        )

        # SNR [dB]
        snr_db = rx_power_dbm - noise_power_density_dbm

        return snr_db if in_dB else dB_to_linear(snr_db)

    def _set_transmission_and_jamming_power_density(self):
        """
        Compute the transmission and jamming power density.
        Implmentation of the Equation (??) in the paper.
        """
        # Physical constants and configuration parameters
        config = self.basestation_type.config
        pathloss_exponent = config.pathloss_exponent
        power_capacity_density = (
            dB_to_linear(config.power_capacity) / config.bandwidth / 1e6
        )  # in mW/Hz
        noise_power_density = dB_to_linear(environmental_variables.noise_power_density)
        tau = environmental_variables.SPSC_probability
        kappa = (
            np.pi
            * self.basestation_type.config.eavesdropper_density
            / np.sin(2 * np.pi / pathloss_exponent)
        ) ** 0.806 / 0.11  # Curve fitting
        max_distance = self._get_farthest_forward_link_distance()

        # Equation for threshold
        jamming_power_density_mW_over_Hz = (
            max(
                (-(kappa * max_distance**2) / np.log(tau)) ** (pathloss_exponent / 2)
                - 1,
                0,
            )
            * noise_power_density
            * 3.1623  # 5 dBm offset
        )
        jamming_power_density_mW_over_Hz = min(
            jamming_power_density_mW_over_Hz, power_capacity_density
        )
        transmission_power_density_mW_over_Hz = (
            power_capacity_density - jamming_power_density_mW_over_Hz
        )
        self.transmission_power_density = linear_to_dB(
            transmission_power_density_mW_over_Hz + 1e-16
        )
        self.jamming_power_density = linear_to_dB(
            jamming_power_density_mW_over_Hz + 1e-16
        )

        return self.transmission_power_density, self.jamming_power_density

    def _get_farthest_forward_link_distance(self):
        max_distance = 0.0
        for node in self.get_children():
            distance = self.get_distance(node)
            if distance > max_distance:
                max_distance = distance
        return max_distance

    def compute_maximum_link_distance(self):
        """
        Compute the maximum link distance.
        Implementation of the Equation (??) in the paper.
        """
        # Physical constants and configuration parameters
        config = self.basestation_type.config
        pathloss_exponent = config.pathloss_exponent
        power_capacity_density = (
            dB_to_linear(config.power_capacity) / config.bandwidth / 1e6
        )
        maximum_jamming_power_density = power_capacity_density * (
            1 - config.minimum_transit_power_ratio
        )
        noise_power_density = dB_to_linear(environmental_variables.noise_power_density)

        tau = environmental_variables.SPSC_probability
        kappa = (
            np.pi * config.eavesdropper_density / np.sin(2 * np.pi / pathloss_exponent)
        ) ** 0.806 / 0.11  # Curve fitting
        jamming_ratio = (
            noise_power_density
            / (maximum_jamming_power_density + noise_power_density)
            * 3.1623
        )  # 5 dBm offset

        max_distance = min(
            (-np.log(tau) / kappa / jamming_ratio ** (2 / pathloss_exponent)) ** 0.5,
            self.basestation_type.config.maximum_link_distance,
        )

        return max_distance


class User(AbstractNode):
    def __init__(self, node_id: int, position: NDArray, isGeographic: bool = True):
        super().__init__(
            node_id=node_id,
            position=position,
        )
        self.hops: int = 0

    def __repr__(self):
        # return f"User(node_id={self._node_id}, position={self._position})"
        return f"UE{self._node_id}"


class IABRelayGraph:
    def __init__(self, environmental_variables=environmental_variables):
        self.nodes: Dict[int, BaseStation | User] = {}
        self.users: List[User] = []
        self.basestations: List[BaseStation] = []

        self.adjacency_list: Dict[int, List[int]] = {}
        self.environmental_variables = environmental_variables
        self.is_hop_computed = False

    @property
    def edges(self):
        edges = []
        for from_node_id, neighbors in self.adjacency_list.items():
            for to_node_id in neighbors:
                edges.append((from_node_id, to_node_id))
        return edges

    def add_node(self, node: AbstractNode):
        node_id = node.get_id()
        if node_id not in self.nodes:
            self.nodes[node_id] = node
            self.adjacency_list[node_id] = []
            if isinstance(node, User):
                self.users.append(node)
            elif isinstance(node, BaseStation):
                self.basestations.append(node)
            else:
                raise ValueError("Unsupported node type.")

    def add_edge(self, from_node_id: int, to_node_id: int):
        # check if the edge already exists
        if to_node_id in self.adjacency_list[from_node_id]:
            return

        if from_node_id in self.nodes and to_node_id in self.nodes:
            self.adjacency_list[from_node_id].append(to_node_id)
            self.nodes[from_node_id].add_child_link(self.nodes[to_node_id])
            self.nodes[to_node_id].add_parent_link(self.nodes[from_node_id])
        else:
            raise ValueError(
                f"One or more nodes do not exist in the graph: {from_node_id}, {to_node_id}"
            )

    def remove_node(self, node_id: int):
        """
        Removes a node and all associated edges from the graph.
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph.")

        # Remove all incoming edges to the node
        for from_id, neighbors in self.adjacency_list.items():
            if node_id in neighbors:
                neighbors.remove(node_id)
                self.nodes[from_id].remove_child_link(self.nodes[node_id])
                self.nodes[node_id].remove_parent_link(self.nodes[from_id])

        # Remove all outgoing edges from the node
        for to_id in self.adjacency_list[node_id]:
            self.nodes[to_id].remove_parent_link(self.nodes[node_id])
        del self.adjacency_list[node_id]

        # Remove the node from the nodes dictionary
        node = self.nodes.pop(node_id)

        # Remove the node from users or basestations list
        if isinstance(node, User):
            self.users.remove(node)
        elif isinstance(node, BaseStation):
            self.basestations.remove(node)
        else:
            raise ValueError("Unsupported node type.")

    def remove_edge(self, from_node_id: int, to_node_id: int):
        """
        Removes an edge between two nodes in the graph.
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return

        if to_node_id not in self.adjacency_list[from_node_id]:
            return

        # Remove the edge from the adjacency list
        self.adjacency_list[from_node_id].remove(to_node_id)

        # Update the parent and child links of the involved nodes
        self.nodes[from_node_id].remove_child_link(self.nodes[to_node_id])
        self.nodes[to_node_id].remove_parent_link(self.nodes[from_node_id])

    def reset(self):
        # Removes all edges in the graph.
        for node in self.nodes.values():
            node._children = []
            node._parent = []
        self.adjacency_list = {node_id: [] for node_id in self.adjacency_list}

        # Removes all user information
        for user in self.users:
            user.hops = 0

    def get_neighbors(self, node_id: int) -> List[int]:
        return self.adjacency_list.get(node_id, [])

    def compute_hops(self):
        # reset the hops of users
        for user in self.users:
            user.hops = 0
        for base_station in self.basestations:
            base_station.connected_user = []
        #
        #
        # if self.is_hop_computed:
        #     return

        self.is_hop_computed = True
        for user in self.users:
            self.compute_hops_for_one_user(user.get_id())
            # current_node = user
            # while True:
            #     assert current_node is not None, f"Current node {current_node} is None."
            #     if current_node.has_parent():
            #         user.hops += 1
            #         parent_nodes: List[AbstractNode] = current_node.get_parent()
            #         assert (
            #             len(parent_nodes) == 1
            #         ), f"There are more than one parent node. Current node: {current_node} Parent node: {parent_nodes}"
            #         current_node = parent_nodes[0]
            #         current_node.connected_user.append(user)
            #     else:
            #         break

    def compute_hops_for_one_user(self, user_id):
        user = self.nodes[user_id]
        current_node = user
        while True:
            assert current_node is not None, f"Current node {current_node} is None."
            if current_node.has_parent():
                user.hops += 1
                parent_nodes: List[AbstractNode] = current_node.get_parent()
                assert (
                    len(parent_nodes) == 1
                ), f"There are more than one parent node. Current node: {current_node} Parent node: {parent_nodes}"
                current_node = parent_nodes[0]
                current_node.connected_user.append(user)
            else:
                break

    def connect_reachable_nodes(
        self, target_node_id: Optional[int] = None, source_node_id: int = 0
    ):
        """
        Connects all reachable nodes in the graph.
        """
        # Connect all reachable nodes in the graph
        if target_node_id is None:
            for from_node in self.basestations:
                from_node_id = from_node.get_id()
                for to_node_id in self.compute_reachable_nodes(from_node_id):
                    if from_node_id == to_node_id or to_node_id == source_node_id:
                        continue
                    self.add_edge(from_node_id, to_node_id)
        else:
            for to_node_id in self.compute_reachable_nodes(target_node_id):
                if to_node_id == source_node_id:
                    continue
                self.add_edge(target_node_id, to_node_id)

        # Remove the direct connection between the source node and the users
        for node in self.nodes[source_node_id].get_children():
            if isinstance(node, User):
                self.adjacency_list[source_node_id].remove(node.get_id())

                # Update the parent and child links of the involved nodes
                self.nodes[source_node_id].remove_child_link(self.nodes[node.get_id()])
                self.nodes[node.get_id()].remove_parent_link(self.nodes[source_node_id])

    def compute_reachable_nodes(self, node_id: int):
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist in the graph.")

        test_node = self.nodes[node_id]
        assert isinstance(
            test_node, BaseStation
        ), f"Node {node_id} is not a base station."

        maximum_link_distance = test_node.compute_maximum_link_distance()
        reachable_nodes = []
        for node in self.nodes.values():
            # if node.has_parent() or test_node == node:
            if test_node == node:
                continue
            if test_node.get_distance(node) <= maximum_link_distance:
                reachable_nodes.append(node.get_id())

            if isinstance(node, BaseStation):
                if (
                    node.basestation_type == BaseStationType.LEO
                    and test_node.get_distance(node) <= 800
                ):
                    reachable_nodes.append(node.get_id())
                elif (
                    node.basestation_type == BaseStationType.HAPS
                    and test_node.get_distance(node) <= 500
                ):
                    reachable_nodes.append(node.get_id())
        return reachable_nodes

    def copy(self):
        """
        Create a copy of the graph.
        """
        return self.copy_graph_with_selected_nodes(list(self.nodes.keys()))

    def copy_graph_with_selected_nodes(self, selected_nodes: List[int]):
        """
        Create a new graph with the selected nodes.
        """
        new_graph = IABRelayGraph(self.environmental_variables)
        # add node
        for node_id in selected_nodes:
            node = self.nodes[node_id]
            if isinstance(node, BaseStation):
                node_copy = BaseStation(
                    node_id,
                    node.get_position(),
                    node.basestation_type,
                )
            else:
                node_copy = User(node_id, node.get_position())

            new_graph.add_node(node_copy)

        # add edge
        for from_node_id in selected_nodes:
            for to_node_id in self.get_neighbors(from_node_id):
                if to_node_id in selected_nodes:
                    new_graph.add_edge(from_node_id, to_node_id)

        return new_graph

    def compute_network_throughput(self):
        self.compute_hops()

        throughput_list = []
        for node in self.basestations[1:]:
            throughput = node.compute_throughput()
            throughput_list.append(throughput)
        return min(throughput_list)

    def save_graph(self, filepath: str, pkl=True):
        """
        Save the graph to a file.
        """
        if pkl:
            with open(filepath, "wb") as file:
                pickle.dump(self, file)
            return

        # Save the graph as a JSON file
        data = {
            "nodes": {
                node_id: {
                    "position": node.get_position().tolist(),
                    "type": node.basestation_type.name
                    if isinstance(node, BaseStation)
                    else "User",
                }
                for node_id, node in self.nodes.items()
            },
            "edges": self.edges,
        }

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

    def load_graph(self, filepath: str, pkl=True):
        """
        Load the graph from a file.
        """
        # Check whether the file exists.
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        if pkl:
            with open(filepath, "rb") as file:
                graph = pickle.load(file)
                self.__dict__.update(graph.__dict__)
        else:
            # Load the graph from a JSON file
            with open(filepath, "r") as file:
                data = json.load(file)

            # Create nodes
            for node_id, node_data in data["nodes"].items():
                position = np.array(node_data["position"])
                if node_data["type"] == "User":
                    node = User(int(node_id), position)
                else:
                    node = BaseStation(
                        int(node_id),
                        position,
                        BaseStationType[node_data["type"]],
                    )
                self.add_node(node)

            # Create edges
            for from_node_id, to_node_id in data["edges"]:
                self.add_edge(from_node_id, to_node_id)

    def to_networkx(self):
        """
        Convert the graph to a networkx graph.
        """
        graph = nx.DiGraph()
        for node_id, node in self.nodes.items():
            # Empty placeholder
            feature = np.empty(0, dtype=np.float32)
            # Concatenate the position of the user into feature
            feature = np.concatenate((feature, node.get_position()))
            # Convert node attributes to one np array
            if isinstance(node, User):
                configs = {
                    "power_capacity": 0.0,
                    "minimum_transit_power_ratio": 0.0,
                    "carrier_frequency": 0.0,
                    "bandwidth": 0.0,
                    "transmit_antenna_gain": 0.0,
                    "receive_antenna_gain": 0.0,
                    "antenna_gain_to_noise_temperature": 0.0,
                    "pathloss_exponent": 0.0,
                    "eavesdropper_density": 0.0,
                }
                feature = np.concatenate((feature, np.array(list(configs.values()))))
            elif isinstance(node, BaseStation):
                feature = np.concatenate(
                    (
                        feature,
                        np.array(list(vars(node.basestation_type.config).values())),
                    )
                )
            graph.add_node(node_id, feature=feature)
        for from_node_id, neighbors in self.adjacency_list.items():
            for to_node_id in neighbors:
                # Compute the distance, snr, spectral efficiency as a feature
                from_node = self.nodes[from_node_id]
                to_node = self.nodes[to_node_id]
                distance = from_node.get_distance(to_node)
                if isinstance(from_node, BaseStation):
                    from_node._set_transmission_and_jamming_power_density()
                    snr = from_node._compute_snr(to_node)
                    spectral_efficiency = np.log2(1 + snr)
                    feature = np.array(
                        [distance, snr, spectral_efficiency], dtype=np.float32
                    )
                else:
                    feature = np.array([distance, 0.0, 0.0], dtype=np.float32)
                graph.add_edge(from_node_id, to_node_id, feature=feature)

        return graph

    def to_torch_geometric(self):
        """
        Convert the graph to a torch_geometric.data.Data object.
        Assumes that node_id increases sequentially from 0.
        """
        node_features = []
        for node_id, node in self.nodes.items():
            # Initialize an empty feature array
            feature = np.empty(0, dtype=np.float32)

            # Append the node id
            feature = np.append(feature, node_id)
            # Append the node type to the feature
            if node_id == 0:
                feature = np.append(feature, 0)
            elif isinstance(node, User):
                feature = np.append(feature, 1)
            else:  # BaseStation
                feature = np.append(feature, 2)

            # Append the node position to the feature
            feature = np.concatenate((feature, node.get_position()))
            # Append additional attributes based on the node type
            if isinstance(node, User):
                configs = {
                    "power_capacity": 0.0,
                    "minimum_transit_power_ratio": 0.0,
                    "carrier_frequency": 0.0,
                    "bandwidth": 0.0,
                    "transmit_antenna_gain": 0.0,
                    "receive_antenna_gain": 0.0,
                    "antenna_gain_to_noise_temperature": 0.0,
                    "pathloss_exponent": 0.0,
                    "eavesdropper_density": 0.0,
                }
                feature = np.concatenate((feature, np.array(list(configs.values()))))
            elif isinstance(node, BaseStation):
                feature = np.concatenate(
                    (
                        feature,
                        np.array(list(vars(node.basestation_type.config).values())),
                    )
                )
            node_features.append(feature)

        # Create lists for edge indices and edge features
        edge_index = []
        edge_features = []
        for from_node_id, neighbors in self.adjacency_list.items():
            for to_node_id in neighbors:
                from_node = self.nodes[from_node_id]
                to_node = self.nodes[to_node_id]
                # Compute the distance between nodes
                distance = from_node.get_distance(to_node)
                # Compute additional edge features based on the type of the from_node
                if isinstance(from_node, BaseStation):
                    from_node._set_transmission_and_jamming_power_density()
                    snr = from_node._compute_snr(to_node)
                    spectral_efficiency = np.log2(1 + snr)
                    edge_feat = np.array(
                        [distance, snr, spectral_efficiency], dtype=np.float32
                    )
                else:  # User
                    edge_feat = np.array([distance, 0.0, 0.0], dtype=np.float32)
                # Append edge index using the original node_id
                edge_index.append([from_node_id, to_node_id])
                edge_features.append(edge_feat)

        # Convert lists to torch tensors
        x = torch.tensor(np.stack(node_features), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_features:
            edge_attr = torch.tensor(np.stack(edge_features), dtype=torch.float)
        else:
            edge_attr = torch.tensor([], dtype=torch.float)

        # Create and return the torch_geometric.data.Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    # def to_torch_geometric(self, return_index_map=False):
    #     """
    #     Convert the graph to a torch_geometric.data.Data object.
    #     """
    #     # Create a mapping from node ID to index
    #     node_idx_map = {}
    #     node_features = []
    #     for idx, (node_id, node) in enumerate(self.nodes.items()):
    #         node_idx_map[node_id] = idx
    #         # Initialize feature with an empty array
    #         feature = np.empty(0, dtype=np.float32)
    #
    #         # Concatenate the node type into the feature
    #         if node_id == 0:
    #             feature = np.append(feature, 0)
    #         elif isinstance(node, User):
    #             feature = np.append(feature, 1)
    #         else:  # BaseStation
    #             feature = np.append(feature, 2)
    #
    #         # Concatenate the node position into the feature
    #         feature = np.concatenate((feature, node.get_position()))
    #         # Append additional attributes based on the node type
    #         if isinstance(node, User):
    #             configs = {
    #                 "power_capacity": 0.0,
    #                 "minimum_transit_power_ratio": 0.0,
    #                 "carrier_frequency": 0.0,
    #                 "bandwidth": 0.0,
    #                 "transmit_antenna_gain": 0.0,
    #                 "receive_antenna_gain": 0.0,
    #                 "antenna_gain_to_noise_temperature": 0.0,
    #                 "pathloss_exponent": 0.0,
    #                 "eavesdropper_density": 0.0,
    #             }
    #             feature = np.concatenate((feature, np.array(list(configs.values()))))
    #         elif isinstance(node, BaseStation):
    #             feature = np.concatenate(
    #                 (
    #                     feature,
    #                     np.array(list(vars(node.basestation_type.config).values())),
    #                 )
    #             )
    #         node_features.append(feature)
    #
    #     # Prepare lists for edge indices and edge features
    #     edge_index = []
    #     edge_features = []
    #     for from_node_id, neighbors in self.adjacency_list.items():
    #         for to_node_id in neighbors:
    #             from_node = self.nodes[from_node_id]
    #             to_node = self.nodes[to_node_id]
    #             # Compute the distance between nodes
    #             distance = from_node.get_distance(to_node)
    #             # Compute additional edge features based on from_node type
    #             if isinstance(from_node, BaseStation):
    #                 from_node._set_transmission_and_jamming_power_density()
    #                 snr = from_node._compute_snr(to_node)
    #                 spectral_efficiency = np.log2(1 + snr)
    #                 edge_feat = np.array(
    #                     [distance, snr, spectral_efficiency], dtype=np.float32
    #                 )
    #             else:  # User
    #                 edge_feat = np.array([distance, 0.0, 0.0], dtype=np.float32)
    #             # Append edge index (using node indices) and edge features
    #             edge_index.append(
    #                 [node_idx_map[from_node_id], node_idx_map[to_node_id]]
    #             )
    #             edge_features.append(edge_feat)
    #
    #     # Convert lists to torch tensors
    #     x = torch.tensor(np.stack(node_features), dtype=torch.float)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(np.stack(edge_features), dtype=torch.float)
    #
    #     # Create and return the torch_geometric.data.Data object
    #     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #     return data if not return_index_map else (data, node_idx_map)

    def from_networkx(self, graph: nx.DiGraph):
        """
        Convert a networkx graph to the IABRelayGraph.
        """
        self.nodes = {}
        self.adjacency_list = {}
        for node_id in graph.nodes:
            self.add_node(AbstractNode(node_id, np.array([0, 0, 0])))
        for from_node_id, to_node_id in graph.edges:
            self.add_edge(from_node_id, to_node_id)

    def __repr__(self):
        num_maritime_nodes = 0
        num_ground_nodes = 0
        num_leo_nodes = 0
        num_haps_nodes = 0
        num_users = 0
        for node in self.nodes.values():
            if isinstance(node, BaseStation):
                if node.basestation_type == BaseStationType.MARITIME:
                    num_maritime_nodes += 1
                elif node.basestation_type == BaseStationType.GROUND:
                    num_ground_nodes += 1
                elif node.basestation_type == BaseStationType.LEO:
                    num_leo_nodes += 1
                elif node.basestation_type == BaseStationType.HAPS:
                    num_haps_nodes += 1
            elif isinstance(node, User):
                num_users += 1

        # Total number of nodes
        total_nodes = (
            num_maritime_nodes + num_ground_nodes + num_leo_nodes + num_haps_nodes
        )

        # Generate the representation string
        return (
            f"<NodeSummary: Total Nodes={total_nodes}, "
            f"Maritime={num_maritime_nodes}, Ground={num_ground_nodes}, "
            f"LEO={num_leo_nodes}, HAPS={num_haps_nodes}, Users={num_users}>"
        )


if __name__ == "__main__":
    graph = IABRelayGraph(environmental_variables)
    bs0 = BaseStation(0, np.array([0, 0, 0]), BaseStationType.GROUND, False)
    bs1 = BaseStation(1, np.array([250, 100, 400]), BaseStationType.LEO, False)
    bs2 = BaseStation(2, np.array([150, 200, 25]), BaseStationType.HAPS, False)
    bs3 = BaseStation(3, np.array([300, 301, 0]), BaseStationType.MARITIME, False)
    bs4 = BaseStation(4, np.array([450, 300, 0]), BaseStationType.MARITIME, False)
    user1 = User(19, np.array([1, 1, 0]))
    user2 = User(20, np.array([1, 2, 0]))
    user3 = User(21, np.array([1, 6, 0]))
    user4 = User(22, np.array([1, 6, 0]))
    user5 = User(23, np.array([1, 6, 0]))
    graph.add_node(bs0)
    graph.add_node(bs1)
    graph.add_node(bs2)
    graph.add_node(bs3)
    graph.add_node(bs4)
    graph.add_node(user1)
    graph.add_node(user2)
    graph.add_node(user3)
    graph.add_node(user4)
    graph.add_node(user5)
    # graph.add_edge(0, 1)
    # graph.add_edge(0, 2)
    # graph.add_edge(2, 3)
    # graph.add_edge(3, 4)
    # graph.add_edge(1, 19)
    # graph.add_edge(2, 20)
    # graph.add_edge(4, 21)
    # graph.add_edge(4, 22)
    # graph.add_edge(4, 23)

    # graph.compute_hops()
    # # print(graph.get_neighbors(1))
    # t, j = bs0._set_transmission_and_jamming_power_density()
    # t, j = dB_to_linear(t), dB_to_linear(j)
    # t, j = (
    #     t * bs0.basestation_type.config.bandwidth * 1e6,
    #     j * bs0.basestation_type.config.bandwidth * 1e6,
    # )
    # print(f"Transmit power: {t}, jamming power:{j}")

    # print(bs0.compute_throughput())
    # print(bs0.get_distance(bs1))
    # print(bs0.compute_maximum_link_distance())
    # print(graph.compute_reachable_nodes(0))

    for from_node in graph.basestations:
        print(
            from_node.basestation_type,
            from_node.compute_maximum_link_distance(),
            graph.compute_reachable_nodes(from_node.get_id()),
        )
        for to_node in graph.compute_reachable_nodes(from_node.get_id()):
            graph.add_edge(from_node.get_id(), to_node)
    print(graph.adjacency_list)

    from ssir.pathfinder import astar

    a, b = astar.a_star(graph)
    print(a, b)
    for node in graph.nodes.values():
        print(astar.get_shortest_path(b, node.get_id()))
    # print(graph.adjacency_list)
    # print(graph.nodes[0])
    #
    # nx_graph = graph.to_networkx()
    # print(nx_graph.nodes)
    # a = nx.all_simple_paths(nx_graph, 0, 1, cutoff=3)
    # for path in a:
    #     print(path)
    gnx = graph.to_networkx()
    print(gnx.nodes(data=True))
