#!/usr/bin/env python3

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class EnvironmentalVariables:
    noise_power_density: float = -174
    SPSC_probability: float = 0.99


environmental_variables = EnvironmentalVariables()


def dB_to_linear(dB: float) -> float:
    return 10 ** (dB / 10)


def linear_to_dB(linear: float) -> float:
    return 10 * np.log10(linear)


@dataclass
class BaseStationConfig:
    power_capacity: float  # in dBm
    carrier_frequency: float  # in GHz
    bandwidth: float  # in MHz
    transmit_antenna_gain: float  # in dBi
    receive_antenna_gain: float  # in dBi
    antenna_gain_to_noise_temperature: float  # in dB
    pathloss_exponent: float  # dimensionless
    eavesdropper_density: float  # in m^-2


class BaseStationType(Enum):
    SEA = BaseStationConfig(
        power_capacity=30,  # in dBm
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.0,  # dimensionless
        eavesdropper_density=1 / 1e5,  # in km^-2
    )
    GROUND = BaseStationConfig(
        power_capacity=30,  # in dBm
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.8,  # dimensionless
        eavesdropper_density=1 / 9e2,  # in km^-2
    )
    AIR = BaseStationConfig(
        power_capacity=30,  # in dBm
        carrier_frequency=14,  # in GHz
        bandwidth=250,  # in MHz
        transmit_antenna_gain=25,  # in dBi
        receive_antenna_gain=25,  # in dBi
        antenna_gain_to_noise_temperature=1.5,  # in dB
        pathloss_exponent=2.0,  # dimensionless
        eavesdropper_density=1 / 2.5e3,  # in km^-2
    )
    SPACE = BaseStationConfig(
        power_capacity=21.5,  # in dBm
        carrier_frequency=20,  # in GHz
        bandwidth=400,  # in MHz
        transmit_antenna_gain=38.5,  # in dBi
        receive_antenna_gain=38.5,  # in dBi
        antenna_gain_to_noise_temperature=13,  # in dB
        pathloss_exponent=2,  # dimensionless
        eavesdropper_density=1 / 1e4,  # in km^-2
    )

    @property
    def config(self) -> BaseStationConfig:
        """Returns the configuration for the base station type."""
        return self.value

    def _repr(self):
        return f"BaseStationType({self.name})"


class AbstractNode(ABC):
    """An abstract node class."""

    _node_id: int
    _position: NDArray
    _parent: Optional["AbstractNode"]
    _childeren: List["AbstractNode"]

    def get_position(self) -> NDArray:
        return self._position

    def get_id(self) -> int:
        return self._node_id

    def get_parent(self) -> Optional["AbstractNode"]:
        return self._parent

    def get_childeren(self) -> List["AbstractNode"]:
        return self._childeren

    def has_parent(self) -> bool:
        return self._parent is not None

    def add_child_link(self, node: "AbstractNode"):
        self._childeren.append(node)

    def add_parent_link(self, node: "AbstractNode"):
        self._parent = node


class BaseStation(AbstractNode):
    def __init__(
        self,
        node_id: int,
        position: NDArray,
        basestation_type: BaseStationType,
    ):
        super().__init__()
        self._node_id = node_id
        self._position = position
        self.basestation_type = basestation_type

        self._parent: Optional[AbstractNode] = None
        self._childeren: List[AbstractNode] = []
        self.connected_user: List[User] = []
        self.jamming_power_density: float = 0.0
        self.throughput: float = 0.0

    def __repr__(self):
        return f"BaseStation(node_id={self._node_id}, position={self._position}, type={self.basestation_type})"

    def compute_throughput(self) -> float:
        """
        Compute the throughput of the base station.
        Implementation of the Equation (??) in the paper.
        """
        self._set_transmission_and_jamming_power_density()
        denominator = 0.0
        for node in self.get_childeren():
            distance = np.linalg.norm(self.get_position() - node.get_position())
            snr = self._compute_snr(distance)
            spectral_efficiency = np.log2(1 + snr)
            if isinstance(node, User):
                sum_hops = user.hops
            elif isinstance(node, BaseStation):
                sum_hops = sum([user.hops for user in node.connected_user])
            else:
                raise ValueError("Unsupported node type.")
            denominator += sum_hops / spectral_efficiency

        self.throughput = self.basestation_type.config.bandwidth * 1e6 / denominator
        return self.throughput

    def _compute_snr(self, distance_km, in_dB: bool = False) -> float:
        """
        Computes SNR (in dB) at a given distance (meters) using a log-distance pathloss model.
        Assumes that power_capacity in BaseStationConfig is in dBm, carrier_frequency is in GHz,
        bandwidth is in MHz, antenna gains are in dBi, and pathloss_exponent is dimensionless.

        SNR(dB) = RxPower(dBm) - NoisePower(dBm)
        RxPower(dBm) = TxPower(dBm) + TxGain(dBi) + RxGain(dBi) - Pathloss(dB)
        Pathloss(dB) = Pathloss(1m) + 10*alpha*log10(d/1m)
        NoisePower(dBm) = -174 + 10*log10(BW_Hz)
        """
        distance_m = distance_km * 1e3
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
        print(f"{pathloss_1m=}")

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
            4
            * np.pi**2
            * self.basestation_type.config.eavesdropper_density
            / np.sin(3 * np.pi / pathloss_exponent)
        )
        max_distance = self._get_farthest_forward_link_distance()

        # Equation for threshold
        jamming_power_density_mW_over_Hz = (
            max(
                abs(-(kappa**3 * max_distance) / np.log(tau)) ** pathloss_exponent - 1,
                0,
            )
            * noise_power_density
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
        for node in self.get_childeren():
            distance = np.linalg.norm(self.get_position() - node.get_position())
            if distance > max_distance:
                max_distance = distance
        return max_distance


class User(AbstractNode):
    def __init__(self, node_id: int, position: NDArray):
        super().__init__()
        self._node_id = node_id
        self._position = position
        self._parent: Optional[AbstractNode] = None
        self.hops: int = 0

    def __repr__(self):
        return f"User(node_id={self._node_id}, position={self._position})"


class IABRelayGraph:
    def __init__(self, environmental_variables):
        self.nodes: Dict[int, AbstractNode] = {}
        self.users: List[User] = []
        self.base_stations: List[BaseStation] = []

        self.adjacency_list: Dict[int, List[int]] = {}
        self.environmental_variables = environmental_variables

    def add_node(self, node: AbstractNode):
        node_id = node.get_id()
        if node_id not in self.nodes:
            self.nodes[node_id] = node
            self.adjacency_list[node_id] = []
            if isinstance(node, User):
                self.users.append(node)
            elif isinstance(node, BaseStation):
                self.base_stations.append(node)
            else:
                raise ValueError("Unsupported node type.")

    def add_edge(self, from_node_id: int, to_node_id: int):
        # check if the edge already exists
        if to_node_id in self.adjacency_list[from_node_id]:
            return

        if from_node_id in self.nodes and to_node_id in self.nodes:
            if self.nodes[to_node_id].has_parent():
                raise ValueError(
                    f"Node {self.nodes[to_node_id]} already has a backhual link."
                )

            self.adjacency_list[from_node_id].append(to_node_id)
            self.nodes[from_node_id].add_child_link(self.nodes[to_node_id])
            self.nodes[to_node_id].add_parent_link(self.nodes[from_node_id])
        else:
            raise ValueError(
                f"One or more nodes do not exist in the graph: {from_node_id}, {to_node_id}"
            )

    def get_neighbors(self, node_id: int) -> List[int]:
        return self.adjacency_list.get(node_id, [])

    def compute_hops(self):
        for user in self.users:
            current_node = user
            while True:
                assert current_node is not None, f"Current node {current_node} is None."
                if current_node.has_parent():
                    user.hops += 1
                    current_node = current_node.get_parent()
                    assert isinstance(
                        current_node, BaseStation
                    ), f"Parent {current_node} is not a base station."
                    current_node.connected_user.append(user)
                else:
                    break

    def __repr__(self):
        return f"Graph(num_nodes={len(self.nodes)}"


if __name__ == "__main__":
    graph = IABRelayGraph(environmental_variables)
    bs0 = BaseStation(0, np.array([0, 0, 0]), BaseStationType.GROUND)
    bs1 = BaseStation(1, np.array([250, 100, 800]), BaseStationType.SPACE)
    bs2 = BaseStation(2, np.array([150, 200, 25]), BaseStationType.AIR)
    bs3 = BaseStation(3, np.array([300, 301, 0]), BaseStationType.SEA)
    bs4 = BaseStation(4, np.array([450, 300, 0]), BaseStationType.SEA)
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
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(1, 19)
    graph.add_edge(2, 20)
    graph.add_edge(4, 21)
    graph.add_edge(4, 22)
    graph.add_edge(4, 23)

    graph.compute_hops()
    # print(graph.get_neighbors(1))
    t, j = bs0._set_transmission_and_jamming_power_density()
    t, j = dB_to_linear(t), dB_to_linear(j)
    t, j = (
        t * bs0.basestation_type.config.bandwidth * 1e6,
        j * bs0.basestation_type.config.bandwidth * 1e6,
    )
    print(f"{t=}, {j=}")

    print(f"{bs4.connected_user=}")
    print(f"{bs3.connected_user=}")
    print(f"{bs2.connected_user=}")
    print(f"{bs1.connected_user=}")
    print(f"{bs0.connected_user=}")
    for user in graph.users:
        print(f"{user.hops=}")
    print(bs0._compute_snr(500, in_dB=True))
    print(bs0.compute_throughput())
