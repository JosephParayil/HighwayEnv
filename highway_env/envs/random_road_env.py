from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import PolyLane, PolyLaneFixedWidth
from highway_env.road.road import RoadNetwork, Road, LineType
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.controller import MDPVehicle
from highway_env.envs.common.observation import ObservationType, observation_factory
from highway_env.envs.common.action import Action, ActionType, action_factory

from highway_env.envs.generation.generator import *

import numpy as np
import math
import random
import pprint




class RandomRoadEnv(AbstractEnv):
    COLLISION_PARTITION_GRIDSIZE = 100

    def __init__(self, config: dict = None, render_mode: str | None = None, lanes = None) -> None:
        self.lanes = lanes
        super().__init__(config, render_mode)
        
        
        
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "screen_width": 1200,
                "screen_height": 700,
                 "observation": {"type": "LaneLidarObservation"},
                 "action": {"type": "ContinuousAction"}
            })
        return config
    
    

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

        if hasattr(self, "lanes") and self.lanes is not None:
            self.observation_type.hash_lanes(self.lanes)


    def _reset(self) -> None:  
        self.lanes = self._make_road(self.lanes)   
        _, self.grid_to_lanes = lanes_spatial_hash(self.lanes, RandomRoadEnv.COLLISION_PARTITION_GRIDSIZE, use_boundaries = True)
        self.vehicle = self.action_type.vehicle_class(self.road, [0.0, 0.0], 0.0, 0.0)
        self.road.vehicles.append(self.vehicle)



    def _make_road(self, lanes = None):
        if lanes is None:
            lanes = generate_random_lanes()
        
        
        # # Creating HighwayEnv objects # #
        net = RoadNetwork()
        for lane in lanes:
            #print(f"Length: {len(lane["points"])}")
            real_lane = PolyLane(
                lane_points = lane["points"],
                left_boundary_points = lane["left_points"],
                right_boundary_points = lane["right_points"],
                line_types = (LineType.CONTINUOUS, LineType.CONTINUOUS)
            )
            net.add_lane(lane["start"], lane["end"], real_lane)
            #print(real_lane.width_at(0.5))

        self.road = Road(net, record_vehicle_lane = False)


        return lanes
    
    def _reward(self, action: Action) -> float:
        return 0.0
    
    def _is_terminated(self):
        gridpoints = set()
        for pt in self.vehicle.polygon():
            gridpoints.add(point_to_gridpoint(pt, RandomRoadEnv.COLLISION_PARTITION_GRIDSIZE))

        proximal_lanes = set()
        for gpt in gridpoints:
            proximal_lanes.update(get_proximal_lanes_wrt_gridpoint(self.grid_to_lanes, gpt))
        
        for id in proximal_lanes:
            lane = self.lanes[id]

            left_pairs = zip(lane['left_points'], lane['left_points'][1:])
            right_pairs = zip(lane['right_points'], lane['right_points'][1:])
            
            for p0, p1 in chain(left_pairs, right_pairs):
                if self.vehicle.intersects_with_line(p0, p1):
                    return True

        return False


    def _is_truncated(self):
        return False
