from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import PolyLane, PolyLaneFixedWidth
from highway_env.road.road import RoadNetwork, Road, LineType
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.objects import RoadObject, Landmark
from highway_env.envs.common.observation import ObservationType, observation_factory
from highway_env.envs.common.action import Action, ActionType, action_factory

from highway_env.envs.generation.generator import *

import numpy as np
import random
import pprint

class ParkingSpot(Landmark):
    LENGTH = 7.0
    WIDTH = 3.0


class RandomRoadEnv(AbstractEnv):
    LANE_PARTITION_GRIDSIZE = 100

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
                "observation": {
                     "type": "NavigationObservation"
                },
                 "action": {"type": "ContinuousAction"}
            })
        return config
    
    

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()



    def _reset(self) -> None:  
        self.lanes = self._make_road(self.lanes)
        
        self.create_parking_spots(num_spots = 2, spot_width = 3, spot_height = 6)   
        spawn_spot = self.road.objects[0]
        
        self.vehicle = self.action_type.vehicle_class(self.road, spawn_spot.position, spawn_spot.heading, 0.0)
        self.vehicle.goal = self.road.objects[1]
        
        self.road.vehicles.append(self.vehicle)



    def _make_road(self, lanes = None):
        if lanes is None:
            lanes = generate_random_lanes()
        
        _, grid_to_lane_ids = lanes_spatial_hash(lanes, RandomRoadEnv.LANE_PARTITION_GRIDSIZE)


        # # Creating HighwayEnv objects # #
        
        id_to_indices = []
        net = RoadNetwork()
        for id, lane in enumerate(lanes):
            #print(f"Length: {len(lane["points"])}")
            real_lane = PolyLane(
                lane_points = lane["points"],
                left_boundary_points = lane["left_points"],
                right_boundary_points = lane["right_points"],
                line_types = (LineType.CONTINUOUS, LineType.CONTINUOUS)
            )
            lane_index = net.add_lane(lane["start"], lane["end"], real_lane, bidirectional= True)
            id_to_indices.append(lane_index)

        grid_to_lane_indices = defaultdict(set)
        for gp, laneset in grid_to_lane_ids.items():
            grid_to_lane_indices[gp] = {id_to_indices[id] for id in laneset}

        net.hash(grid_to_lane_indices, RandomRoadEnv.LANE_PARTITION_GRIDSIZE)
        


        self.road = Road(net, record_vehicle_lane = False)


        return lanes
    
    def create_parking_spots(self, num_spots, spot_width, spot_height):
        curb_spot_offset = 0.1
        #height must be less than forward_speed
        #width must be less than lane_width
        
        #segment_index: {laneID, side, pt_id (1-(len-2))}
        segment_indices = []

        for id, lane in enumerate(self.lanes):
            for side in ['left_points', 'right_points']:
                for pt_id in range(1, len(lane[side])-2):
                    segment_indices.append({
                        'laneID': id,
                        'side': side,
                        'pt_id': pt_id
                    })

        random.shuffle(segment_indices)

        #parking_spots = [] # list of ParkingSpot
        num_parking_spots = 0

        segment_indices_i = 0
        while num_parking_spots < num_spots and segment_indices_i < len(segment_indices):
            segment_index = segment_indices[segment_indices_i]
            laneID = segment_index['laneID']
            side = segment_index['side']
            pt_id = segment_index['pt_id']

            lane = self.lanes[laneID]

            pt0 = lane[side][pt_id]
            pt1 = lane[side][pt_id+1]

            # We will attempt to place a parking spot parallel to our lane segment

            # Requirement 1: This segment must be long enough to encompass the parking spot
            seg_dist = np.linalg.norm(pt0 - pt1)
            if seg_dist < spot_height:
                segment_indices_i += 1
                continue
        
            
            # Computing geometry for new parking spot #

            vec = pt1-pt0
            vec /= np.linalg.norm(vec)
            
            if side == 'right_points': # if we are on the right side, we should use the 'clockwise' vector, and 'counter-clockwise' otherwise (right and left is actually swapped due to python rendering things upside down)
                perp_vec = np.array([vec[1], -vec[0]])
            else:
                perp_vec = np.array([-vec[1], vec[0]])

            center = (pt0+pt1)/2 + (perp_vec * (curb_spot_offset + spot_width/2))
            heading = np.atan2(vec[1], vec[0])

            new_parking_spot = ParkingSpot(self.road, center, heading)
            self.road.objects.append(new_parking_spot)
            num_parking_spots += 1


            # Requirement 2: The rectangular parking space should not intersect with any other lane
            if self.detect_object_lane_collision(new_parking_spot):
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue
                

            # Requirement 3: The rectangular parking space should not intersect with any other already existing parking spot
            collision_detected = False
            for other_object in self.road.objects:
                if other_object is not new_parking_spot:
                    collision_detected, _, _ = new_parking_spot._is_colliding(other_object, 0)
                    if collision_detected:
                        break


            if collision_detected:
                self.road.objects.remove(new_parking_spot)
                num_parking_spots -= 1
                segment_indices_i += 1
                continue


            segment_indices_i += 1
            
        if num_parking_spots < num_spots:
            print(f"INSUFFICIENT SPOTS FOUND; {num_parking_spots} / {num_spots} parking spots generated")

    
    def _reward(self, action: Action) -> float:
        return 0.0
    
    def detect_object_lane_collision(self, object : RoadObject):
        gridpoints = set()
        for pt in object.polygon():
            gridpoints.add(point_to_gridpoint(pt, self.road.network.partition_gridsize))

        proximal_lanes = set()
        for gpt in gridpoints:
            proximal_lanes.update(get_proximal_lanes_wrt_gridpoint(self.road.network.grid_to_lanes, gpt))
        
        for lane_index in proximal_lanes:
            lane = self.road.network.get_lane(lane_index)

            left_pairs = zip(lane.left_boundary_points, lane.left_boundary_points[1:])
            right_pairs = zip(lane.right_boundary_points, lane.right_boundary_points[1:])
            
            for p0, p1 in chain(left_pairs, right_pairs):
                if object.intersects_with_line(p0, p1):
                    return True

        return False
        

    def _is_terminated(self):
        return self.detect_object_lane_collision(self.vehicle)


    def _is_truncated(self):
        return False
