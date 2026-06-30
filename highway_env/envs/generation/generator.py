import random
from noise import pnoise2
import numpy as np
import sys
import pprint
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from copy import deepcopy

from highway_env.road.spline import LinearSpline2D

l_to_i = {'start': 0, 'end': -1}


def default_params():
    return {
        'target_num_endpoints': 50, 
        'forward_speed': 10,
        'age_of_maturity': 4,
        'lane_width': 10,
        'perlin_variation_params': {
            'jitteriness': {
                'x': random.randint(0,10000),
                'y': random.randint(0,10000),
                'upper': 0.25,
                'lower': 0.1
            },
            'max_turn_speed': {
                'x': random.randint(0,10000),
                'y': random.randint(0,10000),
                'upper': 4.0,
                'lower': 0.01
            },
            'replication_chance': {
                'x': random.randint(0,10000),
                'y': random.randint(0,10000),
                'upper': 0.7,
                'lower': 0.0
            },
            'spontaneous_death_chance': {
                'x': random.randint(0,10000),
                'y': random.randint(0,10000),
                'upper': 0.0,
                'lower': 0.0
            }
        }
    }


def generate_random_lanes(params = None):
    if params is None:
        params = default_params()

    
    merge_radius = params['forward_speed'] * 2
    prevent_replication_radius = params['age_of_maturity'] * params['forward_speed']

    twist_iterations = 2*params['forward_speed']
    twist_step = 0.0002/params['forward_speed']
    
    lanes = generate_rough_map(
        target_num_endpoints = max(2, params['target_num_endpoints']),
        forward_speed = params['forward_speed'],
        merge_radius = merge_radius,
        prevent_replication_radius=prevent_replication_radius,
        age_of_maturity = params['age_of_maturity'],
        perlin_variation_params = params['perlin_variation_params']
    )

    rectify_map(lanes, merge_radius = merge_radius, forward_speed = params['forward_speed'])

    twist_optimize(lanes, iterations = twist_iterations, step = twist_step, lane_width = params['lane_width'])


    generate_lane_boundaries(lanes, params['lane_width'])

    for node in get_nodeset(lanes):
        correct_junction_boundaries(lanes, node)
        seal_dead_end(lanes, node)
    

    # Lane network correction loop

    invalids = get_invalid_lanes(lanes, params['forward_speed'])

    print(f"Removing {len(invalids)} obstructed lanes")

    kil(lanes, invalids)
    """
    while len(invalids) > 0 and len(lanes) > 0:
        # Surgically removing these invalid lanes
        affected_nodes = kil(lanes, invalids)

        # Getting all lanes that had a connection with these removed lanes
        affected_lane_ids = []
        for node in affected_nodes.keys():
            affected_lane_ids += [ep['id'] for ep in getRadiallySortedEndpoints(lanes, node)]
        
        # Expanding our reach to the lanes in the local neighborhood
        lane_to_grid, grid_to_lanes = lanes_spatial_hash(lanes, gridsize = 50)
        proximal_lanes = set()
        for laneID in affected_lane_ids:
            proximal_lanes.update(get_proximal_lanes_wrt_lane(laneID, lane_to_grid, grid_to_lanes))

        
        lanes_to_check = []
        for laneID in proximal_lanes:
            lanes_to_check.append(lanes[laneID])

        invalids = get_invalid_lanes(lanes_to_check, params['forward_speed'])
    
    

    # In the process of lane removal, there is a small chance that there is a dead-end left that was not sealed properly
    for node in get_nodeset(lanes):
        seal_dead_end(lanes, node)
    #"""
    
    remove_disjoint_clusters(lanes)

    


    return lanes
    


def get_nodeset(lanes):
    nodeset = set()
    for lane in lanes:
        nodeset.add(lane['start'])
        nodeset.add(lane['end'])

    return nodeset


def remove_disjoint_clusters(lanes):
    nodeset = get_nodeset(lanes)

    partition = []
    while len(nodeset) > 0:
        traversed = traverse_lane_graph(lanes, next(iter(nodeset)))
        partition.append(traversed)
        nodeset -= traversed

    for partition_element in partition:
        if len(partition_element) > len(nodeset):
            nodeset = partition_element

    # nodeset now contains the largest partition-element. We must remove all lanes who does not connect to these nodes
    lane_ids_to_remove = []
    for laneID, lane in enumerate(lanes):
        if not lane['start'] in nodeset:
            lane_ids_to_remove.append(laneID)

    for laneID in reversed(lane_ids_to_remove):
        lanes.pop(laneID)

def traverse_lane_graph(lanes, node):
    nodeset = {node}
    laneset = set()

    prev_laneset_size = -1
    while len(laneset) != prev_laneset_size:
        prev_laneset_size = len(laneset)
        for laneID, lane in enumerate(lanes):
            if lane['start'] in nodeset or lane['end'] in nodeset:
                nodeset.add(lane['start'])
                nodeset.add(lane['end'])

                laneset.add(laneID)

    return nodeset


def get_invalid_lanes(lanes, forward_speed): 
    gridsize = 20
    _, grid_to_lanes = lanes_spatial_hash(lanes, gridsize)
    
    invalids = []

    for lane in tqdm(lanes, desc = "Checking lanes for blockages"):
        start_junction_pos = getJunctionPos(lanes, lane['start'])
        end_junction_pos = getJunctionPos(lanes, lane['end'])
        valid = determine_lane_validity(lanes, lane, 
            start_junction_pos, end_junction_pos,
            grid_to_lanes, gridsize,
            forward_speed
        )

        if not valid:
            invalids.append(lane)

    return invalids
    

def determine_lane_validity(lanes, lane, start_pt, end_pt, grid_to_lanes, gridsize, forward_speed): #forward_speed is needed because it allows us to gauge the length of a lane easily
    # Generate potential fields that draws car-width-sized particles through the tunnel to find out if it is traversible or not
    lane_length = len(lane['points'])*forward_speed

    pathway = [start_pt] + lane['points'] + [end_pt]
    if np.array_equal(pathway[0], pathway[1]):
        pathway.pop(0)
    if np.array_equal(pathway[-1], pathway[-2]):
        pathway.pop()

    


    ### FORCES
    # Pulling force: leads ball to end of lane
    # Repelling force: pushes ball from proximal line barriers
    # Friction
    # Inelastic line barrier collisions

    ball_radius = 2 # CONSTRAINT: 2*ball_radius > vehicle width
    pull_force = 0.3/4 # CONSTRAINT: pull_force/friction <= ball_radius
    friction = 0.2/4 
    repel_force = ball_radius * pull_force # CONSTRAINT: repel_force/ball_radius <= pull_force
    repel_radius = 5 # radius at which the ball runs away from the walls 
    cross_particle_repel_force = repel_force

    average_speed = (pull_force/friction) 

    death_timestep_threshold = 20 # if you are in the same spot (within repel_radius distance away) after this much time, you are considered dead.
    
    
    particles = []
    max_timesteps_cap = int(5 * lane_length / average_speed)
    timesteps_before_particle_spam = int(1.5 * lane_length/ average_speed)
    max_population = 10
    timesteps_per_history_update = 1
    #We will start out with one particle. If enough time passes and we still haven't reached the goal yet, we start recruiting a bunch more particles for further exploration

    """
    print("death_timestep_threshold:", death_timestep_threshold)
    print("max_timesteps_cap:", max_timesteps_cap)
    print("timesteps_before_particle_spam:", timesteps_before_particle_spam)
    """
    

    # Simulation is run until one of the following conditions is met:
    #  A: particle reaches the goal
    #  B: all particles become trapped
    # or C: maximum allotted timesteps is reached
    
    reached_goal = False
    for timestep in range(max_timesteps_cap):
        #Try to spawn in a particle
        if len(particles)==0 or (len(particles) < max_population and timestep > timesteps_before_particle_spam):
            proposed_particle = {
                'pos': start_pt.copy(),
                'vel': np.array([0.0, 0.0]) if len(particles)== 0 else np.random.uniform(-0.5, 0.5, 2),
                'hist': []
            }

            occupied = False
            for par in particles:
                if np.linalg.norm(par['pos'] - proposed_particle['pos']) < ball_radius:
                    ocuppied = True
            
            if not occupied:
                particles.append(proposed_particle)

        
        for par in particles:
            #calculating pull vector
            closest_i = 0
            closest_dist = np.linalg.norm(par['pos']-pathway[closest_i])
            for i, pt in enumerate(pathway):
                if i == 0 or i == len(pathway) - 1: 
                    continue
                    
                dist = np.linalg.norm(par['pos'] - pt)
                if dist < closest_dist:
                    closest_i = i
                    closest_dist = dist

            if closest_i == len(pathway) - 2:
                pull_vector = pathway[closest_i+1] - par['pos']
            else:
                pull_vector = pathway[closest_i+1] - pathway[closest_i]

            pull_vector *= pull_force / np.linalg.norm(pull_vector)
            par['vel'] += pull_vector


            #print('pull_vec:', pull_vector)


            #Computing lane segment collisions + repell force
            repel_vector = np.array([0.0, 0.0])
            gridpoint = point_to_gridpoint(par['pos'], gridsize)
            proximal_lanes = get_proximal_lanes_wrt_gridpoint(grid_to_lanes, gridpoint, extended=True)

            #proximal_lanes.add(len(lanes))
            for other_laneID in proximal_lanes:
                #todo: remove this if statement debug thing
                if other_laneID < len(lanes):
                    other_lane = lanes[other_laneID]
                else:
                    other_lane = {
                        'left_points':  [],
                        'right_points': [np.array([100,100]), np.array([119.8,83.4])]
                    }
                left_pairs = zip(other_lane['left_points'], other_lane['left_points'][1:])
                right_pairs = zip(other_lane['right_points'], other_lane['right_points'][1:])
                for a, b in chain(left_pairs, right_pairs):
                    ab = b - a
                    ap = par['pos'] - a
                    ab_sq_len = np.sum(ab**2)
                    if ab_sq_len == 0:
                        to_ball = ap
                        distance =  np.linalg.norm(ap)
                    else:
                        t = np.dot(ap, ab) / ab_sq_len
                        t_clamped = np.clip(t, 0.0, 1.0)
                        closest_point = a + t_clamped * ab
                        to_ball = par['pos'] - closest_point
                        distance =  np.linalg.norm(to_ball)
                    

                    if distance < repel_radius:
                        repel_vector += to_ball * repel_force / max(distance, ball_radius)**2

                    if distance < ball_radius:
                        if distance == 0:
                            if ab_sq_len == 0: # should really never happen
                                if np.sum(par['vel']**2) > 0:
                                    normal = -par['vel']
                                else:
                                    normal = np.array([1.0, 1.0])
                            else:
                                normal = np.array([-ab[1], ab[0]])
                            
                            normal /= np.linalg.norm(normal)
                        else:
                            normal = to_ball / distance

                        #Adjust position
                        par['pos'] += normal * (ball_radius - distance)

                        #Cancel velocity
                        vel_normal_magnitude = np.dot(par['vel'], normal)
                        if vel_normal_magnitude < 0:
                            par['vel'] -= vel_normal_magnitude * normal

            # Computing repelling from other particles
            for other_par in particles:
                if par is not other_par:
                    vec = par['pos']-other_par['pos']
                    vec *= cross_particle_repel_force / max(np.linalg.norm(vec), ball_radius)**2
                    repel_vector += vec


            par['vel'] += repel_vector

            par['vel'] *= (1-friction) 

            par['pos'] += par['vel']

            if timestep % timesteps_per_history_update == 0:
                par['hist'].append(par['pos'].copy())


            # Check if we are near the goal
            if np.linalg.norm(par['pos'] - pathway[-1]) < ball_radius:
                reached_goal = True
                break

        if reached_goal:
            break
    
        # check for sign of life
        if len(particles) == max_population:
            someones_still_alive = False

            indices_past = int(death_timestep_threshold/timesteps_per_history_update)
            
            for par in particles:
                if len(par['hist']) < indices_past:
                    someones_still_alive = True
                    break

                displacement = np.linalg.norm(par['pos']-par['hist'][-indices_past])
                if displacement >= repel_radius:
                    someones_still_alive = True
                    break



    return reached_goal

    """
    for par in particles:
        for pos in par['hist']:
            print(pos[0], pos[1])

    print()
    print_single_lane_points(lane)
    print()
    """




def kil(lanes, lanes_to_kil):
    lane_ids_to_kil = [i for i, lane in enumerate(lanes) if id(lane) in {id(l) for l in lanes_to_kil}]
    lane_ids_to_kil.sort(reverse=True)
    
    affected_nodes = defaultdict(list)
    for laneID in lane_ids_to_kil:
        for loc in ['start', 'end']:
            node = lanes[laneID][loc]
            if len(getRadiallySortedEndpoints(lanes, node)) > 1:
                affected_nodes[node].append((lanes[laneID]['left_points'][l_to_i[loc]], lanes[laneID]['right_points'][l_to_i[loc]]))


    for laneID in lane_ids_to_kil:
        del lanes[laneID]    

    for node, segments in affected_nodes.items():
        junction = getRadiallySortedEndpoints(lanes, node)
        if len(junction) != 0:
            for segment in segments:
                closest_ep = None
                closest_side = None
                closest_dist = None
                first_point_is_p0 = False #false means the first point is p0. True means the first point is p1
                for ep in junction:
                    for side in ['left_points', 'right_points']:
                        point = lanes[ep['id']][side][l_to_i[ep['loc']]]
                        dist0 = np.linalg.norm(segment[0]-point)
                        dist1 = np.linalg.norm(segment[1]-point)
                        if closest_dist is None or dist0 < closest_dist or dist1 < closest_dist:
                            closest_ep = ep
                            closest_side = side
                            closest_dist = min(dist0,dist1)
                            first_point_is_p0 = dist0 < dist1

                if closest_ep['loc'] == 'start':
                    lanes[closest_ep['id']][closest_side].insert(0, segment[1] if first_point_is_p0 else segment[0])
                else:
                    lanes[closest_ep['id']][closest_side].append(segment[1] if first_point_is_p0 else segment[0])

    
    return affected_nodes # a defaultdict


def generate_rough_map(target_num_endpoints, forward_speed, merge_radius, prevent_replication_radius, age_of_maturity, perlin_variation_params):
                    #forward_speed, jitteriness, max_turn_speed,
                    #replication_chance, spontaneous_death_chance,
                    #merge_radius, prevent_replication_radius, age_of_maturity):
    class PerlinVariation:
        scale = 200
        octaves = 1
        persistence = 0.1
        lacunarity = 2.0

        @staticmethod
        def paramAt(param, pos):
            x = perlin_variation_params[param]['x']
            y = perlin_variation_params[param]['y']
            upper = perlin_variation_params[param]['upper']
            lower = perlin_variation_params[param]['lower']
            noise_val = pnoise2(
                (pos[0]/PerlinVariation.scale) + x,
                (pos[1]/PerlinVariation.scale) + y,
                octaves = PerlinVariation.octaves, persistence = PerlinVariation.persistence, lacunarity = PerlinVariation.lacunarity
            )
            return (((upper-lower)*noise_val*abs(noise_val))+upper+lower)/2.0

        
    class ConstructionAgent: 
        fork_angles = [-np.pi/2, 0, np.pi/2]

        fork_possibilities = [
            [0,0,1],[0,1,0],[0,1,1],
            [1,0,0], [1,0,1], [1,1,0],
            [1,1,1]
        ]
        
        def __init__(self, start_location,
                    position=None, orientation=0, angular_velocity=0):
            if position is None:
                self.position = np.array([0.0, 0.0])
            else:
                self.position = position
            
            self.start_location = str(start_location)
            self.end_location = str(-1)
            self.orientation = orientation
            self.angular_velocity = angular_velocity
            self.history = []
            

        def step(self):
            jitteriness = PerlinVariation.paramAt('jitteriness', self.position)
            max_turn_speed = PerlinVariation.paramAt('max_turn_speed', self.position)

            turn_friction = 1-jitteriness
            turn_acceleration_range = max_turn_speed * (1/turn_friction - 1)

            self.position += np.array([np.cos(self.orientation), np.sin(self.orientation)]) * forward_speed

            self.angular_velocity += turn_acceleration_range*(random.random()-0.5)
            self.angular_velocity *= turn_friction 
            self.orientation += self.angular_velocity

            self.history.append(self.position.copy())


    
    

    ###Lane data-structure
    #points (list of tuples)
    #start (str)
    #end (str)
    spatial_hash_gridsize = max(50, prevent_replication_radius)
    
    while True:
        lanes = []
        grid_to_lanes = defaultdict(set)
        agents = [ConstructionAgent(start_location = 0)]
        num_locations = 1
        simulation_step = 0

        while num_locations < target_num_endpoints and len(agents) > 0:
            print(f"Step {simulation_step} - Population: {len(agents)}; num_locations: {num_locations}")
            
            for agent in agents:
                #print(f"\tAgent position: {agent.position}")
                agent.step()

            agents_to_remove = []
            agents_to_add = []
            for agent in agents:
                if len(agent.history) <= age_of_maturity:
                    continue # we are not yet old enough for merging or replication

                # Death of agent if running into other road
                prevent_replication = False
                merge_enacted = False

                #Checking the histories of other agents
                for other_agent in agents:
                    for i, position in enumerate(other_agent.history):
                        if agent is other_agent and i >= len(agent.history) - age_of_maturity:
                            break
                        
                        dist = np.linalg.norm(position - agent.position)
                        if (agent is not other_agent and dist < prevent_replication_radius):
                            prevent_replication = True
                        if (dist < merge_radius):
                            merge_enacted = True
                            break
                
                #Checking placed lanes left behind by dead agents
                if not merge_enacted:
                    gridpoint = point_to_gridpoint(agent.position, spatial_hash_gridsize)
                    proximal_lanes = get_proximal_lanes_wrt_gridpoint(grid_to_lanes, gridpoint)
                    for laneID in proximal_lanes: 
                        lane = lanes[laneID] #RARE ENCOUNTERED ERROR: id out of range
                        for i, position in enumerate(lane['points']):
                            dist = np.linalg.norm(position - agent.position)
                            if (dist < prevent_replication_radius):
                                prevent_replication = True
                            if (dist < merge_radius):
                                merge_enacted = True
                                break                            
                
                #Death due to merging or spontaneous death chance
                true_population = len(agents) + len(agents_to_add) - len(agents_to_remove)
                spontaneous_death_chance = PerlinVariation.paramAt('spontaneous_death_chance', agent.position)

                if merge_enacted or (true_population > 3 and random.random() < spontaneous_death_chance):
                    agents_to_remove.append(agent)
                    agent.end_location = str(num_locations)
                    num_locations += 1
                    continue

                if prevent_replication:
                    continue
                
                replication_chance = PerlinVariation.paramAt('replication_chance', agent.position)
                #Replication
                if random.random() < replication_chance:
                    agent.end_location = str(num_locations)
                    agents_to_remove.append(agent)

                    fork_config = random.choice(ConstructionAgent.fork_possibilities)
                    for i, angle in enumerate(ConstructionAgent.fork_angles):
                        if fork_config[i] == 1 or true_population < 3:
                            newAgent = ConstructionAgent(
                                start_location = num_locations,
                                position = agent.position.copy(), 
                                orientation = agent.orientation + angle
                            )

                            agents_to_add.append(newAgent)

                    num_locations += 1
            
            for dying_agent in agents_to_remove:
                #'crystalizing' agent into lane data
                newLane = {
                    'points': dying_agent.history[:-1],
                    'start': dying_agent.start_location,
                    'end': dying_agent.end_location
                }

                for point in newLane['points']:
                    gridpoint = point_to_gridpoint(point, spatial_hash_gridsize)
                    grid_to_lanes[gridpoint].add(len(lanes))
                
                lanes.append(newLane)
                agents.remove(dying_agent)
            for new_agent in agents_to_add:
                agents.append(new_agent)

            simulation_step+=1

        #Crystalizing all still existing agents
        for agent in agents:
            newLane = {
                'points': agent.history,
                'start': agent.start_location,
                'end': str(num_locations)
            }
            lanes.append(newLane)

            num_locations += 1


        if num_locations >= target_num_endpoints:
            break

    return lanes


def rectify_short_lanes(lanes):
    lanes_to_remove = []
    for lane in lanes:
        if len(lane['points']) <= 1:
            lanes_to_remove.append(lane)
        elif len(lane['points']) == 2:
            a = lane['points'][0]
            b = lane['points'][1]
            lane['points'].insert(1, (a+b)/2)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]



def combine_nodes(lanes, merge_radius = 20, mark = False):
    lane_to_grid, grid_to_lanes = lanes_spatial_hash(lanes, gridsize = max(merge_radius, 50), use_boundaries= False)

    if mark:
        conjoined_nodes = []
    else:
        node_power = defaultdict(int)

    for laneID, lane in enumerate(tqdm(lanes, desc = "Merging nodes")):
        proximal_lanes = get_proximal_lanes_wrt_lane(laneID, lane_to_grid, grid_to_lanes, extended = True)
        for other_id in proximal_lanes:
            other_lane = lanes[other_id]
            for loc in ['start', 'end']:
                for other_loc in ['start', 'end']:
                    p0 = lane['points'][l_to_i[loc]]
                    p1 = other_lane['points'][l_to_i[other_loc]]
                    dist = np.linalg.norm(p0-p1)
                    if dist < merge_radius: 
                        if mark:
                            # We need to first ensure that no lane runs in between these two nodes
                            obstruction_found = False
                            for foreign_id in proximal_lanes:
                                foreign_lane = lanes[foreign_id]
                                pos_pairs = zip(foreign_lane['points'], foreign_lane['points'][1:])
                                for fp0, fp1 in pos_pairs:
                                    t_a, t_b = line_intersection_t(p0, p1-p0, fp0, fp1-fp0)
                                    if t_a > 0.01 and t_a < 0.99 and t_b > 0.01 and t_b < 0.99:
                                        obstruction_found = True
                                        break
                            if not obstruction_found:
                                conjoined_nodes.append(lane[loc])
                        else:
                            if node_power[other_lane[other_loc]] > node_power[lane[loc]]:
                                lane[loc] = other_lane[other_loc]
                                node_power[other_lane[other_loc]]+=1
                            else:
                                other_lane[other_loc] = lane[loc]
                                node_power[lane[loc]]+=1
    
    if mark:
        return conjoined_nodes

                            


def split_lanes(lanes, conjoined_nodes, merge_radius, forward_speed):
    #Splitting of lanes who have nodes ramming into them
    cutoff_length = np.ceil(merge_radius*2.0/forward_speed)
    lanes_to_add = []
    for lane in tqdm(lanes, desc = "Creating intersections between proximal lanes"):
        if len(lane['points']) == 0:
            continue
        for loc in ['start', 'end']:
            if lane[loc] in conjoined_nodes:
                continue
            loc_pos = lane['points'][l_to_i[loc]]

            for other_lane in lanes:
                if lane.get('parent') is other_lane or other_lane.get('parent') is lane:
                    continue
                found_index = -1
                closest_dist = sys.maxsize
                for i, pos in enumerate(other_lane['points']):
                    dist = np.linalg.norm(pos - loc_pos)
                    if (lane is not other_lane or (i > cutoff_length and i < len(lane['points']) - cutoff_length)) and dist < closest_dist:
                        found_index = i
                        closest_dist = dist
                
                
                if closest_dist < merge_radius:
                    if found_index < 2:
                        found_index = 2
                    if found_index > len(other_lane['points'])-2:
                        found_index = len(other_lane['points'])-2

                    #'old' meaning bottom half of points (more ancient part of agent history)
                    older_half = other_lane['points'][:found_index]
                    other_lane['points'] = other_lane['points'][found_index:]
                    old_start = other_lane['start']
                    other_lane['start'] = lane[loc]

                    lanes.append({
                        'points': older_half,
                        'start': old_start,
                        'end': lane[loc],
                        'parent': other_lane
                    })

                    conjoined_nodes.append(lane[loc])
                    #print(lane[loc])
                    break
        

    for lane in lanes:
        lane.pop('parent', None)


def remove_identical_reference_lanes(lanes):
    #Removing lanes whose start and end location is the same
    lanes_to_remove = []
    for lane in lanes:
        if lane['start'] == lane['end']:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def line_intersection_t(a, av, b, bv):
    A = np.column_stack((av, -bv))
    B = b-a

    try:
        t_a, t_b = np.linalg.solve(A, B)
        return t_a, t_b
    except np.linalg.LinAlgError:
        return 0.0, 0.0



def do_line_segments_intersect(a0, a1, b0, b1):
    t_a, t_b = line_intersection_t(a0, a1-a0, b0, b1-b0)
    return t_a >= 0 and t_a <= 1 and t_b >= 0 and t_b <= 1



def find_line_intersection(a, av, b, bv, return_t = False):
    t_a, t_b = line_intersection_t(a,av, b, bv)

    pt =  a + (av * t_a)

    if return_t:
        return pt, t_a, t_b
    else:
        return pt


def prune_intersecting_lanes(lanes):
    lane_to_grid, grid_to_lanes = lanes_spatial_hash(lanes, gridsize = 50, use_boundaries= False)
    
    # Pruning those lanes which intersect
    lanes_to_remove = []
    for laneID, lane in enumerate(tqdm(lanes, desc="Pruning Intersecting Lanes...")):
        proximal_lanes = get_proximal_lanes_wrt_lane(laneID, lane_to_grid, grid_to_lanes)
        collision_detected = False
        for other_id in proximal_lanes:
            if laneID < other_id:
                other_lane = lanes[other_id]                    
                pairs = zip(lane['points'], lane['points'][1:])
                for p0, p1 in pairs:
                    other_pairs = zip(other_lane['points'], other_lane['points'][1:])
                    for op0, op1 in other_pairs:
                        if do_line_segments_intersect(p0, p1, op0, op1):
                            collision_detected = True
                            break
                if collision_detected:
                    break
        if collision_detected:
            lanes_to_remove.append(lane)


    for dying_lane in lanes_to_remove:
        lanes[:] = [lane for lane in lanes if lane is not dying_lane]


def rectify_map(lanes, merge_radius, forward_speed):
    rectify_short_lanes(lanes)
    conjoined_nodes = combine_nodes(lanes, merge_radius, mark = True)
    split_lanes(lanes, conjoined_nodes, merge_radius = merge_radius, forward_speed = forward_speed)
    rectify_short_lanes(lanes) # again
    combine_nodes(lanes, merge_radius)
    remove_identical_reference_lanes(lanes)
    prune_intersecting_lanes(lanes)

    return lanes




def getEpPos(lanes, ep):
    return lanes[ep['id']]['points'][l_to_i[ep['loc']]]



def getEpVectorRaw(lanes, ep):
    pos = lanes[ep['id']]['points'][l_to_i[ep['loc']]]

    if ep['loc'] == 'start':
        pos2 = lanes[ep['id']]['points'][1]
    else:
        pos2 = lanes[ep['id']]['points'][-2]

    return pos-pos2

def getEpVector(lanes, ep):
    #Vector that denotes direction endpoint is facing
    vec = getEpVectorRaw(lanes, ep)
    
    return vec/np.linalg.norm(vec)



def getRadiallySortedEndpoints(lanes, node):
    endpoints = []

    for i, lane in enumerate(lanes):
        for loc in ['start', 'end']:
            if lane[loc] == node:
                endpoints.append({
                    'id': i,
                    'loc': loc
                })

    if len(endpoints)==0:
        return []

    midpoint = np.array([0.0,0.0])
    for ep in endpoints:
        pos = getEpPos(lanes, ep)
        midpoint += pos
        
    midpoint /= len(endpoints)



    def getTheta(ep):
        pos = getEpPos(lanes, ep) - midpoint
        return np.arctan2(pos[1], pos[0])
    

    endpoints.sort(key = getTheta)

    return endpoints




def i_to_index(lanes, ep, n, i):
    # Maps i in x(i) or y(i) to point indices
    #i = 0 is the 'trunk' or base point
        #for 'start', index becomes n
        #for 'end', index becomes len(points)-n-1 
    #i = n is the final point
        #for 'start', index becomes 0
        #for 'end', index becomes len(points)-1

    if ep['loc'] == 'start':
        return n-i
    else:
        return i + len(lanes[ep['id']]['points'])-n-1
    



def getPolarSequence(lanes, ep, n):
    polarCoordSequence = [(-999,-999)] #the 0th index of this is invalid
    for i in range(1,n+1):
        pos0 = lanes[ep['id']]['points'][i_to_index(lanes, ep, n, i)]
        pos1 = lanes[ep['id']]['points'][i_to_index(lanes, ep, n, i-1)] #pos1 is 'closer to the base' than pos0
        vec = pos0 - pos1

        r = np.linalg.norm(vec)
        theta = np.atan2(vec[1], vec[0])
                
        polarCoordSequence.append((theta, r)) #turning a constant amount 

    return polarCoordSequence





def twistEndpoint(lanes, ep, angle, n = 3):
    #registering current anglature
    polarCoordSequence = getPolarSequence(lanes, ep, n)

    for i, (theta, r) in enumerate(polarCoordSequence):
        if i == 0:
            continue
        polarCoordSequence[i] = (theta + angle *i, r)


    #converting back to regular points
    for i in range (1, n+1):
        index = i_to_index(lanes, ep, n, i)
        base_index = i_to_index(lanes, ep, n, i-1)

        x_offset = np.cos(polarCoordSequence[i][0]) * polarCoordSequence[i][1]
        y_offset = np.sin(polarCoordSequence[i][0]) * polarCoordSequence[i][1]
        base_point = lanes[ep['id']]['points'][base_index]

        lanes[ep['id']]['points'][index] = base_point + np.array([x_offset, y_offset])

    

def stepTwistGradient(lanes, junction, ep, step = 0.0002, n = 3, r = 3):
    if len(junction) <= 1:
        return 0
    #Minimize Loss: (x(a) - x_t)^2 + (y(a) - y_t)^2     (With respect to a)
    #   x_t and y_t are the average of all endpoints in the junction
    #   (x(a), y(a)) means the endpoint transformed after applying a twist of angle offset a
    #   x(a) and y(a) can be expressed as recursive functions
    #       x(i, a) = r_i * cos(theta_i + a) + x(i-1, a)
    #       x(0) = anchor point
    #   The derivative of these two functions need to be computed iteratively
    #   Also, the derivative will be computed at a = 0
    #   

    # Computing x_t and y_t, x(a), and y(a)
    x_t = 0
    y_t = 0
    x_a = 0
    y_a = 0
    theta_a = 0

    
    for endpoint in junction:
        pos = getEpPos(lanes, endpoint)
        vec = getEpVector(lanes, endpoint)

        if endpoint is ep:
            x_a = pos[0]
            y_a = pos[1]
            theta_a = np.atan2(vec[1], vec[0])
        else:
            x_t += pos[0]
            y_t += pos[1]

            x_t += r * vec[0]
            y_t += r * vec[1]

        
    x_t /= len(junction)-1
    y_t /= len(junction)-1

    # Computing x'(a, n) and x'(a, n): 
    #   x'(i) = -r_i * sin(theta_i) + x'(i-1)
    #   y'(i) = r_i * cos(theta_i) + y'(i-1)
    #   x'(0) = y'(0) = 0

    # theta_i and r_i are given by getPolarSequence
    polarSequence = getPolarSequence(lanes, ep, n)

    x_a_derivative = 0
    y_a_derivative = 0
    for i in range (1, n+1):
        c = polarSequence[i]
        x_a_derivative += -c[1] * np.sin(c[0])
        y_a_derivative += c[1] * np.cos(c[0])

    theta_a_derivative = n


    #Computing the loss gradient
    # L' =  (x(a) + rcos(theta(a)) - x_t)(x'(a) - (rsin(theta(a)) * theta'(a)))
    #       + (y(a) + rsin(theta(a)) - y_t)(y'(a) + (rcos(theta(a)) * theta'(a)))
    #old: L' = 2(x(a) - x_t) * x'(a) + 2(y(a) - y_t) * y'(a)

    negative_loss_gradient = (x_a + r*np.cos(theta_a) - x_t) * (x_a_derivative - (r*np.sin(theta_a) * theta_a_derivative)) \
                            + (y_a + r*np.sin(theta_a) - y_t) * (y_a_derivative + (r*np.cos(theta_a) * theta_a_derivative))
    negative_loss_gradient *= -1
    #negative_loss_gradient = 2*((x_t - x_a) *x_a_derivative + (y_t - y_a) *y_a_derivative)
    #print(f"computed gradient: {negative_loss_gradient}" )
    #print(f"* step size: {step*negative_loss_gradient}" )
    

    twistEndpoint(lanes, ep, step*negative_loss_gradient, n)



            
def rotate_optimize(lanes, n = 3):
    # Rotate optimization (for lanes that are too short to be twisted)           
    for laneID, lane in enumerate(lanes):
        if len(lane['points']) <= n:
            startJunction = getRadiallySortedEndpoints(lanes, lane['start'])
            endJunction = getRadiallySortedEndpoints(lanes, lane['end'])

            start = np.array([0.0, 0.0])
            if len(startJunction) > 1:
                for ep in startJunction:
                    if ep['id'] != laneID:
                        start += getEpPos(lanes, ep)

                start /= len(startJunction)-1
            else:
                start = getEpPos(lanes, startJunction[0])

            end = np.array([0.0, 0.0])

            if len(endJunction) > 1:
                for ep in endJunction:
                    if ep['id'] != laneID:
                        end += getEpPos(lanes, ep)
                end /= len(endJunction)-1
            else:
                end = getEpPos(lanes, endJunction[0])


            for i in range(len(lane['points'])):
                num_pts = len(lane['points'])
                lane['points'][i] = (end-start) * ((i+1)/(num_pts+1)) + start

def squish_optimize(lanes, junction, r):
     #Squish optimization:
    for ep in junction:
        if len(junction) <= 1: 
            continue

        mid = np.array([0.0, 0.0])
        for other_ep in junction:
            if ep is not other_ep:
                #vec = getEpVector(lanes, other_ep)
                mid += getEpPos(lanes, other_ep)# + r * vec[0]

        mid /= len(junction)-1

        
        b = getEpVectorRaw(lanes, ep)
        b_mag_squared = b @ b
        b_mag = np.sqrt(b_mag_squared)
        offset = r*b/b_mag

        for i in range(5):
            pos = getEpPos(lanes, ep)
            a = mid-offset-pos

            a1 = a @ b / b_mag_squared

            if a1 <= 0.01:
                if len(lanes[ep['id']]['points']) > 2:
                    lanes[ep['id']]['points'].pop(l_to_i[ep['loc']])
                else:
                    break
            elif a1 < 1:
                new = (a1-1)*b + pos
                lanes[ep['id']]['points'][l_to_i[ep['loc']]] = new
            else:
                break





def prune_redundant_lanes(lanes, lane_width):
    # Removing 'redundant' lanes - 2 lanes that have the same start and end and are effectively on top of each other
    while True:
        duplicate_found = False
        for lane in lanes:
            for other_lane in lanes:
                if lane is not other_lane:
                    if (lane['start'] == other_lane['start'] and lane['end'] == other_lane['end']) or (lane['end'] == other_lane['start'] and lane['start'] == other_lane['end']):
                        number_of_points_too_close = 0
                        for point in lane['points']:
                            closest_distance = sys.maxsize
                            for point2 in other_lane['points']:
                                dist = np.linalg.norm(point - point2)
                                if dist < closest_distance:
                                    closest_distance = dist
                            if closest_distance < lane_width*1.5:
                                number_of_points_too_close += 1

                            

                        if number_of_points_too_close > 2:
                            duplicate_found = True
                            lanes[:] = [lane for lane in lanes if lane is not other_lane]
                        break
            if duplicate_found:
                break
        
        if not duplicate_found:
            break
    

def twist_optimize(lanes, iterations = 40, step = 0.00001, n = 3, lane_width = 10): 
    rotate_optimize(lanes, n)

    r = 3
    nodeset = get_nodeset(lanes)

    for node in tqdm(nodeset, desc="Twisting Endpoints..."):
        junction = getRadiallySortedEndpoints(lanes, node)

        
        for _ in range(iterations):
            for ep in junction:
                length = len(lanes[ep['id']]['points'])
                if length > n:
                    stepTwistGradient(lanes, junction, ep, step = step, n = n, r = r)
                elif length > 2:
                    stepTwistGradient(lanes, junction, ep, step = step, n = 2, r = r)

        squish_optimize(lanes, junction, r)


    prune_redundant_lanes(lanes, lane_width)




def generate_lane_boundaries(lanes, lane_width):
    for lane in lanes:
        lane['left_points'] = []
        lane['right_points'] = []

        for i, point in enumerate(lane['points']):
            d = []
            
            if i != 0:
                d.append(lane['points'][i-1] - point)
                
            if i != len(lane['points'])-1:
                d.append(lane['points'][i+1] - point)

            v = d[0].copy()
            if i != 0:
                v *= -1

            
            lat = np.array([0.0, 0.0])
            for _ in range(2): #In 99.999999% of cases this loop will run the first iteration only
                if len(d) == 1:
                    lat = np.array([-d[0][1], d[0][0]])
                    break
                elif len(d) == 2:
                    mag = np.linalg.norm(d[0])
                    d[0] /= mag
                    
                    mag = np.linalg.norm(d[1])
                    d[1] /= mag
                    
                    lat = (d[0]+d[1])/2.0
                    
                    mag = np.linalg.norm(lat)
                    if mag == 0:
                        d.pop()
                        continue
                    break
                else:
                    assert(False)

            mag = np.linalg.norm(lat)
            lat *=(lane_width/2)/mag
            


            if (lat[0] * v[1] - lat[1] * v[0]) < 0:
                lane['right_points'].append(point + lat)
                lane['left_points'].append(point - lat)
            else:
                lane['right_points'].append(point - lat)
                lane['left_points'].append(point + lat)
        

        #del lane['points']


    

            


def correct_junction_boundaries(lanes, node):
    junction = getRadiallySortedEndpoints(lanes, node)
    if (len(junction) <= 1):
            return
    #for ep in junction:
    #    print(f"loc: {ep['loc']}")
    #    print(lanes[ep['id']])
    
    # right-hand neighbor: up an index  
    # left-hand neighbor: down an index

    #Rule: Your left side should join with your left neighbor's right side
    for epID, ep in enumerate(junction):
        
        while True:
            otherEp = junction[epID-1]

            selfSide = 'right_points'
            otherSide = 'left_points'
            if ep['loc'] == 'start':
                selfSide = 'left_points'
            if otherEp['loc'] == 'start':
                otherSide = 'right_points'


            

            pos = lanes[ep['id']][selfSide][l_to_i[ep['loc']]]
            dir = getEpVector(lanes, ep)

            otherPos = lanes[otherEp['id']][otherSide][l_to_i[otherEp['loc']]]
            otherDir = getEpVector(lanes, otherEp)

            vecToOther = otherPos - pos
            dot1 = vecToOther @ dir
            dot2 = -(vecToOther @ otherDir)
            if dot1 > 0 or dot2 > 0 or len(lanes[ep['id']][selfSide]) <= 3 or len(lanes[otherEp['id']][otherSide]) <= 3 :
                break

            if dot1 <= 0:
                lanes[ep['id']][selfSide].pop(l_to_i[ep['loc']])
            if dot2 <= 0:
                lanes[otherEp['id']][otherSide].pop(l_to_i[otherEp['loc']])


        newPos = find_line_intersection(pos, dir, otherPos, otherDir)
        
        #print(type(newPos))

        b = otherPos - pos
        a = newPos - pos
        
        a1 =(a @ b) / (b @ b)

        if a1 < 0 or a1 > 1:
            newPos = (pos+otherPos)/2


        #print(f"pos: {pos}")
        #print(f"otherPos: {otherPos}")
        #print(f"newPos: {newPos}")
        lanes[ep['id']][selfSide][l_to_i[ep['loc']]] = newPos
        lanes[otherEp['id']][otherSide][l_to_i[otherEp['loc']]] = newPos


def seal_dead_end(lanes, node):
    junction = getRadiallySortedEndpoints(lanes, node)
    if (len(junction) > 1):
        return
    
    ep = junction[0]
    lane = lanes[ep['id']]  

    if ep['loc'] == 'start':
        lane['right_points'].insert(0, lane['left_points'][0])
    else:
        lane['right_points'].append(lane['left_points'][-1])

    
    i = (l_to_i[ep['loc']]*3 + 1) #transforms {-1, 0} to {-2, 1}
    lane['points'][l_to_i[ep['loc']]] = (lane['points'][l_to_i[ep['loc']]]+ lane['points'][i]) / 2
    


def getJunctionPos(lanes, node):
    junction = getRadiallySortedEndpoints(lanes, node)
    if len(junction) == 1:
        pt = getEpPos(lanes, junction[0])
    else:
        pt = np.array([0.0,0.0])

        for ep in junction:
            pt += lanes[ep['id']]['left_points'][l_to_i[ep['loc']]]
            pt += lanes[ep['id']]['right_points'][l_to_i[ep['loc']]]
        pt /= len(junction) * 2

    return pt




def point_to_gridpoint(point, gridsize):
    return tuple(np.floor(point / gridsize).astype(int))



def lanes_spatial_hash(lanes, gridsize = 100, use_boundaries = True):
    lane_to_grid = defaultdict(set)
    grid_to_lanes  = defaultdict(set)

    for laneID, lane in enumerate(lanes):
        if use_boundaries:
            pts = chain(lane['left_points'], lane['right_points'])
        else:
            pts = lane['points']
        
        last_gridpoint = None        
        for point in pts:
            gridpoint = point_to_gridpoint(point, gridsize)
            lane_to_grid[laneID].add(gridpoint)
            grid_to_lanes[gridpoint].add(laneID)

            # In the case that we traverse precisely diagonally, 'skipping' over a grid:
            if last_gridpoint is not None and gridpoint[0] != last_gridpoint[0] and gridpoint[1] != last_gridpoint[1]:
                gp1 = (gridpoint[0], last_gridpoint[1])
                gp2 = (last_gridpoint[0], gridpoint[1])
                lane_to_grid[laneID].update((gp1, gp2))
                grid_to_lanes[gp1].add(laneID)
                grid_to_lanes[gp2].add(laneID)
            last_gridpoint = gridpoint
    return lane_to_grid, grid_to_lanes


gridhash_offsets = [
    (-1,-1), (0,-1), (1,-1),
    (-1,0), (0,0), (1,0),
    (-1,1), (0,1), (1,1)
]

def get_proximal_lanes_wrt_gridpoint(grid_to_lanes, gridpoint, extended = False):
    proximal_lanes = set()
    for offset in (gridhash_offsets if extended else [(0, 0)]):
        new_point = (gridpoint[0]+offset[0], gridpoint[1]+offset[1])
        proximal_lanes.update(grid_to_lanes[new_point])
    
    return proximal_lanes


    

def get_proximal_lanes_wrt_lane(laneID, lane_to_grid, grid_to_lanes, extended = False):
    proximal_lanes = set()
    for gridpoint in lane_to_grid[laneID]:
        proximal_lanes.update(get_proximal_lanes_wrt_gridpoint(grid_to_lanes, gridpoint, extended = extended))

    proximal_lanes.discard(laneID)

    return proximal_lanes




def get_all_intersection_points(lanes, lane_to_grid, grid_to_lanes):
    intersecting_points = []
    for laneID, lane in enumerate(tqdm(lanes, desc = "Flagging intersection points")):
        proximal_lanes = get_proximal_lanes_wrt_lane(laneID, lane_to_grid, grid_to_lanes)
        for other_id in proximal_lanes:
            if laneID < other_id:
                other_lane = lanes[other_id]
                left_pairs = zip(lane['left_points'], lane['left_points'][1:])
                right_pairs = zip(lane['right_points'], lane['right_points'][1:])
                for p0, p1 in chain(left_pairs, right_pairs):
                    other_left_pairs = zip(other_lane['left_points'], other_lane['left_points'][1:])
                    other_right_pairs = zip(other_lane['right_points'], other_lane['right_points'][1:])
                    for op0, op1 in chain(other_left_pairs, other_right_pairs):
                        t_a, t_b = line_intersection_t(p0, p1-p0, op0, op1-op0)
                        if t_a > 0.01 and t_a < 0.99 and t_b > 0.01 and t_b < 0.99:
                            intersecting_points.append(find_line_intersection(p0, p1-p0, op0, op1-op0))
    
    return intersecting_points

                    


def print_single_lane_points(lane):
    print("left: ")
    for pt in lane['left_points']:
        print(f"{pt[0]} {pt[1]}")
    print("right: ")
    for pt in lane['right_points']:
        print(f"{pt[0]} {pt[1]}")
    print()

def print_lane_points(lanes):
    for _, lane in enumerate(lanes):
        print("left: ")
        for pt in lane['left_points']:
            print(f"{pt[0]} {pt[1]}")
        print("right: ")
        for pt in lane['right_points']:
            print(f"{pt[0]} {pt[1]}")
        print()



def serialize_lanes(lanes):
    lanes_serialized = deepcopy(lanes)

    for lane in lanes_serialized:
        for i, pt in enumerate(lane['points']):
            lane['points'][i] = (pt[0], pt[1])
        if 'left_points' in lane:
            for i, pt in enumerate(lane['left_points']):
                lane['left_points'][i] = (pt[0], pt[1])
            for i, pt in enumerate(lane['right_points']):
                lane['right_points'][i] = (pt[0], pt[1])

    return lanes_serialized


def unserialize_lanes(lanes_serialized):
    lanes = deepcopy(lanes_serialized)

    for lane in lanes:
        for i, pt in enumerate(lane['points']):
            lane['points'][i] = np.array([pt[0], pt[1]])
        if 'left_points' in lane:
            for i, pt in enumerate(lane['left_points']):
                lane['left_points'][i] = np.array([pt[0], pt[1]])
            for i, pt in enumerate(lane['right_points']):
                lane['right_points'][i] = np.array([pt[0], pt[1]])

    return lanes



