import math
import random
import numpy as np
import sys
import pprint
from collections import defaultdict
from tqdm import tqdm

from highway_env.road.spline import LinearSpline2D

l_to_i = {"start": 0, "end": -1}
def tupleDist(tupleA, tupleB):
    x1, y1 = tupleA
    x2, y2 = tupleB
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generate_rough_map(target_num_endpoints, 
                    forward_speed, jitteriness, max_turn_speed,
                    replication_chance, spontaneous_death_chance,
                    merge_radius, prevent_replication_radius, age_of_maturity):
    class ConstructionAgent: 
        turn_friction = 1-jitteriness
        turn_acceleration_range = max_turn_speed * (1/turn_friction - 1)

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
            self.position[0] += math.cos(self.orientation) * forward_speed
            self.position[1] += math.sin(self.orientation) * forward_speed

            self.angular_velocity += ConstructionAgent.turn_acceleration_range*(random.random()-0.5)
            self.angular_velocity *= ConstructionAgent.turn_friction 
            self.orientation += self.angular_velocity

            self.history.append(tuple(self.position.tolist()))

    

    #agent_steps = 100
    fork_angles = [-math.pi/2, 0, math.pi/2]
    fork_possibilities = [
        [0,0,1],[0,1,0],[0,1,1],
        [1,0,0], [1,0,1], [1,1,0],
        [1,1,1]
    ]
    



    ###Lane data-structure
    #points (list of tuples)
    #start (str)
    #end (str)

    lanes = []
    agents = [ConstructionAgent(start_location = 0)]
    num_locations = 1
    i = 0

    while num_locations < target_num_endpoints and len(agents) != 0:
        print(f"Step {i} - Population: {len(agents)}; num_locations: {num_locations}")
        i+=1
        agents_to_remove = []
        agents_to_add = []
        for agent in agents:
            #print(f"\tAgent position: {agent.position}")
            agent.step()

            if (len(agent.history) <= age_of_maturity):
                continue # we are not yet old enough for merging or replication

            # Death of agent if running into other road
            prevent_replication = False
            merge_enacted = False

            for other_agent in agents:
                for i, position in enumerate(other_agent.history):
                    if agent==other_agent and i >= len(agent.history) - age_of_maturity:
                        break

                    if (agent != other_agent and tupleDist(position, agent.position) < prevent_replication_radius):
                        prevent_replication = True
                    if (tupleDist(position, agent.position) < merge_radius):
                        merge_enacted = True
                        break

            if not merge_enacted:
                for lane in lanes:
                    for i, position in enumerate(lane['points']):
                        if (tupleDist(position, agent.position) < prevent_replication_radius):
                            prevent_replication = True
                        if (tupleDist(position, agent.position) < merge_radius):
                            merge_enacted = True
                            break                            
            
            #Death due to merging or spontaneous death chance
            true_population = len(agents) + len(agents_to_add) - len(agents_to_remove)

            if merge_enacted or (true_population > 3 and random.random() < spontaneous_death_chance):
                agents_to_remove.append(agent)
                agent.end_location = str(num_locations)
                num_locations += 1
                continue

            if prevent_replication:
                continue

            #Replication
            if  random.random() < replication_chance:
                agent.end_location = str(num_locations)
                agents_to_remove.append(agent)

                fork_config = random.choice(fork_possibilities)
                for i, angle in enumerate(fork_angles):
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
            lanes.append({
                "points": dying_agent.history,
                "start": dying_agent.start_location,
                "end": dying_agent.end_location
            })
            
            agents.remove(dying_agent)
        
        for new_agent in agents_to_add:
            agents.append(new_agent)

    #Crystalizing all still existing agents
    for agent in agents:
        lanes.append({
            "points": agent.history,
            "start": agent.start_location,
            "end": str(num_locations)
        })
        num_locations += 1

        return lanes


def rectify_short_lanes(lanes):
    lanes_to_remove = []
    for lane in lanes:
        if len(lane['points']) <= 1:
            lanes_to_remove.append(lane)
        elif len(lane['points']) == 2:
            a = lane['points'][0]
            b = lane['points'][1]
            lane['points'].insert(1, ((a[0]+b[0])/2, (a[1]+b[1])/2))

    for dying_lane in lanes_to_remove:
        lanes.remove(dying_lane)



def mark_combining_nodes(lanes, merge_radius = 20):
    conjoined_nodes = []
    for lane in lanes:
        for loc in ["start", "end"]:
            for other_lane in lanes:
                if lane != other_lane:
                    for other_loc in ["start", "end"]:
                        dist = tupleDist(lane['points'][l_to_i[loc]], other_lane['points'][l_to_i[other_loc]])
                        if dist < merge_radius: 
                            conjoined_nodes.append(lane[loc])
    
    return conjoined_nodes



def combine_nodes(lanes, merge_radius = 20):
    node_power = defaultdict(int)
    for lane in lanes:
        for loc in ["start", "end"]:
            for other_lane in lanes:
                if lane != other_lane:
                    for other_loc in ["start", "end"]:
                        dist = tupleDist(lane['points'][l_to_i[loc]], other_lane['points'][l_to_i[other_loc]])
                        if dist < merge_radius:
                            if node_power[other_lane[other_loc]] > node_power[lane[loc]]:
                                lane[loc] = other_lane[other_loc]
                                node_power[other_lane[other_loc]]+=1
                            else:
                                other_lane[other_loc] = lane[loc]
                                node_power[lane[loc]]+=1


def split_lanes(lanes, conjoined_nodes, merge_radius = 20):
    #Splitting of lanes who have nodes ramming into them
    lanes_to_add = []
    for lane in lanes:
        for loc in ["start", "end"]:
            if lane[loc] in conjoined_nodes:
                continue
            loc_pos = lane['points'][l_to_i[loc]]
            for other_lane in lanes:
                found_index = -1
                if lane == other_lane:
                    continue
                for i, pos in enumerate(other_lane['points']):
                    if (tupleDist(pos, loc_pos) < merge_radius):
                        found_index = i
                        break
                
                if found_index != -1:
                    if found_index < 2:
                        found_index = 2
                    if found_index > len(other_lane['points'])-2:
                        found_index = len(other_lane['points'])-2

                    #'old' meaning bottom half of points (more ancient part of agent history)
                    older_half = other_lane['points'][:found_index]
                    other_lane['points'] = other_lane['points'][found_index:]
                    old_start = other_lane['start']
                    other_lane['start'] = lane[loc]

                    lanes_to_add.append({
                        "points": older_half,
                        "start": old_start,
                        "end": lane[loc]
                    })
                    break

    lanes += lanes_to_add

def remove_identical_reference_lanes(lanes):
    #Removing lanes whose start and end location is the same
    lanes_to_remove = []
    for lane in lanes:
        if lane['start'] == lane['end']:
            lanes_to_remove.append(lane)

    for dying_lane in lanes_to_remove:
        lanes.remove(dying_lane)


def rectify_map(lanes, merge_radius = 20, age_of_maturity = 4):
    rectify_short_lanes(lanes)

    conjoined_nodes = mark_combining_nodes(lanes, merge_radius)

    split_lanes(lanes, conjoined_nodes, merge_radius = merge_radius)

    rectify_short_lanes(lanes)
    combine_nodes(lanes, merge_radius)

    remove_identical_reference_lanes(lanes)

    return lanes



def check_for_too_short_lanes(lanes):
    for lane in lanes:
        if len(lane['points']) <= 2:
            print("YO ALERT ALERT")

def getEpPos(lanes, ep):
    return lanes[ep['id']]['points'][l_to_i[ep['loc']]]
    

def getEpVectorRaw(lanes, ep):
    pos = lanes[ep['id']]['points'][l_to_i[ep['loc']]]

    if ep['loc'] == 'start':
        pos2 = lanes[ep['id']]['points'][1]
    else:
        pos2 = lanes[ep['id']]['points'][-2]

    return (pos[0]-pos2[0], pos[1]-pos2[1])

def getEpVector(lanes, ep):
    #Vector that denotes direction endpoint is facing
    vec = getEpVectorRaw(lanes, ep)
    vec_len = tupleDist((0,0), vec)

    return (vec[0]/vec_len, vec[1]/vec_len)



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

    midpoint = [0.0,0.0]
    for ep in endpoints:
        pos = getEpPos(lanes, ep)
        midpoint[0] += pos[0]
        midpoint[1] += pos[1]
        
    midpoint[0] /= len(endpoints)
    midpoint[1] /= len(endpoints)



    def getTheta(ep):
        pos = getEpPos(lanes, ep)
        pos = (pos[0]-midpoint[0], pos[1]-midpoint[1])
        return math.atan2(pos[1], pos[0])
    

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
        vec = (pos0[0]-pos1[0], pos0[1]-pos1[1]) #points from pos 1 to pos 0

        r = tupleDist(pos0, pos1)
        theta = math.atan2(vec[1], vec[0])
                
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

        x_offset = math.cos(polarCoordSequence[i][0]) * polarCoordSequence[i][1]
        y_offset = math.sin(polarCoordSequence[i][0]) * polarCoordSequence[i][1]
        base_point = lanes[ep['id']]['points'][base_index]

        lanes[ep['id']]['points'][index] = (base_point[0] + x_offset, base_point[1] + y_offset)

    

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

        if endpoint == ep:
            x_a = pos[0]
            y_a = pos[1]
            theta_a = math.atan2(vec[1], vec[0])
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
        x_a_derivative += -c[1] * math.sin(c[0])
        y_a_derivative += c[1] * math.cos(c[0])

    theta_a_derivative = n


    #Computing the loss gradient
    # L' =  (x(a) + rcos(theta(a)) - x_t)(x'(a) - (rsin(theta(a)) * theta'(a)))
    #       + (y(a) + rsin(theta(a)) - y_t)(y'(a) + (rcos(theta(a)) * theta'(a)))
    #old: L' = 2(x(a) - x_t) * x'(a) + 2(y(a) - y_t) * y'(a)

    negative_loss_gradient = (x_a + r*math.cos(theta_a) - x_t) * (x_a_derivative - (r*math.sin(theta_a) * theta_a_derivative)) \
                            + (y_a + r*math.sin(theta_a) - y_t) * (y_a_derivative + (r*math.cos(theta_a) * theta_a_derivative))
    negative_loss_gradient *= -1
    #negative_loss_gradient = 2*((x_t - x_a) *x_a_derivative + (y_t - y_a) *y_a_derivative)
    #print(f"computed gradient: {negative_loss_gradient}" )
    #print(f"* step size: {step*negative_loss_gradient}" )
    

    twistEndpoint(lanes, ep, step*negative_loss_gradient, n)



            
def rotate_optimize(lanes, n = 3):
    # Rotate optimization (for lanes that are too short to be twisted)           
    for id, lane in enumerate(lanes):
        if len(lane['points']) <= n:
            startJunction = getRadiallySortedEndpoints(lanes, lane['start'])
            endJunction = getRadiallySortedEndpoints(lanes, lane['end'])
         
            start_x = 0
            start_y = 0
            for ep in startJunction:
                if ep['id'] != id:
                    pos = getEpPos(lanes, ep)
                    start_x += pos[0]
                    start_y += pos[1]

            start_x /= len(startJunction)-1
            start_y /= len(startJunction)-1

            end_x = 0
            end_y = 0
            for ep in endJunction:
                if ep['id'] != id:
                    pos = getEpPos(lanes, ep)
                    end_x += pos[0]
                    end_y += pos[1]
            end_x /= len(endJunction)-1
            end_y /= len(endJunction)-1


            for i in range(len(lane['points'])):
                lane['points'][i] = (
                        (end_x-start_x) * ((i+1)/(len(lane['points'])+1)) + start_x,
                        (end_y-start_y) * ((i+1)/(len(lane['points'])+1)) + start_y
                )

def squish_optimize(lanes, junction, r):
     #Squish optimization:
    for ep in junction:
        if len(junction) <= 1: 
            continue

        mid_x = mid_y = 0
        for other_ep in junction:
            if ep != other_ep:
                pos = getEpPos(lanes, other_ep)
                #vec = getEpVector(lanes, other_ep)
                mid_x += pos[0]# + r * vec[0]
                mid_y += pos[1]# + r * vec[1]
        mid_x /= len(junction)-1
        mid_y /= len(junction)-1

        
        b = getEpVectorRaw(lanes, ep)
        b_mag_squared = b[0]**2 + b[1]**2
        b_mag = math.sqrt(b_mag_squared)
        offset_x = r*b[0]/b_mag
        offset_y = r*b[1]/b_mag

        for i in range(5):
            pos = getEpPos(lanes, ep)
            a = (mid_x-offset_x -pos[0], mid_y-offset_y -pos[1])

            a1 = (a[0]*b[0] + a[1]*b[1])/b_mag_squared

            if a1 <= 0.01:
                if len(lanes[ep['id']]['points']) > 3:
                    lanes[ep['id']]['points'].pop(l_to_i[ep['loc']])
                else:
                    break
            elif a1 < 1:
                new_x = ((a1-1)*b[0]) + pos[0]
                new_y = ((a1-1)*b[1]) + pos[1]
                lanes[ep['id']]['points'][l_to_i[ep['loc']]] = (new_x, new_y)
            else:
                break


def doLineSegmentsIntersect(a0, a1, b0, b1): #each argument is a tuple
    av = (a1[0]-a0[0], a1[1]-a0[1])
    bv = (b1[0]-b0[0], b1[1]-b0[1])

    det = (av[1]*bv[0] - av[0]*bv[1])
    if det==0:
        return False

    t_a = -bv[1]*(b0[0]-a0[0]) + bv[0]*(b0[1]-a0[1]) 
    t_b = -av[1]*(b0[0]-a0[0]) + av[0]*(b0[1]-a0[1]) 
    t_a /= det
    t_b /= det

    return t_a >= 0 and t_a <= 1 and t_b >= 0 and t_b <= 1


def prune_intersecting_lanes(lanes):
    # Pruning those lanes which intersect
    lanes_to_remove = []
    for lane in tqdm(lanes, desc="Pruning Intersecting Lanes..."):
        collision_detected = False
        for other_lane in lanes:
            if lane != other_lane and other_lane not in lanes_to_remove:
                for i in range(1, len(lane['points'])):
                    for j in range(1, len(other_lane['points'])):
                        if doLineSegmentsIntersect(lane['points'][i-1], lane['points'][i],
                                                   other_lane['points'][j-1], other_lane['points'][j]):
                            collision_detected = True
                            break
                if collision_detected:
                    break
            
            if collision_detected:
                break
        
        if collision_detected:
            lanes_to_remove.append(lane)
            
    for dying_lane in lanes_to_remove:
        lanes.remove(dying_lane)


def prune_redundant_lanes(lanes, lane_width):
    # Removing 'redundant' lanes - 2 lanes that have the same start and end and are effectively on top of each other
    while True:
        duplicate_found = False
        for lane in lanes:
            for other_lane in lanes:
                if lane != other_lane:
                    if (lane['start'] == other_lane['start'] and lane['end'] == other_lane['end']) or (lane['end'] == other_lane['start'] and lane['start'] == other_lane['end']):
                        number_of_points_too_close = 0
                        for point in lane['points']:
                            closest_distance = sys.maxsize
                            for point2 in other_lane['points']:
                                dist = tupleDist(point, point2)
                                if dist < closest_distance:
                                    closest_distance = dist
                            if closest_distance < lane_width*1.5:
                                number_of_points_too_close += 1

                            

                        if number_of_points_too_close > 2:
                            duplicate_found = True
                            lanes.remove(other_lane)
                        break
            if duplicate_found:
                break
        
        if not duplicate_found:
            break
    
def twist_optimize(lanes, iterations = 40, step = 0.00001, n = 3, lane_width = 10):
    prune_intersecting_lanes(lanes)
    
    rotate_optimize(lanes, n)

    r = 3
    nodeset = set()
    for lane in lanes:
        nodeset.add(lane['start'])
        nodeset.add(lane['end'])

    for node in tqdm(nodeset, desc="Twisting Endpoints..."):
        junction = getRadiallySortedEndpoints(lanes, node)

        
        for i in range(iterations):
            for ep in junction:
                if len(lanes[ep['id']]['points']) > n:
                    stepTwistGradient(lanes, junction, ep, step = step, n = n, r = r)
                else:
                    stepTwistGradient(lanes, junction, ep, step = step, n = 2, r = r)

        squish_optimize(lanes, junction, r)


    prune_redundant_lanes(lanes, lane_width)



    return nodeset



def interpolate_tuples(t1, t2, t):
    return tuple(a + t * (b - a) for a, b in zip(t1, t2))


def generate_lane_boundaries(lanes, lane_width):
    for lane in lanes:
        curve = LinearSpline2D(lane['points'])

        lane['left_points'] = []
        lane['right_points'] = []

        for i, point in enumerate(lane['points']):
            dx = []
            dy = []
            

            if i != 0:
                dy.append(lane['points'][i-1][1] - point[1])
                dx.append(lane['points'][i-1][0] - point[0])
                
            if i != len(lane['points'])-1:
                dy.append(lane['points'][i+1][1] - point[1])
                dx.append(lane['points'][i+1][0] - point[0])

            vx = dx[0]
            vy = dy[0]
            if i != 0:
                vx = -vx
                vy = -vy

            
            
            lat_dx = lat_dy = 0
            for j in range(2): #In 99.999999% of cases this loop will run the first iteration only
                if len(dx) == 1:
                    lat_dx = -dy[0]
                    lat_dy = dx[0]

                    break
                elif len(dx) == 2:
                    mag = math.sqrt(dx[0]**2 + dy[0]**2)
                    dx[0] /= mag
                    dy[0] /= mag
                    mag = math.sqrt(dx[1]**2 + dy[1]**2)
                    dx[1] /= mag
                    dy[1] /= mag
                    lat_dx = (dx[0]+dx[1])/2.0
                    lat_dy = (dy[0]+dy[1])/2.0
                    mag = math.sqrt(lat_dx**2 + lat_dy**2)
                    if mag == 0:
                        dx.pop()
                        dy.pop()
                        continue
                    break
                else:
                    assert(False)

            mag = math.sqrt(lat_dx**2 + lat_dy**2)
            if mag == 0:
                pprint.pprint(lane)
            lat_dx *= (lane_width/2)/mag
            lat_dy *= (lane_width/2)/mag


            if lat_dx * vy - lat_dy * vx < 0:
                lane['right_points'].append((point[0]+lat_dx, point[1]+lat_dy))
                lane['left_points'].append((point[0]-lat_dx, point[1]-lat_dy))
            else:
                lane['right_points'].append((point[0]-lat_dx, point[1]-lat_dy))
                lane['left_points'].append((point[0]+lat_dx, point[1]+lat_dy))


        

def findLineIntersection(a, av, b, bv): #each argument is a tuple
    det = (av[1]*bv[0] - av[0]*bv[1])
    if det==0:
        det = sys.float_info.min

    t_a = -bv[1]*(b[0]-a[0]) + bv[0]*(b[1]-a[1]) 
    t_a /= det

    return np.array([a[0] + t_a * av[0], a[1] + t_a * av[1]])

            


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

            vecToOther = (otherPos[0]-pos[0], otherPos[1]-pos[1])
            dot1 = vecToOther[0]*dir[0] + vecToOther[1]*dir[1]
            dot2 = -(vecToOther[0]*otherDir[0] + vecToOther[1]*otherDir[1])
            if dot1 > 0 or dot2 > 0 or len(lanes[ep['id']][selfSide]) <= 3 or len(lanes[otherEp['id']][otherSide]) <= 3 :
                break

            if dot1 <= 0:
                lanes[ep['id']][selfSide].pop(l_to_i[ep['loc']])
            if dot2 <= 0:
                lanes[otherEp['id']][otherSide].pop(l_to_i[otherEp['loc']])


        newPos = findLineIntersection(pos, dir, otherPos, otherDir)
        
        #print(type(newPos))

        b = (otherPos[0]-pos[0], otherPos[1]-pos[1])
        a = (newPos[0]-pos[0], newPos[1]-pos[1])
        
        a1 = (a[0]*b[0] + a[1]*b[1]) / (b[0]**2 + b[1]**2)

        if a1 < 0 or a1 > 1:
            newPos = ((pos[0] + otherPos[0])/2, (pos[1] + otherPos[1])/2)


        #print(f"pos: {pos}")
        #print(f"otherPos: {otherPos}")
        #print(f"newPos: {newPos}")
        lanes[ep['id']][selfSide][l_to_i[ep['loc']]] = newPos
        lanes[otherEp['id']][otherSide][l_to_i[otherEp['loc']]] = newPos





        


