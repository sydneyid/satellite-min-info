"""
    Navigation for `n` agents to `n` goals from random initial positions
    With random obstacles added in the environment
    Each agent is destined to get to its own goal unlike
    `simple_spread.py` where any agent can get to any goal (check `reward()`)
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
from numpy import ndarray as arr
from scipy import sparse
import math
import os,sys
sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import Agent, Landmark, Entity,  SatWorld
from multiagent.scenario import BaseScenario

entity_mapping = {'agent': 0, 'landmark': 1, 'obstacle':2}

class SatelliteScenario(BaseScenario):
    def make_world(self, args:argparse.Namespace) -> SatWorld:
        """
            Parameters in args
            ––––––––––––––––––
            • num_agents: int
                Number of agents in the environment
                NOTE: this is equal to the number of goal positions
            • num_obstacles: int
                Number of num_obstacles obstacles
            • collaborative: bool
                If True then reward for all agents is sum(reward_i)
                If False then reward for each agent is what it gets individually
            • max_speed: Optional[float]
                Maximum speed for agents
                NOTE: Even if this is None, the max speed achieved in discrete 
                action space is 2, so might as well put it as 2 in experiments
                TODO: make list for this and add this in the state
            • collision_rew: float
                The reward to be negated for collisions with other agents and 
                obstacles
            • goal_rew: float
                The reward to be added if agent reaches the goal
            • min_dist_thresh: float
                The minimum distance threshold to classify whether agent has 
                reached the goal or not
            • use_dones: bool
                Whether we want to use the 'done=True' when agent has reached 
                the goal or just return False like the `simple.py` or 
                `simple_spread.py`
            • episode_length: int
                Episode length after which environment is technically reset()
                This determines when `done=True` for done_callback
            • graph_feat_type: str
                The method in which the node/edge features are encoded
                Choices: ['global', 'relative']
                    If 'global': 
                        • node features are global [pos, vel, goal, entity-type]
                        • edge features are relative distances (just magnitude)
                        • 
                    If 'relative':
                        • TODO decide how to encode stuff

            • max_edge_dist: float
                Maximum distance to consider to connect the nodes in the graph
        """
        # pull params from args
        self.world_size = args.world_size
        self.num_agents = args.num_agents
        self.num_scripted_agents = args.num_scripted_agents
        self.num_obstacles = args.num_obstacles
        self.collaborative = args.collaborative
        self.max_speed = args.max_speed
        self.collision_rew = args.collision_rew
        self.goal_rew = args.goal_rew
        self.min_dist_thresh = args.min_dist_thresh
        self.use_dones = args.use_dones
        self.episode_length = args.episode_length
        self.goal_type = args.goal_type
        self.goal_sharing = args.goal_sharing
        if not hasattr(args, 'max_edge_dist'):
            self.max_edge_dist = 1
            print('_'*60)
            print(f"Max Edge Distance for graphs not specified. "
                    f"Setting it to {self.max_edge_dist}")
            print('_'*60)
        else:
            self.max_edge_dist = args.max_edge_dist
        

        world = SatWorld()
        # graph related attributes
        world.cache_dists = True # cache distance between all entities
        world.graph_mode = True
        world.graph_feat_type = args.graph_feat_type
        world.world_length = args.episode_length
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # set any world properties
        world.dim_c = 2
        num_landmarks = self.num_agents # no. of goals equal to no. of agents
        num_scripted_agents_goals = self.num_scripted_agents
        world.collaborative = args.collaborative

        # add agents
        global_id = 0
        world.agents = [Agent() for i in range(self.num_agents)]
        world.scripted_agents = [Agent() for _ in range(self.num_scripted_agents)]
        for i, agent in enumerate(world.agents + world.scripted_agents):
            agent.id = i
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.global_id = global_id
            global_id += 1
            # NOTE not changing size of agent because of some edge cases; 
            # TODO have to change this later
            # agent.size = 0.15
            agent.max_speed = self.max_speed
        # add landmarks (goals)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        world.scripted_agents_goals = [Landmark() for i in range(num_scripted_agents_goals)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False
            landmark.global_id = global_id
            global_id += 1
        # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f'obstacle {i}'
            obstacle.collide = True
            obstacle.movable = False
            obstacle.global_id = global_id
            global_id += 1
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:SatWorld) -> None:
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)

        #################### set colours ####################
        # set colours for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # set colours for scripted agents
        for i, agent in enumerate(world.scripted_agents):
            agent.color = np.array([0.15, 0.15, 0.15])
        # set colours for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        # set colours for scripted agents goals
        for i, landmark in enumerate(world.scripted_agents_goals):
            landmark.color = np.array([0.15, 0.95, 0.15])
        # set colours for obstacles
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.25, 0.25, 0.25])
        #####################################################
        self.random_scenario(world)

    def random_scenario(self, world):
        """
            Randomly place agents and landmarks
        """
        ####### set random positions for entities ###########
        # set random static obstacles first
        for obstacle in world.obstacles:

            random_a = np.random.uniform(6776, 6777,1)
            random_e = np.random.uniform(.0001, 0.01,1)
            random_i = np.random.uniform(98.7*(math.pi/180), 99.0* (math.pi/180),1)
            random_omega = np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_w = np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_M = np.random.uniform(15*(math.pi/180), 15*(math.pi/180),1)

            obstacle.state.a = random_a
            obstacle.state.e = random_e
            obstacle.state.i = random_i
            obstacle.state.omega = random_omega
            obstacle.state.w = random_w
            obstacle.state.M= random_M


        #####################################################

        # set agents at random positions not colliding with obstacles
        num_agents_added = 0
        agents_added = []
        while True:
            
            if num_agents_added == self.num_agents:
                break

            random_a = np.random.uniform(6676, 6677,1)
            random_e = np.random.uniform(.0001, 0.001,1)
            random_i = np.random.uniform(98.7*(math.pi/180), 99.0* (math.pi/180),1)
            random_omega = np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_w = np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_M = np.random.uniform(15*(math.pi/180), 15*(math.pi/180),1)

            # agent_size = world.agents[num_agents_added].size
            # obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            # agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
            # if not obs_collision and not agent_collision:
            world.agents[num_agents_added].state.a = random_a
            world.agents[num_agents_added].state.e = random_e
            world.agents[num_agents_added].state.i = random_i
            world.agents[num_agents_added].state.omega = random_omega
            world.agents[num_agents_added].state.w = random_w
            world.agents[num_agents_added].state.M= random_M

                # world.agents[num_agents_added].state.p_vel = (1/1000)*np.ones(world.dim_p)
            world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
            agents_added.append(world.agents[num_agents_added])
            num_agents_added += 1

        #####################################################
        
        # set scripted agents goals at random positions not colliding with obstacles
        num_scripted_agents_added = 0
        while True:
            if num_scripted_agents_added == self.num_scripted_agents:
                break

            random_a = 6677.1 #np.random.uniform(6676, 6677,1)
            random_e = .001 #np.random.uniform(.0001, 0.001,1)
            random_i = 99.1* math.pi/180 # np.random.uniform(98.7*(math.pi/180), 99.0* (math.pi/180),1)
            random_omega =60*math.pi/180 #np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_w = 60*math.pi/180 #np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_M = 15*math.pi/180 #np.random.uniform(15*(math.pi/180), 15*(math.pi/180),1)

            # agent_size = world.agents[num_agents_added].size
            # obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            # agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
            # if not obs_collision and not agent_collision:
            world.scripted_agents[num_scripted_agents_added].state.a = random_a
            world.scripted_agents[num_scripted_agents_added].state.e = random_e
            world.scripted_agents[num_scripted_agents_added].state.i = random_pos
            world.scripted_agents[num_scripted_agents_added].state.omega = random_omega
            world.scripted_agents[num_scripted_agents_added].state.w = random_w
            world.scripted_agents[num_scripted_agents_added].state.M= random_M

            agents_added.append(world.scripted_agents[num_scripted_agents_added])
            num_scripted_agents_added += 1


        #####################################################

        # set landmarks (goals) at random positions not colliding with obstacles 
        # and also check collisions with already placed goals
        num_goals_added = 0
        goals_added = []
        while True:
            if num_goals_added == self.num_agents:
                break
            random_a = 6677.1 #np.random.uniform(6676, 6677,1)
            random_e = .001 #np.random.uniform(.0001, 0.001,1)
            random_i = 99.1* math.pi/180 # np.random.uniform(98.7*(math.pi/180), 99.0* (math.pi/180),1)
            random_omega =60*math.pi/180 #np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_w = 60*math.pi/180 #np.random.uniform(60*(math.pi/180), 60*(math.pi/180),1)
            random_M = 15*math.pi/180 #np.random.uniform(15*(math.pi/180), 15*(math.pi/180),1)

            # agent_size = world.agents[num_agents_added].size
            # obs_collision = self.is_obstacle_collision(random_pos, agent_size, world)
            # agent_collision = self.check_agent_collision(random_pos, agent_size, agents_added)
            # if not obs_collision and not agent_collision:
            world.landmarks[num_goals_added].state.a = random_a
            world.landmarks[num_goals_added].state.e = random_e
            world.landmarks[num_goals_added].state.i = random_i
            world.landmarks[num_goals_added].state.omega = random_omega
            world.landmarks[num_goals_added].state.w = random_w
            world.landmarks[num_goals_added].state.M= random_M

            # agents_added.append(world.scripted_agents[num_goals_added])
            num_goals_added += 1

          
        #####################################################

        ############ find minimum times to goals ############
        if self.max_speed is not None:
            for agent in world.agents:
                self.min_time(agent, world)
        #####################################################
        ############ update the cached distances ############
        world.calculate_distances()
        self.update_graph(world)
        ####################################################

    def info_callback(self, agent:Agent, world:SatWorld) -> Tuple:
        # TODO modify this 
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        goal = world.get_entity('landmark', agent.id)
        dist = np.sqrt(np.sum(np.square(agent.state.a - 
                                        goal.state.a)))
        world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
            world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            if self.is_obstacle_collision(agent.state.a, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            'Dist_to_goal': world.dist_left_to_goal[agent.id],
            'Time_req_to_goal': world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            'Num_agent_collisions': world.num_agent_collisions[agent.id], 
            'Num_obst_collisions': world.num_obstacle_collisions[agent.id],
        }
        if self.max_speed is not None:
            agent_info['Min_time_to_goal'] = agent.goal_min_time
        return agent_info

    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size:float, world:SatWorld) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.a - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision
    
    # check collision of agent with other agents
    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.a - pos
                dist = np.linalg.norm(delta_pos)
                if dist < (agent.size + agent_size):
                    collision = True
                    break
        return collision

    # check collision of agent with another agent
    def is_collision(self, agent1:Agent, agent2:Agent) -> bool:
        delta_pos = agent1.state.a - agent2.state.a
        dist = np.linalg.norm(delta_pos)

        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_landmark_collision(self, pos, size:float, landmark_list:List) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.a - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + landmark.size
            if dist < dist_min:
                collision = True
                break
        return collision

    # get min time required to reach to goal without obstacles
    def min_time(self, agent:Agent, world:SatWorld) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed"
        agent_id = agent.id
        # get the goal associated to this agent
        # landmark = world.get_entity(entity_type='landmark', id=agent_id)
        # dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
        #                                 landmark.state.p_pos)))
        # min_time = dist / agent.max_speed

        min_time = 6*3600
        agent.goal_min_time = min_time
        return min_time

    # done condition for each agent
    def done(self, agent:Agent, world:SatWorld) -> bool:
        # if we are using dones then return appropriate done
        #### SYDNEY YOU HARDCODED IT IN!!!!!
        # if self.use_dones:
        #     landmark = world.get_entity('landmark', agent.id)
        #     dist = np.sqrt(np.sum(np.square(agent.state.p_pos - 
        #                                     landmark.state.p_pos)))
        #     if dist < self.min_dist_thresh:
        #         return True
        #     else:
        #         return False
        # it not using dones then return done 
        # only when episode_length is reached
        # else:
        if world.current_time_step >= world.world_length:
            return True
        else:
            return False

    def reward(self, agent:Agent, world:SatWorld) -> float:
        # Agents are rewarded based on distance to 
        # its landmark, penalized for collisions
        rew = 0
        agents_goal = world.get_entity(entity_type='landmark', id=agent.id)
        # dist_to_goal = np.sqrt(np.sum(np.square(agent.state.a - 
                                                # agents_goal.state.p_pos)))

        ### WARNING -------- 
        #### Hardcoding the goal to be that you are close with the semmi major axis 
        if isinstance(agent.state.a,(list,np.ndarray)):
            agent.state.a = agent.state.a[0]
        if isinstance(agents_goal.state.a,(list,np.ndarray)):
            agents_goal.state.a = agents_goal.state.a[0]


        dist_to_goal = abs(agent.state.a - agents_goal.state.a) #+ abs(agent.state.a - agents_goal.state.a)
        # print('dist to goal is ' + str(dist_to_goal) +' min dist threshold ' +str(self.min_dist_thresh))
        if dist_to_goal < self.min_dist_thresh:
            rew += self.goal_rew
        else:
            rew -= dist_to_goal
        if agent.collide:
            for a in world.agents:
                # do not consider collision with itself
                if a.id == agent.id:
                    continue
                if self.is_collision(a, agent):
                    rew -= self.collision_rew
            
            if self.is_obstacle_collision(pos=agent.state.a,
                                        entity_size=agent.size, world=world):
                rew -= self.collision_rew
        return rew

    def observation(self, agent:Agent, world:SatWorld, local_obs:bool=None) -> arr:

        ##### THIS HAS TO BE RE-WRITTEN LATER 
        """
            Return:
                [agent_vel, agent_pos, goal_pos]
            NOTE: `local_obs` is a useless argument here. Just adding so that it 
            fits the observation_callback signature for navigation.py
        """
        # get positions of all entities in this agent's reference frame
        goal_pos = []
        agents_goal = world.get_entity('landmark', agent.id)
        if isinstance(agent.state.a,(list,np.ndarray)):
            agent.state.a = agent.state.a[0]
            if isinstance(agent.state.a,(list,np.ndarray)):
                agent.state.a = agent.state.a[0]
        goal_pos.append(agents_goal.state.a - agent.state.a)
        xx= np.asarray(  [agent.state.a, agent.state.e[0],agent.state.i[0], agent.state.omega[0], agent.state.w[0], agent.state.M[0],  goal_pos[0] ]   )  

        return xx 
        #np.concatenate([np.asarray(agent.state.a), agent.state.e,agent.state.i, agent.state.omega, agent.state.w, agent.state.M] + goal_pos)

    def get_id(self, agent:Agent) -> arr:
        return np.array([agent.global_id])
    
    def graph_observation(self, agent:Agent, world:SatWorld) -> Tuple[arr, arr]:
        """
            FIXME: Take care of the case where edge_list is empty
            Returns: [node features, adjacency matrix]
            • Node features (num_entities, num_node_feats):
                If `global`: 
                    • node features are global [pos, vel, goal, entity-type]
                    • edge features are relative distances (just magnitude)
                    NOTE: for `landmarks` and `obstacles` the `goal` is 
                            the same as its position
                If `relative`:
                    • TODO decide how to encode stuff
            • Adjacency Matrix (num_entities, num_entities)
                NOTE: using the distance matrix, need to do some post-processing
                If `global`:
                    • All close-by entities are connectd together
                If `relative`:
                    • Only entities close to the ego-agent are connected
                        NOTE: This considers only first hop neighbours
            
        """
        num_entities = len(world.entities)
        # node observations
        node_obs = []
        if world.graph_feat_type == 'global':
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)
        elif world.graph_feat_type == 'relative':
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_relative(agent, entity, world)
                node_obs.append(node_obs_i)

        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag

        return node_obs, adj

    def update_graph(self, world:SatWorld):
        """
            Construct a graph from the cached distances.
            Nodes are entities in the environment
            Edges are constructed by thresholding distances
        """
        dists = world.cached_dist_mag
        # just connect the ones which are within connection 
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * \
                            (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == 'global':
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == 'relative':
            world.edge_weight = dists[row, col]

    def remove_array_doubles(self, a) -> float:
        if isinstance(a,(list,np.ndarray)):
                a = a[0]
                if isinstance(a,(list,np.ndarray)):
                    a = a[0]
        return a

    
    def _get_entity_feat_global(self, entity:Entity, world:SatWorld) -> arr:
        """
            Returns: ([velocity, position, goal_pos, entity_type])
            in global coords for the given entity
        """

        a = self.remove_array_doubles(entity.state.a)
        e = self.remove_array_doubles(entity.state.e)
        i = self.remove_array_doubles(entity.state.i)
        omega  = self.remove_array_doubles(entity.state.omega)
        w = self.remove_array_doubles(entity.state.w)
        M =  self.remove_array_doubles(entity.state.M)
        # vel = entity.state.p_vel
        if 'agent' in entity.name:
            goal_pos = self.remove_array_doubles(world.get_entity('landmark', entity.id).state.a)
            entity_type = entity_mapping['agent']
        elif 'landmark' in entity.name:
            goal_pos = a
            entity_type = entity_mapping['landmark']
        elif 'obstacle' in entity.name:
            goal_pos = a
            entity_type = entity_mapping['obstacle']
        else:
            raise ValueError(f'{entity.name} not supported')

        if self.goal_sharing == False:
            return np.hstack([a,e,i,omega,w,M,  goal_pos,  entity_type])

        
        
        cc = np.hstack([a,e,i,omega,w,M, goal_pos, entity_type])
        if len(cc)!=8:
            print('enitiy feat global is returnign length:  '+str(len(cc)))
            print('bro what the lhell is going on a is '+ str(a))
            print('state w '+str(entity.state.w))
            print('entity state goal pos '+str(goal_pos))
        return cc

    def _get_entity_feat_relative(self, agent:Agent, entity:Entity, world:SatWorld) -> arr:
        """
            Returns: ([velocity, position, goal_pos, entity_type])
            in coords relative to the `agent` for the given entity
        """
        # print('ios it enetering ENTITY FEAT RELATIVE')
        entity_a = entity.state.a
        entity_e = entity.state.a
        entity_i = entity.state.a
        entity_omega  = entity.state.omega
        entity_w = entity.state.w
        entity_M =  entity.state.M

        agent_a = agent.state.a
        agent_e = agent.state.a
        agent_i = agent.state.a
        agent_omega  = agent.state.omega
        agent_w = agent.state.w
        agent_M =  agent.state.M

        # agent_pos = agent.state.p_pos
        # agent_vel = agent.state.p_vel
        # entity_pos = entity.state.p_pos
        # entity_vel = entity.state.p_vel
        rel_a = entity_a - agent_a
        rel_e = entity_e - agent_e
        rel_i = entity_i - agent_i
        rel_omega = entity_omega - agent_omega
        rel_w = entity_w - agent_w
        rel_M = entity_M - agent_M
        # rel_vel = entity_vel - agent_vel
        if 'agent' in entity.name:
            goal_pos = world.get_entity('landmark', entity.id).state.a
            rel_goal_pos = goal_pos - agent_a
            entity_type = entity_mapping['agent']
        elif 'landmark' in entity.name:
            rel_goal_pos = rel_a
            entity_type = entity_mapping['landmark']
        elif 'obstacle' in entity.name:
            rel_goal_pos = rel_a
            entity_type = entity_mapping['obstacle']
        else:
            raise ValueError(f'{entity.name} not supported')

            # print('rel_a  '+str(rel_a) + 'rel e is ' +str(rel_e))
        if args.goal_sharing == False:
            return np.hstack([rel_a,rel_e,rel_i,rel_omega,rel_w,rel_M,  entity_type])

        return np.hstack([rel_a,rel_e,rel_i,rel_omega,rel_w,rel_M, rel_goal_pos, entity_type])
        # np.hstack([rel_vel, rel_pos, rel_goal_pos, entity_type])

# actions: [None, ←, →, ↓, ↑, comm1, comm2]
if __name__ == "__main__":

    from multiagent.environment import SatelliteMultiAgentGraphEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents:int=3
            self.world_size=3
            self.num_scripted_agents=0
            self.num_obstacles:int=1
            self.collaborative:bool=False 
            self.max_speed:Optional[float]=2
            self.collision_rew:float=5
            self.goal_rew:float=5
            self.min_dist_thresh:float=0.1
            self.use_dones:bool=False
            self.episode_length:int=25
            self.max_edge_dist:float=1
            self.graph_feat_type:str='global'
            self.satellite:str='satellite'
    args = Args()

    
    
    # create multiagent environment 
    scenario = SatelliteScenario()
    # create world
    world = scenario.make_world(args)    
    env = SatelliteMultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
                        reward_callback=scenario.reward, 
                        observation_callback=scenario.observation, 
                        graph_observation_callback=scenario.graph_observation,
                        info_callback=scenario.info_callback, 
                        done_callback=scenario.done,
                        id_callback=scenario.get_id,
                        update_graph=scenario.update_graph,
                        shared_viewer=False)   

    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n, agent_id_n, node_obs_n, adj_n = env.reset()
    stp=0
    while True:
        # query for action from each agent's policy
        act_n = []
        dist_mag = env.world.cached_dist_mag

        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)


        # render all agent views
        env.render()
        stp+=1
        # display rewards
