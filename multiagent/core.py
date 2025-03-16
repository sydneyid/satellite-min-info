import numpy as np
import math


class EntityState(object):  # physical/external base state of all entites
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class SatState(EntityState):
    def __init__(self):
        super(SatState, self).__init__()
        self.c = None # communication utterance

class Action(object):
    def __init__(self):
        self.u = None  # physical action



class Entity(object): # properties and state of physical world entity
    def __init__(self):
        # id
        self.id = None
        self.global_id = None
        self.name = ''
        
        
        self.size = 0.050
        self.movable = False # entity can move / be pushed
        self.collide = True  # entity collides with others
        self.color = None # color
        self.max_speed = None# max speed and accel
        self.accel = None

        self.state = EntityState()  # state
        self.initial_mass = 1.0 #kilograms


    @property
    def mass(self):
        return self.initial_mass

class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()


class Satellite(Entity):# properties of satellites
    def __init__(self):
        super(Satellite, self).__init__()
        self.movable = True  # movable by default
        self.u_range = 1000000.0  # control range
        self.state = SatState()  # state
        self.action = Action()  # action
        self.action_callback = None  # script behavior to execute
        self.goal_min_time = np.inf # min time required to get to its allocated goal
        self.t = 0.0 # time passed for each agent
        self.actions_hist = [] #action history
        self.positions_hist = [] #position history


    

# multi-agent world
class SatWorld(object):
    def __init__(self):
        # if we want to construct graphs with the entities 
        self.graph_mode = False
        self.edge_list = None
        self.graph_feat_type = None
        self.edge_weight = None
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.agents_goals = []
        self.obstacles = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1

        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    
    @property
    def entities(self): #return everything in the envirnment
        return self.agents + self.landmarks + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return self.agents
        # return [agent for agent in self.agents if agent.action_callback is None]


    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                            len(self.entities),
                                            self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    # get the entity given the id and type
    def get_entity(self, entity_type: str, id:int) -> Entity:
        # TODO make this more elegant instead of iterating through everything
        if entity_type == 'agent':
            for agent in self.agents:
                if agent.name == f'agent {id}':
                    return agent
            raise ValueError(f"Agent with id: {id} doesn't exist in the world")
        if entity_type == 'landmark':
            for landmark in self.landmarks:
                if landmark.name == f'landmark {id}':
                    return landmark
            raise ValueError(f"Landmark with id: {id} doesn't exist in the world")
        if entity_type == 'obstacle':
            for obstacle in self.obstacles:
                if obstacle.name == f'obstacle {id}':
                    return obstacle
            raise ValueError(f"Obstacle with id: {id} doesn't exist in the world")



    def update_agent_state(self, agent:Satellite):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * \
                    agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    # NOTE: this is better than using get_collision_force() since 
    # it takes into account if the entity is movable or not
    def get_entity_collision_force(self, ia, ib):
        entity_a = self.entities[ia]
        entity_b = self.entities[ib]
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]  # neither entity moves
        if (entity_a is entity_b):
            return [None, None]  # don't collide against itself
        if (self.cache_dists) and (self.cached_dist_vect is not None):
            delta_pos = self.cached_dist_vect[ia, ib]
            dist = self.cached_dist_mag[ia, ib]
            dist_min = self.min_dists[ia, ib]
        else:
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if entity_a.movable and entity_b.movable:
            # consider mass in collisions
            force_ratio = entity_b.mass / entity_a.mass
            force_a = force_ratio * force
            force_b = -(1 / force_ratio) * force
        else:
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    

    
    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors  + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color
            
    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                # [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            
        return p_force

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            
    def step(self):
        # set actions for scripted agents 
        for agent in self.agents:
            agent.t += self.dt
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_sat_action_force(p_force)
        
        # apply environment forces
        p_force = self.apply_environment_force(p_force)

        # integrate physical state
        self.integrate_sat_state(p_force)
        # update agent state
        for agent in self.agents:
            agent.t += self.dt
            self.update_agent_state(agent)
        if self.cache_dists:
            self.calculate_distances()
            
        
            
            
    def perturbed_dynamics(self,s,u):
        mu = (398600)*1000
        a = (6378+300 )* 1000
        n = (mu/(a**3))**.5
        re= 6378.1363*1000 #m 
        J2= 1.08262668e-3
        ss = ((3*J2*(re**2))/(8*a**2)) * (1+3*math.cos(2*self.i))
        c =  (1+ss)**.5

        A = np.zeros((4,4))
        A[0][2] = 1
        A[1][3] = 1
        A[2][0] =  -(5*c**2 -2)*n**2  
        A[2][3] =   -2*n*c  
        A[3][2] =  2*n*c   

        B = np.zeros((4,2))
        B[2][0] = 1
        B[3][1] = 1
        
        s = np.asarray(s)
        u = np.asarray(u)

        dsdt = s.dot(np.array(A).T) + u.dot(np.array(B).T)
        return dsdt 
        
        
    def dynamics(self, s,u):
        ## linearized J2 equations taken from: https://www.ijser.org/researchpaper/Satellite-Tracking-Control-under-J2-Perturbations.pdf
        mu = (398600)*1000
        a = (6378+300 )* 1000
        n = (mu/(a**3))**.5

        A = np.zeros((4,4))
        A[0][2] = 1
        A[1][3] = 1
        A[2][0] = 3*n**2
        A[2][3] =  2*n 
        A[3][2] =  -2*n 

        B = np.zeros((4,2))
        B[2][0] = 1
        B[3][1] = 1
        
        s = np.asarray(s)
        u = np.asarray(u)

        dsdt = s.dot(np.array(A).T) + u.dot(np.array(B).T)
        return dsdt 
    
    def call_dynamics(self,x,u):
        if self.perturbed:
            return self.perturbed_dynamics(x,u)
        
        return self.dynamics(x,u) 

    def integrate_sat_state(self, p_force):
        mu= 398600
        count =0
        for i,entity in enumerate(self.entities):
            x = [entity.state.p_pos[0],entity.state.p_pos[1],entity.state.p_vel[0],entity.state.p_vel[1]]
            if p_force[i]is None:
                u = [0,0]
            else:
                u = p_force[i]
            dsdt = self.call_dynamics(x,u) 
            x =x + dsdt*self.dt	

            entity.state.p_pos[0] = x[0]
            entity.state.p_pos[1] = x[1]
            entity.state.p_vel[0] = x[2]
            entity.state.p_vel[1] = x[3]
            if p_force[i] is not None:
                if ((1000*entity.state.p_vel[0])**2+ (1000*entity.state.p_vel[1])**2)**.5>100:
                    print('\t\t** WARNING*** velocity is large, the system will be uncontrollable')
                    
                    
    def apply_sat_action_force(self, p_force):
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None 
                            else agent.mass) * agent.action.u + noise  

        
        return p_force