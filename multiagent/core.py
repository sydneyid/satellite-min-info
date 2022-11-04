import numpy as np
import math

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties of wall entities
class Wall(object):
    def __init__(self, orient='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1,
                hard=True):
        # orientation: 'H'orizontal or 'V'ertical
        self.orient = orient
        # position along axis which wall lays on (y-axis for H, x-axis for V)
        self.axis_pos = axis_pos
        # endpoints of wall (x-coords for H, y-coords for V)
        self.endpoints = np.array(endpoints)
        # width of wall
        self.width = width
        # whether wall is impassable to all agents
        self.hard = hard
        # color of wall
        self.color = np.array([0.0, 0.0, 0.0])


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # id
        self.id = None
        self.global_id = None
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agent are adversary
        self.adversary = False
        # agent are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1000000.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # min time required to get to its allocated goal
        self.goal_min_time = np.inf
        # time passed for each agent
        self.t = 0.0


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
        self.scripted_agents = []
        self.scripted_agents_goals = []
        self.obstacles, self.walls = [], []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

    # return all entities in the worlhttps://owa.exchange.mit.edu/owa/auth/logon.aspx?replaceCurrent=1&reason=3&url=https%3a%2f%2fowa.exchange.mit.edu%2fowa%2fauth%2ferrorfe.aspx%23httpCode%3d500d
    @property
    def entities(self):
        return self.agents + self.landmarks + self.obstacles

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return self.agents
        # return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def get_scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

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


    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_entity_collision_force(a, b)
                # [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]    
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None: p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        #print('p force as dictated in the apply_environ_force function ' + str(p_force)) 
        return p_force

    def update_agent_state(self, agent:Agent):
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

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None  # entity is beyond endpoints of wall
        elif (ent_pos[prll_dim] < wall.endpoints[0] or
                ent_pos[prll_dim] > wall.endpoints[1]):
            # part of entity is beyond wall
            if ent_pos[prll_dim] < wall.endpoints[0]:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[0]
            else:
                dist_past_end = ent_pos[prll_dim] - wall.endpoints[1]
            theta = np.arcsin(dist_past_end / entity.size)
            dist_min = np.cos(theta) * entity.size + 0.5 * wall.width
        else:  # entire entity lies within bounds of wall
            theta = 0
            dist_past_end = 0
            dist_min = entity.size + 0.5 * wall.width

        # only need to calculate distance in relevant dim
        delta_pos = ent_pos[perp_dim] - wall.axis_pos
        dist = np.abs(delta_pos)
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force_mag = self.contact_force * delta_pos / dist * penetration
        force = np.zeros(2)
        force[perp_dim] = np.cos(theta) * force_mag
        force[prll_dim] = np.sin(theta) * np.abs(force_mag)
        return force

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
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
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])
            
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.t += self.dt
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_sat_action_force(p_force)
        # apply environment forces
        #p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_sat_state(p_force)
        # update agent state
        for agent in self.agents:
            agent.t += self.dt
            self.update_agent_state(agent)
        if self.cache_dists:
            self.calculate_distances()
    def dynamics(self, s,u):
        mu = 398600
        a = 6378+300
        n = (mu/(a**3))**.5
        A = np.zeros((4,4))
        A[0][2] = 1
        A[1][3] = 1
        A[2][0] = 3*n**2
        A[2][3] = 2*n
        A[3][2] = -2 * n

        B = np.zeros((4,2))
        B[2][0] = 1
        B[3][1] = 1
        s = np.asarray(s)
        u = np.asarray(u)
        dsdt = s.dot(np.array(A).T) + u.dot(np.array(B).T)
        return dsdt 

    def integrate_sat_state(self, p_force):
        mu= 398600
        count =0
        for i,entity in enumerate(self.entities):
            x = [entity.state.p_pos[0],entity.state.p_pos[1],entity.state.p_vel[0],entity.state.p_vel[1]]
            if p_force[i]is None:
                u = [0,0]
            else:
                u = p_force[i]
            dsdt = self.dynamics(x,u) 
            x =x + dsdt*self.dt	
	
            entity.state.p_pos[0] = x[0]
            entity.state.p_pos[1] = x[1]

            entity.state.p_vel[0] = x[2]
            entity.state.p_vel[1] = x[3]
            # if p_force[i] is not None:
                # print('\np force from control is '+str(u) )#           #print('pvelbefore is '+ str(entity.state.p_vel))
                # print('position after '+ str(entity.state.p_pos) + ' km')
                # print('pvel after '+ str(1000*entity.state.p_vel) +' m/s')
                # if ((1000*entity.state.p_vel[0])**2+ (1000*entity.state.p_vel[1])**2)**.5>100:
                #     print('\t\t** WARNING*** velocity is large, the system will be uncontrollable')
    def apply_sat_action_force(self, p_force):
        # set applied forces
### this should be the guy that generates a legitimate response 
        for i,agent in enumerate(self.agents):
            if agent.movable:
                #print('agent action in sat action force is '+ str(agent.action.u))
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = (agent.mass * agent.accel if agent.accel is not None 
                            else agent.mass) * agent.action.u + noise  
               # print('handdone calculation ' + str(agent.mass*agent.action.u) + 'vs the other guy '+str(p_force[i]))  
               # print('p force as dictated in the apply_sat_action_force function ' + str(p_force[i]))            
        return p_force
