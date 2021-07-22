"""MIT License
Copyright (c) [2020] [Hugo Dolan]
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import math
import numpy as np
import pandas as pd
import random
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from collections import namedtuple
import pickle
import seaborn as sns


adj_matrix = pd.read_csv('./outputs/airport_routes_matrix.csv',index_col=0)
alphas = pd.read_csv('./outputs/airport_alphas.csv', index_col=0)
centralities = pd.read_csv('./outputs/airport_centralities.csv', index_col=0)
populations = pd.read_csv('./outputs/airport_populations.csv', index_col=0)
communities = pd.read_csv('./outputs/bsm_communities.csv', index_col=0).set_index('IATA').reindex(adj_matrix.columns)
airports = pd.read_csv('./outputs/airports_w_populations.csv',index_col=0)
airport_country = pd.read_csv('./outputs/airport_country_mapping.csv', index_col=0).set_index('IATA').reindex(adj_matrix.columns)
betweeness = pd.read_csv('./outputs/airport_betweeness.csv', index_col=0).reindex(adj_matrix.columns)

LHR = 147 # London Heathrow
ATL = 2880 # Atlanta
DXB = 3199 # Dubai
WUH = 724 # Wuhan Airport
JFK = 3097 # JFK New York
HKG = 1197 # Hong Kong

vals = lambda X: X.values

def get_community(airport_idx):
    community_id = communities['community'][airport_idx]
    return communities.reset_index()[communities.reset_index()['community'] == community_id].index.values

def close_airport_pct(threshold_above, metric=populations):
    threshold = metric.quantile(q=threshold_above)[0]
    return (metric > threshold).values.reshape(-1)

ActionSpace = namedtuple('ActionSpace', ['n','sample'])
ObservationSpace = namedtuple('ObservationSpace', ['n'])

countries = airport_country['country_IATA'].unique()
dimension = len(airport_country['country_IATA'].unique())

class EpidemicEnvironment:
    def __init__(self, adj_matrix, 
                 population_vector, 
                 agent_idx, 
                 community, 
                 infected_idx = 0,
                 alpha_plus = 0.3, 
                 alphas_plus = None, 
                 c_plus = 0.1,
                 c_minus = 1, 
                 beta = 57/160 + 1/7, 
                 beta_reduced = 25/160 + 1/7, 
                 gamma = 1/16, 
                 delta = 1/(2*365), 
                 epsilon = 1/7, 
                 centrality = None,
                 p = 1, 
                 lmbda = 1.2, 
                 mu = 0.2, 
                 lockdown_threshold = -1, 
                 decay = 0):
        """
        SEIRS Network Epidemic Model Environment
        :param adj_matrix: Adjacency Matrix (A)ij indicates edge from i to j (NxN)
        :param population_vector: Nx1 Vector of populations for each node
        :param agent_idx: Airport index associated with the agent
        :param community: A list of airport indexes of airports in the same community
        :param infected_idx: airport where the epidemic starts
        :param alpha_plus: Percentage of base population which can fly 
        :param alphas_plus: Percentage of base population which can fly (Specifed as an Nx1 vector)
        :param c_plus: Percentage of flying population who can embark on any day
        :param c_minus: Percentage of flying population who can return on any day
        :param beta: Rate of infection
        :param beta_reduced: Rate of infection for when agent is locked down
        :param gamma: Rate of recovery
        :param delta: Rate of immunity loss
        :param epsilon: Rate at which people move from the exposed to infected stage (syptomatic)
        :param p: base economic penalty for lockdowns or infections
        :param lmbda: rate at which penalty for infections in unmitigated state should be applied > 1
        :param mu: rate at which penalty for infections in lockdown state should be applied < 1
        :param lockdown_threshold: within (0,1) defining pct of population in any node which can get infected before a lockdown
        :param decay: non-negative number defining the decay rate of the beta parameter over time, value 0 means no decay
        """
        
        # Initialisation
        self.A = adj_matrix # Adjacency Matrix
        self.A_ones = np.ones_like(self.A) # Array of ones (for efficiency its computed only once)
        self.population_vector = population_vector
        self.N = self.A.shape[0] # Number of airports
        self.ID = np.eye(self.N)
        self.agent_idx = agent_idx
        self.c_plus = c_plus
        self.c_minus = c_minus
        self.decay = decay
        self.beta = beta
        self.beta_reduced = beta_reduced
        self.epsilon = epsilon
        self.gamma = gamma
        self.delta = delta
        self.decay = decay
        
        if type(alphas_plus) == type(None):
            self.alphas_plus = np.diag(np.repeat(alpha_plus,self.N)) # Airport Vector proportion of populations
        else:
            self.alphas_plus = np.diag(alphas_plus.reshape(-1,)) 

        self.D = self.A.sum(axis=1) # Degrees of airports
        self.D_inv = np.array([1 / deg if deg > 0 else 0 for deg in self.D])
        self.infected_idx = infected_idx
        
        # Differential Equation Matrix
        self.B = np.array([[-1*beta, 0, 0, delta],[beta, -1*epsilon, 0,0],[0,epsilon, -1*gamma, 0],[0, 0, gamma, -1*delta]])
        self.B_reduced = np.array([[-1*beta_reduced, 0, 0, delta],[beta_reduced, -1*epsilon, 0,0],[0,epsilon, -1*gamma, 0],[0, 0, gamma, -1*delta]])
        self.B_zero = np.array([[-1*0, 0, 0, delta],[0, -1*epsilon, 0,0],[0,epsilon, -1*gamma, 0],[0, 0, gamma, -1*delta]])
        
        # Centrality Matrix (N x N)
        # We will refer to it as the 'Diffusion Matrix' 
        # since we use a captial C (like the lower case c_plus/minus for diffusion coeff)
        # Used to weight the distribution of flows on the network
        # So that important airports get more traffic
        
        self.use_centrality = type(centrality) != type(None)
        
        if self.use_centrality:
            C = self.A * centrality.T
            C_norms = np.array([1 / norm if norm > 0 else 0 for norm in C.sum(axis=1)]).reshape((-1,1))
            self.C = C * C_norms # Normalised Centrality 
            
        # Initialise temporal state
        self.reset()
        
        # State And Action Spaces
        # Vector observation space V[0] = Local State, V[1] = Community State, V[2] = Global State
        # With state categores (-1,0,1) = (Decreased Infections, Static, Increased Infections)
        self.observation_space = ObservationSpace(27)
        
        # Either 0 = Open , 1 = Lockdown
        self.action_space = ActionSpace(2, lambda: random.choice([0,1]))
        
        # Reward parameters
        self.lmbda = lmbda
        self.mu = mu
        self.p = p
        
        # State Parameters
        self.community = community
        
        # For simple automatic rules (rather than RL)
        self.lockdown_threshold = lockdown_threshold
        
    def reset(self):
        # Reset the matrix B because it might have changed due to exponential decay
        self.B = np.array([[-1*self.beta, 0, 0, self.delta],[self.beta, -1*self.epsilon, 0,0],[0,self.epsilon, -1*self.gamma, 0],[0, 0, self.gamma, -1*self.delta]])
        self.B_reduced = np.array([[-1*self.beta_reduced, 0, 0, self.delta],[self.beta_reduced, -1*self.epsilon, 0,0],[0,self.epsilon, -1*self.gamma, 0],[0, 0, self.gamma, -1*self.delta]])
        self.B_zero = np.array([[-1*0, 0, 0, self.delta],[0, -1*self.epsilon, 0,0],[0,self.epsilon, -1*self.gamma, 0],[0, 0, self.gamma, -1*self.delta]])

        # For most populations in the network they will start disease free
        s_init, e_init, i_init, r_init = 1, 0, 0, 0

        # Selecting the inital infected population
        i_exposed = 1e-5

        # Population Proportions (s, e, i , r)
        theta_prop_init = np.array([s_init, e_init, i_init, r_init], dtype=np.float64).reshape((-1,1))
        theta_prop_infected = np.array([1 - i_exposed, i_exposed, 0, 0], dtype=np.float64)

        # Compute the population (S,E,I,R) values
        theta_props = np.repeat(theta_prop_init, self.N, axis=1) 
        theta_props[:,self.infected_idx] = theta_prop_infected

        self.thetas_B = theta_props * self.population_vector.T # Dimension 3 x N
        self.thetas_T = np.zeros(self.thetas_B.shape) # Dimension 3 x N - No initial people currently abroad

        self.state_history = []
        self.t = 0
        self.total_population = self.population_vector.sum()
        self.population_vector_flat = self.population_vector.reshape((-1,))
        
        
        # Infection States
        self.last_S_t_global = 0
        self.last_S_t_community = 0
        self.last_S_t_local = 0

        self.set_disabled_airports()

        # Starting State
        return self.state_to_idx(np.array([0, 0, 0]))

    def corrected(self, A,action):
        """
        If a airport is locked down it disables it. WARNING: Currently designed for a single agent only.
        """
        if action == 1:
            mask = np.ones_like(A)
            mask[:,self.agent_idx] = 0
            mask[self.agent_idx,:] = 0
            
            return A * mask
        else:
            return A
   
    def corrected_multiple(self, A, actions):
        agent_idxs = actions == 1

        mask = self.A_ones
        mask[:,agent_idxs] = 0
        mask[agent_idxs,:] = 0
        masked_A = A * mask

        return masked_A
        
    def corrected_degree(self, D, A_corrected, action):
        if action == 1:
            return A_corrected.sum(axis=1)
        else:
            return D
    
    @property
    def stateHistory(self):
        return np.array(self.state_history)
    
    def transform_state(self, current, last):
        if current > last:
            return 1 # Increase
        if current == last:
            return 0 # Static
        else:
            return -1 # Decrease
        
    def state_to_idx(self, state):
        # State[i] has -1,0,1 options
        # We want to simplify by starting with 1-indexing 
        
        idx1 = 2 + state[0]
        idx2 = 2 + state[1]
        idx3 = 2 + state[2]
        
        flattened_idx = idx1 + 3 * (idx2 - 1) + 9 * (idx3 - 1) 
        
        return flattened_idx - 1 # Zero indexed
    
    def set_disabled_airports(self, disable_airports = None):
        # Note we have now depreceated individual actions
        self.disable_airports = disable_airports

        if type(disable_airports) != type(None):
            if self.use_centrality:
                self.C_c = self.corrected_multiple(self.C, disable_airports)
                self.ID_c = self.corrected_degree(self.ID, self.C_c, 1)
            else:
                self.A_c = self.corrected_multiple(self.A, disable_airports) 
                self.D_c = self.corrected_degree(self.D, self.C_c, 1)
        else:
            if self.use_centrality:
                self.C_c = self.C.copy()
                self.ID_c = np.ones((self.ID.shape[0],))
            else:
                self.A_c = self.A.copy()
                self.D_c = self.D.copy()

        row, col = np.diag_indices(self.C_c.shape[0])
        if self.use_centrality:
            self.C_c[row, col] = self.C_c[row, col] - self.ID_c
            self.outer_faster = -1 * self.C_c # ID_c - C_c
        else:
            self.A_c[row, col] = self.A_c[row, col] - self.D_c
            self.outer_faster = -1 * self.A_c # ID_c - C_c     
        
            
    # RK4
    def ODE_solve(self, f_prime, y_0, step_size = 1):
        y = y_0

        # Slightly more complicated again step method
        k1 = step_size * f_prime(y)
        k2 = step_size * f_prime(y + 0.5 * k1)
        k3 = step_size * f_prime(y + 0.5 * k2)
        k4 = step_size * f_prime(y + k3)
        y = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

        return y
    
    def step(self, action, disable_travel = False):
        """
        WARNING: Action is no longer implemented
        :param action: 0 = Open; 1 = Lockdown -> leads to beta_reduced being utilised for the agents airport & No travel in or out permitted
        :param disable_travel: Disables all travel 
        :param disable_airports: Disable all travel for selected airports (N Binary Vector)
        :return: (State, Reward)
        """
        
        # reduce beta parameter based on how many days have passed
        scaling_factor = math.exp(-self.decay/365)
        self.B[0,0] *= scaling_factor
        self.B[1,0] *= scaling_factor
        self.B_reduced[0,0] *= scaling_factor
        self.B_reduced[1,0] *= scaling_factor
        
        # Populations
        thetas = self.thetas_B + self.thetas_T
        thetas_B_populations = self.thetas_B.sum(axis=0)
        thetas_T_populations = self.thetas_T.sum(axis=0)
        thetas_populations = thetas_B_populations + thetas_T_populations
        
         # Epidemic model
        def SIRS_coupled_ode(thetas):
            # Creates the vectors [(S_i * I_i/ M_i, I_i, R_i)^T, ....]
            diff_vector = np.ones((4,self.N))
            diff_vector[0,:] = (thetas[0,:]) * (thetas[2,:]) * (1/thetas.sum(axis=0))  # S
            diff_vector[1,:] = (thetas[1,:])                                           # E
            diff_vector[2,:] = (thetas[2,:])                                           # I
            diff_vector[3,:] = (thetas[3,:])                                           # R

            # Computes differential equation (4x4) @ (4,N) => (4,N)
            d_thetas = self.B @ diff_vector
            
            if self.lockdown_threshold >= 0:
                pct_infections = thetas[2,:] / self.population_vector_flat 
                actions = pct_infections > self.lockdown_threshold
                d_thetas[:, actions] = self.B_reduced @ diff_vector[:, actions].reshape(d_thetas.shape[0],actions.sum())
            elif type(self.disable_airports) != type(None):
                d_thetas[:, self.disable_airports] = self.B_reduced @ diff_vector[:, self.disable_airports].reshape(d_thetas.shape[0],self.disable_airports.sum())
         
            return d_thetas
        
        # Updating community sizes
        # Community Spread (Immediately before diffusion)
        #thetas_star = thetas + d_thetas # Airport Community States
        thetas_star = self.ODE_solve(SIRS_coupled_ode, thetas)
        
        thetas_B_ratio = thetas_B_populations / thetas_populations
        thetas_T_ratio = thetas_T_populations / thetas_populations

        thetas_B_star = thetas_B_ratio * thetas_star
        thetas_T_star = thetas_T_ratio * thetas_star

        # Travelling populations
#         omegas_plus_star = thetas_B_star @ self.alphas_plus # Departures
        omegas_plus_star = np.copy(thetas_B_star)
        for c in range(4):
            if c != 2:
                for i in range(self.N):
                    omegas_plus_star[c,i] *= self.alphas_plus[i,i] # Departures, ensuring that infected are not allowed to travel
            else:
                for i in range(self.N):
                    omegas_plus_star[c,i] *= 0
        omegas_minus_star = thetas_T_star # Arrivals (we got rid of alpha minus as it is redundant)

        # International Spread / Diffusion (1/D prevents simultaneous changes exceeding the supply)
        # Note we did the derivation and it turns out we can replace D with identity and A with C
        # This now successfully weights passenger destinations according to centrality    
        
        if self.use_centrality:
            if self.lockdown_threshold >= 0:
                pct_infections = thetas_star[2,:] / self.population_vector_flat
                actions = pct_infections > self.lockdown_threshold
                row, col = np.diag_indices(self.C_c.shape[0])
                self.C_c = self.corrected_multiple(self.C, actions)
                self.ID_c = self.corrected_degree(self.ID, self.C_c, 1)
                self.C_c[row, col] = self.C_c[row, col] - self.ID_c
                self.outer_faster = -1 * self.C_c
           
            d_omegas_plus = -1 * self.c_plus * ((omegas_plus_star * self.D_inv) @ self.outer_faster) 

            if self.t % 5 == 0:
                # Faster implementation:
                d_omegas_minus = -1 * self.c_minus * ((omegas_minus_star * self.D_inv) @ (self.outer_faster))
                # d_omegas_minus = -1 * self.c_minus * ((omegas_minus_star * self.D_inv) @ (ID_c - C_c))
            else:
                d_omegas_minus = np.zeros(omegas_minus_star.shape)

            if self.lockdown_threshold >= 0:
                pct_infections = thetas_star[2,:] / self.population_vector_flat
                actions = pct_infections > self.lockdown_threshold

        else:
            if self.lockdown_threshold >= 0:
                pct_infections = thetas_star[2,:] / self.population_vector_flat
                actions = pct_infections > self.lockdown_threshold
                row, col = np.diag_indices(self.C_c.shape[0])
                self.A_c = self.corrected_multiple(self.A, actions)
                self.D_c = self.corrected_degree(self.D, self.A_c, 1)
                self.A_c[row, col] = self.A_c[row, col] - self.D_c
                self.outer_faster = -1 * self.A_c # ID_c - C_c

            d_omegas_plus = -1 * self.c_plus * ((omegas_plus_star * self.D_inv) @ (self.outer_faster))

            if self.t % 5 == 0:
                d_omegas_minus = -1 * self.c_minus * ((omegas_minus_star * self.D_inv) @ (self.outer_faster))
            else:
                d_omegas_minus = np.zeros(omegas_minus_star.shape)
                
        # Net change in Community States 
        if disable_travel:
            self.thetas_B = thetas_B_star
            self.thetas_T = thetas_T_star
        else:
            self.thetas_B = thetas_B_star + np.minimum(d_omegas_plus,0) + np.maximum(d_omegas_minus,0) # Base population recieves returning travellers
            self.thetas_T = thetas_T_star + np.maximum(d_omegas_plus,0) + np.minimum(d_omegas_minus,0) # Transient population recieves travellers who have left their home country

        # Record state
        thetas = self.thetas_B + self.thetas_T
        thetas_populations = thetas.sum(axis=0)
        self.state_history.append(thetas)
        self.t += 1
        
        # Compute The State and Reward from last action for the agent
        S_t = thetas[2,self.agent_idx] # Number infected
        M_j = thetas_populations[self.agent_idx] # Current Population we will assume transient individuals count
        
        reward_unmitigated = S_t * self.lmbda * self.p / M_j
        reward_lockdown = (S_t * self.mu * self.p / M_j) + self.p
        reward = - 1 * (reward_unmitigated * (1 - action) + reward_lockdown * action)
        
        # Agent State
        S_t_local = S_t
        discrete_state_local = self.transform_state(S_t_local, self.last_S_t_local)
        
        # Community State
        S_t_community = thetas[2,self.community].sum()
        discrete_state_community = self.transform_state(S_t_community, self.last_S_t_community)
        
        # Global State
        S_t_global = thetas[2,:].sum()
        discrete_state_global = self.transform_state(S_t_global, self.last_S_t_global)
        
        self.last_S_t_local = S_t_local
        self.last_S_t_community = S_t_community
        self.last_S_t_global = S_t_global
        current_state = self.state_to_idx(np.array([discrete_state_local, discrete_state_community, discrete_state_global]))
        
        return current_state, reward


def fitness_function_gen(n_days):
    def fitness_function(X):
        # 1. Compute the peak infections and total recoveries
        gamma = 1/25
        beta = 3
        decay = 5
        eps = 1/7
        n_iters = 200
        country_to_disable = np.array(X, dtype=np.bool_)

        mapping = airport_country.reset_index()
        mapping.columns = ['IATA', 'country_IATA']

        # Airports
        to_disable = mapping['country_IATA'].isin(list(countries[country_to_disable])).values

        # As calculated during previous simulation
        max_peak_infections = 2885487730.5877514
        max_total_infections = 7138688697.766091

        env = EpidemicEnvironment(vals(adj_matrix), 
                                  vals(populations), 
                                  ATL, 
                                  get_community(ATL), 
                                  infected_idx = WUH, 
                                  alphas_plus = vals(alphas),
                                  lmbda = 10, 
                                  mu = 0.2, 
                                  centrality = centralities.values, 
                                  gamma = gamma, 
                                  beta = beta, 
                                  epsilon = eps, 
                                  beta_reduced = beta,
                                  decay = decay)

        env.reset()

        for i in range(n_days):
            env.step(0)

        env.set_disabled_airports(to_disable)

        for i in range(n_iters - n_days):
            env.step(0)

        infections = env.stateHistory[:,2,:].sum(axis=1)
        recoveries = env.stateHistory[:,3,:].sum(axis=1)
        population = env.stateHistory[0,:,:].sum()

        peak_infections = infections.max()
        total_infections = recoveries.max()

        reduced_pct_infections = (1 - (infections.max() / max_peak_infections))
        reduced_pct_recoveries = (1 - (recoveries.max() / max_total_infections)) 
        pct_nodes_enabled = (1 - (to_disable.sum() / to_disable.shape[0])) 

        # Preference towards higher number of nodes
        fitness = reduced_pct_recoveries * reduced_pct_infections * np.sin(0.5 * np.pi * pct_nodes_enabled)
        minimise = 1 - fitness

        return minimise
    return fitness_function


