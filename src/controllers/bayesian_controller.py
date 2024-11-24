from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np
from estimator.estimate import estimate
from copy import deepcopy

# This multi-agent controller shares parameters between agents
class BayesianMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None
        self.env_args = getattr(self.args,'env_args',{})
        self.grid_shape = self.env_args.get('grid_shape',[20,30])
        self.map_width = self.grid_shape[0]
        self.map_height = self.grid_shape[1]
        self.estimation_threshold = getattr(self.args,'estimation_threshold',50)

    def select_actions(self, step_number,env,ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        utility_function = getattr(self.args,'utility','expected')
        estimation = getattr(self.args,'estimation','False')
        if utility_function=='random':
            avail_actions_shape = avail_actions.shape
            chosen_actions_shape = (avail_actions_shape[0], avail_actions_shape[1])  # assuming given shape
            return th.randint(0, avail_actions_shape[2], chosen_actions_shape).cpu()
        if estimation:
            chosen_actions =  self.select_action_with_est(step_number,env,ep_batch, t_ep, t_env, bs, test_mode).cpu()
            return chosen_actions
        qvals = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions.cpu()

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
            
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs


    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape



    def select_action_with_est(self, step_number,env, ep_batch, t_ep, t_env, bs, test_mode):
        utility_function = getattr(self.args,'utility','expected')
        step_number_period = getattr(self.args,'estimation_step_size',90)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        x_loc_event, y_loc_event, x_loc_uav, y_loc_uav= self.get_event_and_uav_locations(ep_batch["obs"][:, t_ep],env.get_knowledge_map())

        if step_number < 1:
            self.estimated_intensity = None
        if x_loc_event == None:
            agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
            chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
            return chosen_actions
        if x_loc_event != None: 
            if  (step_number%step_number_period==0 or self.estimated_intensity is None):
                full_estimation_intensity, intensity_max, cov, std = estimate(10,x_loc_event, y_loc_event,self.args)
                max_p = np.max(full_estimation_intensity)

                self.estimated_intensity = full_estimation_intensity/max_p
                self.estimated_intensity,scaling_factor, iteration = self.bisection_scale_full_map(self.estimated_intensity)

            sampled_maps = self.generate_binary_maps_with_probabilities(self.estimated_intensity,1000)
            qvalsList = []
            knowledge_map_shape = self.grid_shape
            knowledge_map_size = np.prod(knowledge_map_shape)
            np.set_printoptions(formatter={'float': lambda x: f"{x:.5g}"}) 
            for sampled_map,probability in sampled_maps:
                model_ep = deepcopy(ep_batch)
                observedState = deepcopy(env.observed_state[0,:,:])
                observedState[observedState == 2] = sampled_map[observedState == 2]
                binary_array_events_tensor = th.tensor(observedState.flatten(), dtype=th.float32).to(model_ep["obs"].device)
                for x in range(ep_batch["obs"][:, t_ep].shape[1]):
                    model_ep["obs"][0, t_ep][x][:knowledge_map_size] = binary_array_events_tensor
                qvals = self.forward(model_ep, t_ep, test_mode=test_mode)
                qvalsList.append((qvals, probability))  
            qvals_tensor = th.stack([q[0] for q in qvalsList])  
            probabilities = th.tensor([q[1] for q in qvalsList], dtype=th.float32).to(qvals_tensor.device)
            if utility_function=='expected':
                expected_qvals = (qvals_tensor.squeeze(1) * probabilities.view(-1, 1, 1)).sum(dim=0)
                max_expected_actions = th.argmax(expected_qvals, dim=-1).view(qvals.shape[0], -1)  
                return max_expected_actions.cpu()
            elif utility_function =='risk_averse':
                adjusted_95th_percentile_qvals = self.probability_adjusted_percentile(qvals_tensor.squeeze(1), probabilities)
                max_percentile_actions = th.argmax(adjusted_95th_percentile_qvals, dim=-1).view(qvals.shape[0], -1)  
                return max_percentile_actions.cpu()


    def bisection_scale_full_map(self,intensity_map, target_mean=0.5, tolerance=1e-3, max_iterations=500):
        if target_mean/intensity_map.mean() > 1:
            low, high = 1, target_mean/(np.min(intensity_map)+1e-6) 
        else:
            return intensity_map*target_mean/intensity_map.mean(), target_mean/intensity_map.mean(), 1
            
        iteration = 0
        scaled_map = intensity_map.copy()
        
        while iteration < max_iterations:
            scaling_factor = (low + high) / 2
            
            scaled_map = np.clip(intensity_map * scaling_factor, 0, 1)
            
            current_mean = scaled_map.mean()
            
            if abs(current_mean - target_mean) < tolerance:
                break
            
            if current_mean < target_mean:
                low = scaling_factor
            else:
                high = scaling_factor
            
            iteration += 1

        return scaled_map, scaling_factor, iteration

    def generate_binary_maps_with_probabilities(self,intensity_map, K):
        N, M = intensity_map.shape
        samples_with_probabilities = []
        
        for _ in range(K):
            sample = (np.random.rand(N, M) < intensity_map).astype(int)
            
            prob_map = np.where(sample == 1, intensity_map, 1 - intensity_map)
            map_probability = np.prod(prob_map)  
            
            samples_with_probabilities.append((sample, map_probability))
        
        return samples_with_probabilities


    def probability_adjusted_percentile(self,qvals_tensor, probabilities, percentile=0.95):

        num_samples, n_agents, n_actions = qvals_tensor.shape
        adjusted_percentile_qvals = th.zeros((n_agents, n_actions), device=qvals_tensor.device)
        
        for agent in range(n_agents):
            agent_qvals = qvals_tensor[:, agent, :]  
            
            for action in range(n_actions):
                action_qvals = agent_qvals[:, action]  
                
                sorted_indices = th.argsort(action_qvals)
                sorted_qvals = action_qvals[sorted_indices]
                sorted_probabilities = probabilities[sorted_indices]
                
                cumulative_probs = th.cumsum(sorted_probabilities, dim=0)
                
                target_cumulative_prob = percentile
                percentile_index = th.searchsorted(cumulative_probs, target_cumulative_prob)
                
                adjusted_percentile_qvals[agent, action] = sorted_qvals[min(percentile_index, len(sorted_qvals) - 1)]
        
        return adjusted_percentile_qvals


    def get_event_and_uav_locations(self,observations,knowledge_map, queue_shape=(15), weights_shape=(15)):
        knowledge_map_shape = self.grid_shape
        knowledge_map_size = np.prod(knowledge_map_shape)

        uav_positions_size = np.prod(self.grid_shape)
        queue_size = np.prod(queue_shape)
        weights_size = np.prod(weights_shape)

        observations = observations.cpu()

        uav_positions_flat = observations[0][0][knowledge_map_size:knowledge_map_size + uav_positions_size]
        flattened_queues = observations[0][0][knowledge_map_size + uav_positions_size:knowledge_map_size + uav_positions_size + queue_size]
        flattened_weights = observations[0][0][knowledge_map_size + uav_positions_size + queue_size:knowledge_map_size*2+queue_size*2]
        

        uav_positions = uav_positions_flat.reshape(self.grid_shape)
        queues = flattened_queues.reshape(queue_shape)
        weights = flattened_weights.reshape(weights_shape)
        event_indices = np.argwhere(knowledge_map == 1)
        uav_indices = np.argwhere(uav_positions > 0)
        uav_x_coords = uav_indices[0].tolist()
        uav_y_coords = uav_indices[1].tolist()
        if event_indices.shape[0] < self.estimation_threshold:
            return None,None,uav_x_coords,uav_y_coords

        event_x_coords = event_indices[:,0].tolist()
        event_y_coords = event_indices[:,1].tolist()
        return event_x_coords, event_y_coords,uav_x_coords, uav_y_coords
