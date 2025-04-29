from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *
import multiprocessing as mp
from tqdm import tqdm
import shutil
import uuid
import copy
import argparse
import random

# Dummy logger
def dummy_logger(msg):
    print(f"[LOG] {msg}")

# Config
c = Namespace(
    res=Path('results/single_ring'),
    render=False,
    sim_step=0.1,
    warmup_steps=500,
    skip_stat_steps=0,
    av_frac=0.5,
    start=True,
    generic_type='rand',
    vehicle_info_save=True,
    save_agent=True,
    sumo_no_errors=True,

    # ADDING FIELDS NEEDED BY YOUR STEP FUNCTION
    n_veh=22,  # default for single lane
    max_speed=10,
    max_accel=0.5,
    max_decel=0.5,
    circumference=250,
    circumference_max=300,
    circumference_min=200,
    circumference_range=None,
    initial_space='free',
    sigma=0.2,

    circ_feature=False,
    accel_feature=False,
    act_type='accel_discrete',  # matches your step() usage\
    low=-1,
    high=1,
    norm_action=True,
    global_reward=True,
    accel_penalty=0,
    collision_penalty=100,

    n_steps=100,
    gamma=0.9,
    alg=None,  # Placeholder (since no PG class defined here)
    norm_reward=True,
    center_reward=True,
    adv_norm=False,
    step_save=None,
)

# CONFIG DEFAULTS
c.log = dummy_logger
c._n_obs = 3 + int(c.circ_feature) + int(c.accel_feature)
c.observation_space = gym.spaces.Box(low=np.array([c.low] * c._n_obs), high=np.array([c.high] * c._n_obs), dtype=np.float32)

c.n_actions = 3
if c.act_type in ['accel_discrete', 'discrete']:
    c.action_space = Discrete(c.n_actions)
elif c.act_type in ['accel', 'discretize', 'continuous']:
    c.action_space = Box(low=c.low, high=c.high, shape=(1,), dtype=np.float32)
else:
    raise ValueError(f"Unsupported act_type: {c.act_type}")
c.start = False

def build_closed_route(edges, n_veh=0, av=0, space='random_free', type_fn=None, depart_speed=0, offset=0, init_length=None):
    assert isinstance(space, (float, int)) or space in ('equal', 'random_free', 'free', 'random', 'base', 'last')
    order = lambda i: edges[i:] + edges[:i + 1]

    # Prepare list of edges
    edge_list = list(edges)  # get list of E('edge', ...) elements
    routes = [E('route', id=f'route_{e.id}', edges=' '.join(e_.id for e_ in order(i))) for i, e in enumerate(edge_list)]
    rerouter = E('rerouter', E('interval', E('routeProbReroute', id=routes[0].id), begin=0, end=1e9), id='reroute', edges=edge_list[0].id)

    vehicles = []
    if n_veh > 0:
        lane_lengths, lane_routes, lane_idxs = map(np.array, zip(*[
            (float(e.length), r.id, i)  # 👈 cast e.length to float, numLanes to int
            for e, r in zip(edge_list, routes)
            for i in range(int(e.numLanes))
        ]))
        lane_ends = np.cumsum(lane_lengths)
        lane_starts = lane_ends - lane_lengths
        total_length = lane_ends[-1]
        init_length = init_length or total_length

        positions = (offset + np.linspace(0, init_length, n_veh, endpoint=False)) % total_length
        veh_lane = (positions.reshape(-1, 1) < lane_ends.reshape(1, -1)).argmax(axis=1)

        if space == 'equal':
            space = total_length / n_veh
        if isinstance(space, (float, int)):
            veh_lane_pos = positions - lane_starts[veh_lane]
        else:
            veh_lane_pos = [space] * n_veh
        
        veh_routes = lane_routes[veh_lane]
        veh_lane_idxs = lane_idxs[veh_lane]

        type_fn = type_fn or (lambda i: 'rl' if i < av else 'human')
        vehicles = [
            E('vehicle', id=f'{i}', type=type_fn(i), route=r, depart='0', departPos=p, departLane=l, departSpeed=depart_speed)
            for i, (r, p, l) in enumerate(zip(veh_routes, veh_lane_pos, veh_lane_idxs))
        ]

    return [*routes, rerouter, *vehicles]

# Custom SumoDef that avoids duplicate arguments and manually defines network and vehicles
class MySumoDef(SumoDef):
    def generate_sumo(self, **kwargs):
        c = self.c
        # Choose sumo or sumo-gui based on whether we want to render
        sumo_binary = 'sumo-gui' if c.render else 'sumo'

        args = [
            sumo_binary,
            '--net-file', str(self.dir / 'net.xml'),  # Load the road network
            '--additional-files', str(self.dir / 'add.xml'),  # Load vehicles and routes
            '--gui-settings-file', str(self.dir / 'gui.xml'),  # Load GUI display settings (optional, only matters if sumo-gui)
            '--begin', '0',
            '--step-length', str(c.sim_step),
            '--no-step-log', 'true',
            '--time-to-teleport', '-1',
            '--no-warnings', 'true',
            '--collision.action', 'remove',
            '--collision.check-junctions', 'true',
            '--max-depart-delay', '0.5',
            '--random', 'true',
            '--start', 'false'  # start paused initially so controller can catch up
        ]

        self.c.log(' '.join(args))
        return args

    def def_sumo(self):
        c = self.c
        r = c.circumference / (2 * np.pi)

        nodes = E('nodes',
            E('node', id='bottom', x=0, y=-r),
            E('node', id='top', x=0, y=r),
        )

        get_shape = lambda start, end: ' '.join('%.5f,%.5f' % (r*np.cos(i), r*np.sin(i)) for i in np.linspace(start, end, 80))

        edges = E('edges',
            E('edge', id='right', **{'from': 'bottom', 'to': 'top', 'length': str(c.circumference/2), 'shape': get_shape(-np.pi/2, np.pi/2), 'numLanes': '1'}),
            E('edge', id='left',  **{'from': 'top', 'to': 'bottom', 'length': str(c.circumference/2), 'shape': get_shape(np.pi/2, 3*np.pi/2), 'numLanes': '1'}),
        )

        connections = E('connections',
            E('connection', **{'from': 'left', 'to': 'right', 'fromLane': '0', 'toLane': '0'}),
            E('connection', **{'from': 'right', 'to': 'left', 'fromLane': '0', 'toLane': '0'})
        )

        additional = E('additional',
            E('vType', id='human', accel="1", decel="1.5", minGap="2", sigma=str(c.sigma), carFollowModel="IDM", laneChangeModel="LC2013"),
            E('vType', id='rl', accel="1", decel="1.5", minGap="2", sigma="0", carFollowModel="IDM", laneChangeModel="LC2013"),
            E('route', id='route0', edges='right left right'), 
            *build_closed_route(edges, c.n_veh, av=1, space=c.initial_space)
        )

        return self.save(nodes, edges, connections, additional)

# Custom Env using the patched SumoDef
class MySingleRingEnv(Env):
    
    def reset_sumo(self):
        c = self.c
        sumo_def = self.sumo_def

        generate_def = c.redef_sumo or not sumo_def.sumo_cmd
        if generate_def:
            kwargs = self.def_sumo()
            kwargs['net'] = sumo_def.generate_net(**kwargs)
            sumo_def.sumo_cmd = sumo_def.generate_sumo(**kwargs)
        self.tc = sumo_def.start_sumo(self.tc)
        if generate_def:
            self.sumo_paths = {k: p for k, p in kwargs.items() if k in SumoDef.file_args}
            defs = {k: E.from_path(p) for k, p in self.sumo_paths.items()}
            self.ts = TrafficState(c, self.tc, **defs)
        else:
            self.ts.reset(self.tc)
        self.ts.setup()
        success = self.init_vehicles()
        return success
    
    def reset(self):
        while True:
            if not self.reset_sumo():
                continue
            if (obs := self.init_env()) is not None:
                return obs


    def __init__(self, c):
        self.c = c.setdefaults(redef_sumo=False, skip_stat_steps=0, skip_vehicle_info_stat_steps=True)
        self.sumo_def = MySumoDef(c)  # Use the patched one
        self.tc = None
        self.ts = None
        self.rollout_info = NamedArrays()
        self._vehicle_info = self._agent_info = None
        if c.get('vehicle_info_save'):
            self._vehicle_info = []
            if c.get('save_agent'):
                self._agent_info = []
        self._step = 0

    def def_sumo(self, *args, **kwargs):
        return self.sumo_def.def_sumo()
        
    def step(self, action=None):
        c = self.c
        ts = self.ts
        max_speed = c.max_speed
        circ_max = max_dist = c.circumference_max
        circ_min = c.circumference_min
        rl_type = ts.types.rl
        
        if not rl_type.vehicles:
            super().step()
            return c.observation_space.low, 0, False, 0
        
        rl = nexti(rl_type.vehicles)
        if action is not None: # action is None only right after reset
            ts.tc.vehicle.setMinGap(rl.id, 0) # Set the minGap to 0 after the warmup period so the vehicle doesn't crash during warmup
            accel, lc = (action, None) #Setting lane change to NONE as we are not doing it in single ring
            if isinstance(accel, np.ndarray): accel = accel.item()
            if isinstance(lc, np.ndarray): lc = lc.item()
            #if c.norm_action and isinstance(accel, (float, np.floating)): 
            # TODO: set norm_action to true always. only experimenting normalized actions
            # accel = (accel - c.low) / (c.high - c.low)
            #TODO: No lanechange in single ring
            
            #TODO: Only trying out discrete and continuous action space
            if c.act_type == 'accel_discrete':
                idx = int(accel)  # accel is action.item(), already 0..(n_actions-1)
                
                # Linearly interpolate between -max_decel and max_accel
                frac = idx / (c.n_actions - 1)  # map idx to [0, 1]
                physical_accel = (1 - frac) * (-c.max_decel) + frac * c.max_accel
                
                ts.accel(rl, physical_accel)
            elif c.act_type == 'accel':
                # TODO: Normalize action here if exploring unnormalized actions
                ts.accel(rl, accel)
                
            #call super step() here for all human controlled vehicle types
            super().step()
            
        #error handling: cases=(collision, bad init)
        if len(ts.new_arrived | ts.new_collided):
            print('Detected collision')
            return c.observation_space.low, -c.collision_penalty, True, None
        elif len(ts.vehicles) < c.n_veh:
            print('Bad initialization occurred, fix the initialization function')
            return c.observation_space.low, 0, True, None
        
        #creating obs ater stepping into new state
        leader, dist = rl.leader()
        #TODO: Assuming only one lane in single ring
        obs = [rl.speed / max_speed, leader.speed / max_speed, dist / max_dist]
        if c.circ_feature:
            obs.append((c.circumference - circ_min) / (circ_max - circ_min))
        if c.accel_feature:
            obs.append(0 if leader.prev_speed is None else (leader.speed - leader.speed) / max_speed)
            
        obs = np.clip(obs, 0, 1) * (1 - c.low) + c.low
        #reward = rl.speed
        reward = np.mean([v.speed for v in (ts.vehicles if c.global_reward else rl_type.vehicles)])
        # print("rl type vehicles: ", len(rl_type.vehicles), " HDV: ", len(ts.vehicles))
        if c.accel_penalty and hasattr(self, 'last_speed'):
            reward -= c.accel_penalty * np.abs(rl.speed - self.last_speed) / c.sim_step
        self.last_speed = rl.speed

        return obs.astype(np.float32), reward, False, None
            
        
class RandomPolicy(torch.nn.Module):
    def forward(self, x):
        return torch.rand((x.shape[0], 1))
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(obs_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc_out = torch.nn.Linear(64, action_dim)
        # Weight Initialization
        for layer in [self.fc1, self.fc2, self.fc_out]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return torch.softmax(self.fc_out(x), dim=-1)
    
def collect_rollout_worker(model_params, config, seed, horizon):
    device = torch.device("cpu")
    
    config = copy.deepcopy(config)  # copy first!
    
    # ➔ 2. Create a unique folder for this worker
    unique_id = uuid.uuid4().hex[:8]  # random short string
    worker_dir = config.res.parent / f"{config.res.name}_worker_{unique_id}"
    worker_dir = Path(worker_dir)  # <- ADD THIS line after defining worker_dir
    worker_dir.mk()
    
    shutil.copytree(config.res.parent / config.res.name, worker_dir, dirs_exist_ok=True)

    
    config.res = worker_dir  # point config to new folder

    # ➔ 3. Create environment
    env = MySingleRingEnv(config)
    model = PolicyNetwork(env.c._n_obs, env.c.n_actions).to(device)
    model.load_state_dict(model_params)
    model.eval()

    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []

    done = False
    step = 0
    while step < horizon and not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_tensor = model(obs_tensor)
        dist = torch.distributions.Categorical(action_tensor)
        action = dist.sample()
        next_obs, reward, done, _ = env.step(action.item())

        obs_list.append(obs)
        action_list.append(action.item())
        reward_list.append(reward)
        done_list.append(done)

        obs = next_obs
        step += 1
    
    # ➔ 4. After rollout, clean up the worker's SUMO files
    try:
        shutil.rmtree(worker_dir)
    except Exception as e:
        print(f"Failed to delete {worker_dir}: {e}")

    return np.array(obs_list), np.array(action_list), np.array(reward_list), np.array(done_list)
    
def collect_rollout(env, model, horizon=1000):
    obs = env.reset()
    rollout = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'dones': [],
    }
    done = False
    step = 0

    while step < horizon and not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device) 
        with torch.no_grad():
            action_tensor = model(obs_tensor)
        dist = torch.distributions.Categorical(action_tensor)
        action = dist.sample()
        # print("action: ", action.item())
        next_obs, reward, done, info = env.step(action.item())

        rollout['obs'].append(obs)
        rollout['actions'].append(action.item())
        rollout['rewards'].append(reward)
        rollout['dones'].append(done)

        obs = next_obs
        step += 1

    return rollout


if __name__ == '__main__':
    mp.set_start_method('spawn')

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', help='Run policy visualization instead of training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model weights')
    parser.add_argument('--alg', type=str, default='baseline', help='Algorithm to use (e.g., baseline)')
    args = parser.parse_args()

    ALG = args.alg

    # device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.validate:
        # === Evaluation Mode ===
        if ALG == "baseline":
            c.render = True  # Force SUMO-GUI
            c.start = True   # Start SUMO immediately
            c.sim_step = 0.5

            env = MySingleRingEnv(c)
            model = PolicyNetwork(env.c._n_obs, env.c.n_actions).to(device)

            if args.model_path is None:
                raise ValueError("Model path must be provided with --model_path when --validate is set.")

            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()

            obs = env.reset()

            # Warmup: random steps between 900 to 1100
            warmup_steps = random.randint(900, 1100)
            print(f"Warmup for {warmup_steps} steps...")
            for _ in range(warmup_steps):
                env.step()  # no action given = random/default

            print("Starting policy control...")
            for step in range(1000):
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action_tensor = model(obs_tensor)
                dist = torch.distributions.Categorical(action_tensor)
                action = dist.sample()

                obs, reward, done, info = env.step(action.item())
                import time
                time.sleep(0.05)  # slow down a bit for visualization
                if done:
                    break

            env.close()
        else:
            print(f"Invalid algorithm: {ALG}")
            exit()

    else:
        # === Training Mode ===
        if ALG == "baseline":
            env = MySingleRingEnv(c)  # Only to get obs_dim and action_dim
            obs_dim = env.c._n_obs
            action_dim = env.c.n_actions
            del env  # no need to keep env at global

            model = PolicyNetwork(obs_dim, action_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            n_epochs = 500
            batch_size = 32
            horizon = 1000

            for epoch in tqdm(range(n_epochs), desc="Training Progress", ncols=100):
                seeds = np.random.randint(0, 100000, size=batch_size)
                model_params = model.cpu().state_dict()  # very important: send model on CPU
                model.to(device)

                with mp.Pool(processes=batch_size) as pool:
                    results = pool.starmap(
                        collect_rollout_worker,
                        [(model_params, c, seeds[i], horizon) for i in range(batch_size)]
                    )

                batch_obs = []
                batch_actions = []
                batch_rewards = []

                for obs_arr, actions_arr, rewards_arr, _ in results:
                    batch_obs.append(obs_arr)
                    batch_actions.append(actions_arr)
                    batch_rewards.append(rewards_arr)

                batch_obs = np.concatenate(batch_obs)
                batch_actions = np.concatenate(batch_actions)
                batch_rewards = np.concatenate(batch_rewards)

                returns = []
                G = 0
                gamma = c.gamma
                for r in reversed(batch_rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float32).to(device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                obs = torch.tensor(batch_obs, dtype=torch.float32).to(device)
                actions = torch.tensor(batch_actions, dtype=torch.int64).to(device)

                logits = model(obs)
                dist = torch.distributions.Categorical(logits)
                log_probs = dist.log_prob(actions)

                loss = -(log_probs * returns).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_reward = np.mean(batch_rewards)
                tqdm.write(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.4f}")

                if (epoch+1) % 100 == 0:
                    save_path = Path(c.res) / f"trained_policy_epoch_{ALG}_{epoch+1}.pth"
                    torch.save(model.state_dict(), save_path)
                    print(f"Model saved to: {save_path} epoch: {epoch+1}")
        else:
            print(f"Invalid algorithm: {ALG}")
            exit()

# rollout = collect_rollout(env, model, horizon=1000)

# print("Rollout collected:")
# print("Observations:", np.array(rollout['obs']).shape)
# print("Actions:", np.array(rollout['actions']).shape)
# print("Rewards:", np.array(rollout['rewards']).shape)
# print("Dones:", np.array(rollout['dones']).shape)
# import time

# for _ in range(1000):
#     obs, reward, done, _ = env.step()
#     print(f"Step reward: {reward}")
#     time.sleep(0.05)
# env.close()