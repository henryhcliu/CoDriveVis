from utils.carla_utils import generate_spawn_points_nearby
import carla

import time
import numpy as np
import cv2
import random

from LMMAgent.driver_agent.driverAgent_llama import LMMScheduleAgent, LMMGraphAgent
from LMMAgent.bevGenerator import BEV_Generator
from LMMAgent.amodEnvParallel import amodEnv
from LMMAgent.driver_agent.BEV_memory import MemoryModule

import yaml
import os
import pickle
import joblib

def setup_env(config):
    os.environ["MEM_PATH"] = config['sche_memory_path']
    os.environ["GRAPH_MEM_PATH"] = config['graph_memory_path']
    os.environ["MODEL_CKPT"] = config['model_ckpt'] # swin: 'microsoft/swin-base-patch4-window12-384-in22k', CLIP: 'openai/clip-vit-base-patch32'
    os.environ["PROC_MEM_PATH"] = config['proc_sche_memory_path']
    os.environ["PROC_GRAPH_MEM_PATH"] = config['proc_graph_memory_path']
    os.environ["MEM_ROOT_PATH"] = config['memory_root_path']
    if config['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = config['AZURE_API_VERSION']
        os.environ["OPENAI_API_BASE"] = config['AZURE_API_BASE']
        os.environ["OPENAI_API_KEY"] = config['AZURE_API_KEY']
        os.environ["AZURE_CHAT_DEPLOY_NAME"] = config['AZURE_CHAT_DEPLOY_NAME']
        os.environ["AZURE_EMBED_DEPLOY_NAME"] = config['AZURE_EMBED_DEPLOY_NAME']
    elif config['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = config['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = config['OPENAI_KEY']
        os.environ["OPENAI_CHAT_MODEL"] = config['OPENAI_CHAT_MODEL']
        os.environ["OPENAI_API_BASE"] = config['OPENAI_API_BASE']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE, should be azure or openai")

if __name__ == '__main__':
    # setup the openai environment
    config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    setup_env(config)
    # generate the initial states
    carla_client = carla.Client('localhost', 2000)
    carla_client.set_timeout(5.0)
    carla_world = carla_client.get_world()
    # set the world to be Town10HD_Opt
    map_name = 'Town10HD_Opt'
    if str(carla_world.get_map()).split('/')[-1][:-1] == map_name:
        print('Already load map {}, skipped'.format(map_name))
    else:
        carla_world = carla_client.load_world(map_name)
    # set synchronous mode
    settings = carla_world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    carla_world.apply_settings(settings)

    carla_world.tick()
    # clean the vehicles and pedestrians
    for actor in carla_world.get_actors().filter('vehicle.*'):
        actor.destroy()
    for actor in carla_world.get_actors().filter('walker.*'):
        actor.destroy()
    carla_world.tick()
    
    # instantiate the BEV generator
    bev_generator = BEV_Generator('localhost', 2000)
    # instantiate the LMM schedule agent (output the couples of vehicle-pedestrain IDs)
    lmm_sche_agent = LMMScheduleAgent(sce=carla_world) # currently 'sce' is not used
    ckpt_path = os.environ["MODEL_CKPT"]
    schedule_memory_path = os.environ["PROC_MEM_PATH"] # change to the path of processed data
    lmm_sche_memory = MemoryModule(model_ckpt=ckpt_path, data_path=schedule_memory_path)
    # instantiate the LMM graph agent (output the couples of vehicles with collision risk)
    graph_memory_path = os.environ["GRAPH_MEM_PATH"] # no change is made here currently
    lmm_graph_memory = MemoryModule(model_ckpt=ckpt_path, BEV_data_path=graph_memory_path, embed_type='pooler')

    sim_duration = config['sim_duration']
    T_s = config['T_s'] # for scheduling
    T_p = config['T_p'] # for ADMM
    T_e = config['T_e'] # simulation time step

    # baseline comparison parameters


    # select a mode for the scheduling tasks from 'random' and 'from_file'
    mode = config['env_generation_mode']
    if mode == 'random':
        # generate vehicles with leagal initial states
        veh_num = config['vehicle_count']
        pedes_buf_num = config['pedestrian_buffer_count']
        few_shot_num = config['few_shot_num']
        transforms = carla_world.get_map().get_spawn_points()
        first_spp = random.choice(transforms)
        transforms.remove(first_spp)
        veh_states_init = []
        veh_states_init.append([first_spp.location.x, first_spp.location.y, first_spp.rotation.yaw/57.3])
        spawn_points_nearby = generate_spawn_points_nearby(carla_world, first_spp, 500, 10, transforms, veh_num-1) # default: 50, 5
        # remove the spawn_points_nearby from the transforms
        for transform in spawn_points_nearby:
            transforms.remove(transform)
            transform = transform
            veh_states_init.append([transform.location.x, transform.location.y, transform.rotation.yaw/57.3])
        veh_states_init = np.array(veh_states_init)
        print('Initial vehicle number: ', len(veh_states_init))

        # generate pedestrian transforms with leagal initial states
        pede_states_init = []
        pede_spawn_points_nearby = generate_spawn_points_nearby(carla_world, first_spp, 500, 15, transforms, pedes_buf_num+10) # generate 10 more spawn points than the buffer number in case of spawn failure because of physical conflicts

        for transform in pede_spawn_points_nearby:
            pede_states_init.append([transform.location.x, transform.location.y, transform.rotation.yaw/57.3])

        pede_states_init = np.array(pede_states_init)
        print('Initial pedestrian number: ', len(pede_states_init))

        # generate the destinations of each pede_state randomly with a range of [min, max]
        pede_destinations = []
        destination_range = [config['destination_dist_range_min'], config['destination_dist_range_max']] # not useful currently
        for i in range(len(pede_states_init)):
            destination = random.choice(transforms)
            # The judgment conditions include: the target point is different from the starting point; the distance from the target point to the starting point is between the min and max ranges.
            while (np.array([destination.location.x, destination.location.y]) == pede_states_init[i][:2]).all() or np.linalg.norm(np.array([destination.location.x, destination.location.y])-np.array(pede_states_init[i][:2])) < destination_range[0] or np.linalg.norm(np.array([destination.location.x, destination.location.y])-np.array(pede_states_init[i][:2])) > destination_range[1]:
                destination = random.choice(transforms)
            # remove the destination from the transforms
            transforms.remove(destination)
            pede_destinations.append([destination.location.x, destination.location.y, destination.rotation.yaw/57.3])
        pede_destinations = np.array(pede_destinations)
        print('Initial pedestrian destination number: ', len(pede_destinations))

        # spawn phase generation: 
        # state 0: spawn 5 more pedestrians than the number of vehicles at the time step 0. Note that the destinations of the pedestrians are randomly selected from the pede_destinations pool
        # for other states, randomly select a number of pedestrians to spawn with a range of [0, Max]
        spawn_num_max = config['spawn_num_max_pts']
        
        spawn_phase_record = []
        for i in range(int(sim_duration/T_s)+1):
            if i == 0:
                spawn_num = len(veh_states_init) + 1
            else:
                spawn_num = random.randint(0, spawn_num_max)
            if np.sum(spawn_phase_record) + spawn_num > pedes_buf_num:
                spawn_num = min(pedes_buf_num - np.sum(spawn_phase_record), 0)
            spawn_phase_record.append(int(spawn_num))
        print('Spawn phase record: ', spawn_phase_record)

        # store the initial states, the destinations, and the spawn phase record to the files
        folder_path = config['episode_folder_path']
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path+'veh_states_init.pkl', 'wb') as f:
            pickle.dump(veh_states_init, f)
        with open(folder_path+'pede_states_init.pkl', 'wb') as f:
            pickle.dump(pede_states_init, f)
        with open(folder_path+'pede_destinations.pkl', 'wb') as f:
            pickle.dump(pede_destinations, f)
        with open(folder_path+'spawn_phase_record.pkl', 'wb') as f:
            pickle.dump(spawn_phase_record, f)
    else:
        folder_path = config['episode_folder_path']
        with open(folder_path+'/veh_states_init.pkl', 'rb') as f:
            veh_states_init = pickle.load(f)
        with open(folder_path+'/pede_states_init.pkl', 'rb') as f:
            pede_states_init = pickle.load(f)
        with open(folder_path+'/pede_destinations.pkl', 'rb') as f:
            pede_destinations = pickle.load(f)
        with open(folder_path+'/spawn_phase_record.pkl', 'rb') as f:
            spawn_phase_record = pickle.load(f)
        few_shot_num = config['few_shot_num']
        # sum up the spawn_phase_record list to get the total number of pedestrians to spawn
        pedes_buf_num = np.sum(spawn_phase_record)


    # set up theamod environment
    amod_env = amodEnv(carla_world, veh_states_init, pede_states_init, pede_destinations, bev_generator, lmm_sche_agent, lmm_graph_agent, lmm_sche_memory, lmm_grap_memory, spawn_phase_record=spawn_phase_record,few_shot_num=few_shot_num, T_s=config['T_s'], T_p=config['T_p'], T_e=config['T_e'], sim_duration=config['sim_duration'], schedule_mode = config['schedule_mode'], graph_evo_mode=config['graph_evo_mode'])

    # spawn the vehicles and the first several pedestrians
    amod_env.spawn_init_agents(mode=mode)

    COMPLETE = False
    round_num = int(sim_duration/T_p)
    start_time = time.time()
    for i in range(round_num): 
        # check if the last pedes_info's ID is the same as the 
        if int(amod_env.pedes_info[-1]['ID']) >= pedes_buf_num-1:
            # check whether there are still pedestrians are not arrived, if not, terminate the simulation
            if amod_env.check_pedes_arrived():
                print('All the pedestrians have arrived at their destinations. The simulation is terminated.')
                break
        amod_env.schedule_vehicles()
        amod_env.update_vehicles_pedes() # high frequency
        print("***************Cooperative Vehicles for the time elapsed: ", amod_env.time_elapsed, "***************")
        amod_env.cooperate_vehicles(transforms=None)
        if amod_env.time_elapsed % amod_env.T_s == 0 and amod_env.time_elapsed != 0:
            amod_env.add_pedes()
        amod_env.time_elapsed += amod_env.T_e # update the time step
    # save the passenger information
    amod_env.save_pedes_info()
    end_time = time.time()
    print('Time elapsed for the whole process: ', end_time-start_time)

    # save updated memories in new dirs
    lmm_sche_memory.save_memory(os.environ["MEM_ROOT_PATH"])
    lmm_grap_memory.save_memory(os.environ["MEM_ROOT_PATH"])
    # save the optimized trajectories, subgraphs, and computing times
    save_path = 'recording/'+amod_env.scenario_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    joblib.dump(amod_env.optimized_trajss, save_path+'optimized_trajss'+'veh_num_'+str(len(veh_states_init))+'_pede_num_'+str(len(pede_states_init))+'_T_e_'+str(T_e)+'_T_s_'+str(T_s)+'_T_p_'+str(T_p)+'_sim_duration_'+str(sim_duration)+'_few_shot_num_'+str(few_shot_num)+'_schedule_mode_'+config['schedule_mode']+'_graph_evo_mode_'+config['graph_evo_mode']+'.pkl')
    joblib.dump(amod_env.subgraphss, save_path+'subgraphss'+'veh_num_'+str(len(veh_states_init))+'_pede_num_'+str(len(pede_states_init))+'_T_e_'+str(T_e)+'_T_s_'+str(T_s)+'_T_p_'+str(T_p)+'_sim_duration_'+str(sim_duration)+'_few_shot_num_'+str(few_shot_num)+'_schedule_mode_'+config['schedule_mode']+'_graph_evo_mode_'+config['graph_evo_mode']+'.pkl')
    joblib.dump(amod_env.computing_timess, save_path+'computingTimess'+'veh_num_'+str(len(veh_states_init))+'_pede_num_'+str(len(pede_states_init))+'_T_e_'+str(T_e)+'_T_s_'+str(T_s)+'_T_p_'+str(T_p)+'_sim_duration_'+str(sim_duration)+'_few_shot_num_'+str(few_shot_num)+'_schedule_mode_'+config['schedule_mode']+'_graph_evo_mode_'+config['graph_evo_mode']+'.pkl')
