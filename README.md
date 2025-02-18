# 🚀 CoDriveVis

![Version](https://img.shields.io/badge/version-1.0-blue)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)
[![GitHub contributors](https://img.shields.io/github/contributors/henryhcliu/CoDriveVis)](https://github.com/henryhcliu/CoDriveVis/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/henryhcliu/CoDriveVis)](https://github.com/henryhcliu/CoDriveVis/commits/main)

CoDriveVis is an advanced simulation framework designed for Autonomous Mobility on Demand (AMoD) systems. It integrates microscopic motion planning and macroscopic vehicle scheduling to optimize cooperative driving. By leveraging CARLA's perception capabilities, CoDriveVis generates multimodal outputs—including LiDAR point clouds and visual contexts—to enhance traffic system operations. This platform is tailored for cooperative driving applications and supports Large Multimodal Models (LMMs) to improve decision-making and efficiency.

## Features
✔️ **Multimodal Perception** – Generates LiDAR point clouds, BEV images, and textual descriptions for enhanced situational awareness.  
✔️ **Integrated AMoD Planning** – Combines vehicle scheduling and motion planning within a unified framework.  
✔️ **Supports Large Multimodal Models (LMMs)** – Enables vision-based decision-making.  
✔️ **CARLA-Enhanced Simulation** – Leverages CARLA’s advanced perception for high-fidelity simulations.  
✔️ **Collaborative Driving** – Implements cooperative strategies for connected autonomous vehicles.

## Overview
CoDriveVis is an innovative simulation framework designed to enhance the integration of microscopic motion planning and macroscopic vehicle scheduling in Autonomous Mobility on Demand (AMoD) systems. Leveraging the advanced perception capabilities of CARLA, CoDriveVis generates multimodal outputs, including LiDAR point clouds and visual contexts, to inform and optimize traffic system operations. This platform is tailored for cooperative driving applications and supports the use of large multimodal models (LMMs) for improved decision-making and efficiency in AMoD environments.

## Setup
1. **Install Dependencies**  
   - Use Python **3.8**. `miniconda` is recommended to manage the Python environments.
   - Install required packages:  
     ```bash
     pip install -r requirements.txt
     ```
2. **Set Up CARLA**  
   - Download and install **CARLA 0.9.14**.  
   - Ensure CARLA is running before launching CoDriveVis.  
3. **Modify Configuration**  
   - Edit `config.yaml` to integrate your **Large Multimodal Model (LMM)** for vision input.

## Running the Simulation
1. **Explore Bird-Eye-View Enhancements**  
   - The `bird-eye-view/` folder includes modifications to `carla-bird-eye-view`, enhancing collaborative scheduling and motion planning.  
2. **Inspect the Code**  
   - Review `main.py` to understand how CoDriveVis works.  
   - Customize key modules such as:  
     - **Scheduling** – `amod_env.schedule_vehicles()`
     - **Collaborative Planning** – `amod_env.cooperate_vehicles(transforms=None, ADMM_ON=ADMM_ON)`
3. **Run the Demo**  
   - Execute `demo.py` to visualize CoDriveVis in action:  
     ```bash
     python demo.py
     ```
   - This generates **BEV images** and corresponding text while vehicles follow planned trajectories.

## Tutorial
### Perception Module
The perception module is responsible for generating a comprehensive representation of the driving environment. It includes BEV image generation and supports multimodal data integration, enhancing vehicle perception and decision-making processes.

### Scheduling Module
This module optimizes vehicle allocation to passenger requests, leveraging real-time data and advanced algorithms to maximize service efficiency and minimize waiting times.

### Updating Module
The updating module ensures the system accurately represents the dynamic environment by updating the states of vehicles and passengers using adaptive algorithms.

### Planning Module
The Planning Module develops safe and efficient driving strategies by evaluating multiple trajectory options.  
- Implements Model Predictive Control (MPC) for real-time motion planning.  
- Supports Proximal Policy Optimization (PPO) for reinforcement learning-based decision-making.  
- Enables adaptive planning under dynamic traffic conditions.

### Replay and Interaction
CoDriveVis organizes runtime information for replay and data analysis, facilitating interaction and continuous improvement in cooperative driving strategies.

## Getting Help
- **Found a bug?** Open an issue on our [GitHub Issues](https://github.com/henryhcliu/CoDriveVis/issues).  
- **Have a question?** Start a discussion in our [Discussions](https://github.com/henryhcliu/CoDriveVis/discussions).  
- **Want to contribute?** Check out our [Contribution Guide](CONTRIBUTING.md).

---
For more information, please refer to the full documentation and examples included in the repository (Stay tuned! Release soon!).

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
