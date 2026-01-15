import hydra
import torch
import warnings
import torch.multiprocessing as mp
import os
import yaml
from omegaconf import OmegaConf

from visualization import gui
from utils.common import MissionRecorder
from simulator import get_simulator
from mapping import get_mapper
from planning import get_planner


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg):
    # ================= Jarvis 注入逻辑 =================
    print("\n🔧 [System] 正在强行注入 Jarvis 交互式配置...")
    from omegaconf import OmegaConf
    
    # 构造配置，指向修改过的 confidence.py
    jarvis_config = OmegaConf.create({
        "_target_": "planning.confidence.Confidence", 
        "planner_name": "confidence",
        "type": "confidence",
        "radius": 0.5,
        "init_pose": [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        "robot_size": 0.3,
        "pitch_angle": None,
        "sample_num": 100,
        "max_roi_sample_num": 30,
        "use_confidence": True,
        "path_length_factor": 0.5,
        "render_ratio": 0.25,
        "explore_weight": 1000.0,
        "n_steps": 2000,
        "visualize": True,
        "replanning_steps": 10
    })
    
    if "planner" in cfg:
        cfg.planner = jarvis_config
    else:
        OmegaConf.update(cfg, "planner", jarvis_config, force_add=True)
        
    print("✅ [System] 注入完成，准备启动引擎！\n")
    # ===================================================

    if cfg.debug:
        mission_recorder = None
    else:
        experiment_path = os.path.join(
            cfg.experiment.output_dir,
            str(cfg.experiment.exp_id),
            cfg.scene.scene_name,
            cfg.planner.planner_name,
            str(cfg.experiment.run_id),
        )
        os.makedirs(experiment_path, exist_ok=True)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(f"{experiment_path}/exp_config.yaml", "w") as file:
            yaml.dump(cfg_dict, file)

        mission_recorder = MissionRecorder(experiment_path, cfg.experiment)

    # load components
    mapping_agent = get_mapper(cfg, device)
    simulator = get_simulator(cfg)
    planner = get_planner(cfg, device)

    # ================= 🔴 核心修复：防崩溃保护 🔴 =================
    # VLM/Ear 模块在加载时已经初始化了 multiprocessing context
    # 所以这里再次初始化会报错。我们需要捕获这个错误并忽略它。
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass # 忽略 "context has already been set" 错误
    # ============================================================

    if cfg.use_gui:
        init_event = mp.Event()
        q_mapper2gui = mp.Queue()
        q_gui2mapper = mp.Queue()
        q_planner2gui = mp.Queue()
        q_gui2planner = mp.Queue()

        mapping_agent.use_gui = True
        mapping_agent.q_mapper2gui = q_mapper2gui
        mapping_agent.q_gui2mapper = q_gui2mapper

        planner.q_planner2gui = q_planner2gui
        planner.q_gui2planner = q_planner2gui

        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
            "planner_receive": q_planner2gui,
            "planner_send": q_gui2planner,
        }
        gui_process = mp.Process(
            target=gui.run,
            args=(init_event, cfg.gui, params_gui),
        )
        gui_process.start()
        init_event.wait()

    mapping_agent.load_recorder(mission_recorder)
    mapping_agent.load_simulator(simulator)
    mapping_agent.load_planner(planner)

    mapping_agent.run()

    if cfg.use_gui:
        gui_process.join()

if __name__ == "__main__":
    main()

