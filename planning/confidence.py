import torch
import numpy as np
import threading
import time
import sys
import cv2
import math
import json
import string
import random
from tqdm import tqdm
from einops import repeat

from planning.plan_base import PlanBase
from utils.common import TextColors
from utils.operations import GaussianRenderer

try:
    from vlm_untils.detector import SemanticBrain 
    from vlm_untils.voice_ear import VoiceEar     
except ImportError:
    SemanticBrain = None
    VoiceEar = None

class Confidence(PlanBase):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.render_ratio = cfg.render_ratio
        self.explore_weight = cfg.explore_weight 
        self.semantic_weight = 10.0              
        
        print("\n" + "█"*70)
        print("█  🤖 Jarvis (v11.1) - 无依赖视觉版 (Decoupled Vision)         █")
        print("█  👁️ 核心修复: 彻底移除 Recorder 依赖，确保旋转时也能检测     █")
        print("█"*70 + "\n")
        
        self.brain = None
        self.ear = None
        
        if SemanticBrain:
            try:
                print("   -> [Init] Connecting Visual Cortex...")
                self.brain = SemanticBrain()
                print("   -> [Brain] ✅ Online")
            except: pass

        if VoiceEar:
            try:
                print("   -> [Init] Connecting Auditory Nerve...")
                self.ear = VoiceEar(model_size="base.en", device_index=6)
                print("   -> [Ear] ✅ Online (Base.en)")
            except: pass

        self.latest_command = None
        self.listening = True
        self.spin_counter = 0 
        self.spin_total_steps = 18 
        self.state = 0 
        self.current_target = None
        self.nav_goal = None 
        self.intervention_action = None 
        self.refine_step = 0 
        self.mode = 0 
        self.simulator_ref = None

        if self.ear:
            self.listen_thread = threading.Thread(target=self._background_listen)
            self.listen_thread.daemon = True
            self.listen_thread.start()

    def _background_listen(self):
        print("   -> 👂 Listening in background...")
        while self.listening:
            try:
                time.sleep(0.05) 
                cmd = self.ear.listen_once()
                if cmd:
                    print(f"\n\033[93m🎤 [HEARD] >>> {cmd.upper()} <<<\033[0m", flush=True)
                    self.latest_command = cmd
            except: pass

    def _get_current_pose_matrix(self, simulator):
        try:
            if hasattr(simulator, 'sim'):
                return np.array(simulator.sim.get_agent(0).scene_node.transformation)
        except: pass
        return None

    def _get_stay_pose(self, simulator):
        """ 稳像悬停 (双帧) """
        pose = self._get_current_pose_matrix(simulator)
        if pose is not None:
            p = torch.from_numpy(pose).float().cpu().contiguous()
            return [p, p] 
        return []

    def _rotate_pose_spherical(self, current_pose_matrix, yaw_deg, pitch_deg):
        theta_y = np.radians(yaw_deg)
        c, s = np.cos(theta_y), np.sin(theta_y)
        R_yaw = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        theta_x = np.radians(pitch_deg)
        c, s = np.cos(theta_x), np.sin(theta_x)
        R_pitch = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
        return current_pose_matrix @ (R_yaw @ R_pitch)

    def _get_image_robust(self, simulator, recorder):
        # 优先仿真器直出
        if hasattr(simulator, 'sim'):
            try:
                obs = simulator.sim.get_sensor_observations()
                if 'color_sensor' in obs: 
                    img = obs['color_sensor']
                    if img.shape[-1] == 4: img = img[:, :, :3]
                    return img
                if 'rgb' in obs: 
                    img = obs['rgb']
                    if img.shape[-1] == 4: img = img[:, :, :3]
                    return img
            except: pass
        # 兜底 recorder
        if hasattr(recorder, 'last_frame') and recorder.last_frame is not None:
            return recorder.last_frame
        return None

    def parse_intent(self, cmd, voxel_map):
        cmd_clean = cmd.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        print(f"👂 [Brain] Processing: '{cmd_clean}'")
        words = cmd_clean.split()

        resume_keywords = ["continue", "resume", "proceed", "go", "move", "start", "keep", "ok", "okay"]
        if any(k in words for k in resume_keywords):
            if "stop" not in words and "wait" not in words:
                print("🟢 [System] Resuming exploration!")
                return 0, "explore"

        stop_keywords = ["stop", "wait", "halt", "hold", "pause", "break", "stay"]
        if any(k in words for k in stop_keywords):
            print("🛑 [Emergency] BRAKE!")
            return 4, "hover"
            
        target_obj = None
        triggers = ["find", "search", "look", "inspect", "where"]
        for t in triggers:
            if t in cmd_clean:
                try:
                    target_obj = cmd_clean.split(t, 1)[1].strip()
                    if target_obj.startswith("for "): target_obj = target_obj[4:]
                    if target_obj.startswith("the "): target_obj = target_obj[4:]
                except: pass
                break
        
        if target_obj and len(target_obj) > 1:
            print(f"🧠 [Memory] Recalling '{target_obj}'...", flush=True)
            # 只有当 voxel_map 有语义信息时才查
            if hasattr(voxel_map, 'semantic_probability'):
                best_pos, confidence = self.search_memory(voxel_map, target_obj)
            else:
                best_pos, confidence = None, 0.0
                
            self.current_target = target_obj
            
            if confidence > 0.15: 
                print(f"✅ [Memory] Found! (Conf: {confidence:.2f}) -> Navigating")
                self.nav_goal = best_pos
                return 2, "navigate" 
            else:
                print(f"❓ [Memory] Unknown. Starting spherical scan.")
                return 0, "search_new" 

        if self.brain:
            print(f"⚡ [FastThink] LLM Parsing...", end="", flush=True)
            ctrl_cmd = self.brain.parse_movement_command(cmd) 
            if ctrl_cmd:
                print(f" -> {ctrl_cmd}", flush=True)
                return 1, ctrl_cmd 
            else:
                print(" Ignored.")
        return 0, "explore"

    def search_memory(self, voxel_map, target_text):
        if voxel_map is None or not hasattr(voxel_map, 'semantic_probability'): return None, 0.0
        try:
            probs = voxel_map.semantic_probability.cpu().numpy()
            flat_indices = np.argsort(probs)[-100:] 
            top_probs = probs[flat_indices]
            mean_prob = np.mean(top_probs)
            if mean_prob < 0.15: return None, mean_prob
            centers = voxel_map.voxel_centers.cpu().numpy()[flat_indices]
            target_center = np.mean(centers, axis=0)
            return target_center, mean_prob
        except: return None, 0.0

    def _execute_intervention(self, current_pose_matrix, cmd_json):
        mode = cmd_json.get('mode', 'stop')
        if mode == 'stop': return self._get_stay_pose(self.simulator_ref)
        deltas = cmd_json.get('delta_pose', {})
        dx, dy, dz, dyaw = deltas.get('x', 0.0), deltas.get('y', 0.0), deltas.get('z', 0.0), deltas.get('yaw', 0.0)
        theta_y = np.radians(dyaw)
        c, s = np.cos(theta_y), np.sin(theta_y)
        R_yaw = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        translation_vec = np.array([dx, dy, -dz, 0]) 
        new_pose = current_pose_matrix @ R_yaw
        world_move = new_pose @ translation_vec
        new_pose[:3, 3] += world_move[:3]
        p = torch.from_numpy(new_pose).float().cpu().contiguous()
        return [p, p]

    def plan(self, current_map_tuple, simulator, recorder):
        # Unpack Map
        real_voxel_map = current_map_tuple[1] if isinstance(current_map_tuple, tuple) and len(current_map_tuple) > 1 else current_map_tuple
        self.simulator_ref = simulator 
        current_rgb = self._get_image_robust(simulator, recorder)

        # 🔴 [核心修复] 移除对 recorder 的依赖！
        if self.current_target and self.brain and current_rgb is not None:
             # 扫描时(spin>0)必定检测，平时随机检测
             should_detect = (self.spin_counter > 0) or (np.random.rand() < 0.3)
             
             if should_detect: 
                 try:
                     # 1. 视觉检测
                     bbox, lock_type = self.brain.detect_object(current_rgb, self.current_target)
                     
                     if bbox: 
                         print(f"\n🚀🚀🚀 [VISUAL LOCK] SIGHTED: {self.current_target}!", flush=True)
                         
                         # 2. 尝试更新语义地图
                         if hasattr(recorder, 'dataframe_list') and len(recorder.dataframe_list) > 0 and real_voxel_map is not None:
                             try:
                                 real_voxel_map.update_semantics(recorder.dataframe_list[-1], bbox)
                             except: pass
                         
                         # 3. 立即行动！
                         print(f"🗺️ [Nav] Visual Lock -> Moving to Target!", flush=True)
                         self.state = 2 
                         self.spin_counter = 0 
                         
                         pose_matrix = self._get_current_pose_matrix(simulator)
                         # 向前冲 1.5 米
                         forward_vec = pose_matrix[:3, 2] * (-1.5) 
                         self.nav_goal = pose_matrix[:3, 3] + forward_vec
                         
                         return self._get_stay_pose(simulator)
                     
                     # 调试：打印小点证明在看
                     elif self.spin_counter > 0:
                         print(".", end="", flush=True)
                         
                 except Exception as e:
                     print(f"Vis Error: {e}")

        # --- B. 指令解析 ---
        if self.latest_command:
            state_code, payload = self.parse_intent(self.latest_command, real_voxel_map)
            self.state = state_code
            if state_code == 0 and payload == "search_new":
                print("🔄 [Search] Starting Spherical Scan...", flush=True)
                self.spin_counter = self.spin_total_steps
            self.latest_command = None 
            if self.state == 1: self.intervention_action = payload
            elif self.state == 2: self.refine_step = 0

        pose = self._get_current_pose_matrix(simulator)
        if pose is None: return []
        curr_pos = pose[:3, 3]

        if self.state == 4:
            time.sleep(0.05) 
            if int(time.time()) % 2 == 0: print("\r\033[93m⏸️ [HOVER] Waiting...\033[0m", end="", flush=True)
            return self._get_stay_pose(simulator)

        if self.state == 1:
            next_pose = self._execute_intervention(pose, self.intervention_action)
            self.state = 4 
            return next_pose

        if self.state == 2 and self.nav_goal is not None:
            dist = np.linalg.norm(curr_pos - self.nav_goal)
            if dist < 1.0:
                print(f"📍 [Arrived] Target Reached.")
                self.state = 4 # Arrived -> Hover
                return self._get_stay_pose(simulator)

        if self.state == 3:
            self.state = 4
            return self._get_stay_pose(simulator)

        if self.spin_counter > 0:
            progress = (self.spin_total_steps - self.spin_counter) / self.spin_total_steps
            yaw_step = 20 
            pitch_delta = 10.0 * math.sin(progress * 4 * math.pi) 
            print(f"🌀 [Scan] {self.spin_counter}: Y+{yaw_step} P{pitch_delta:.1f}", flush=True)
            self.spin_counter -= 1
            
            # 扫描完还没找到 -> 强制移动
            if self.spin_counter == 0:
                print("\n⚠️ [Scan Failed] Nothing found. Forcing random exploration move!", flush=True)
                forward_noise = np.random.uniform(-1, 1, 3) 
                forward_noise[1] = 0 
                self.nav_goal = curr_pos + forward_noise * 3.0
                self.state = 2 
            
            p = torch.from_numpy(self._rotate_pose_spherical(pose, yaw_step, pitch_delta)).float().cpu().contiguous()
            return [p, p]

        self.mode = 1 if (self.current_target) else 0
        return super().plan(current_map_tuple, simulator, recorder)

    @torch.no_grad
    def cal_utility(self, gaussian_map, voxel_map, candidates, simulator):
        if self.latest_command:
            if "stop" in self.latest_command.lower() or "wait" in self.latest_command.lower():
                return torch.zeros(len(candidates)).cpu(), 0

        t_utility = 0
        render_resolution = np.round(self.render_ratio * simulator.resolution).astype(int)
        h, w = render_resolution
        depth_range = simulator.depth_range
        extrinsics = candidates.to(self.device)
        intrinsics = repeat(simulator.intrinsic, " h w -> v h w", v=len(candidates)).to(self.device)
        renderer = GaussianRenderer(extrinsics, intrinsics, gaussian_map.get_attr(), gaussian_map.background_color, (gaussian_map.scene_near, gaussian_map.scene_far), (h, w), self.device)
        
        explore_util = torch.zeros(len(candidates)).to(self.device)
        semantic_util = torch.zeros(len(candidates)).to(self.device)
        exploit_util = torch.zeros(len(candidates)).to(self.device)
        nav_util = torch.zeros(len(candidates)).to(self.device)

        if hasattr(voxel_map, 'semantic_probability'):
            sem_probs = voxel_map.semantic_probability.to(self.device)
        else:
            sem_probs = torch.zeros(torch.prod(voxel_map.dim)).to(self.device)
        occ_probs = voxel_map.voxel_states.to(self.device)

        if self.state == 2 and self.nav_goal is not None:
            cand_pos = extrinsics[:, :3, 3]
            goal_tensor = torch.tensor(self.nav_goal).float().to(self.device)
            dists = torch.norm(cand_pos - goal_tensor, dim=1)
            nav_util = 1.0 / (dists + 0.1)
            if torch.max(nav_util) > 0: nav_util /= torch.max(nav_util)

        for i in tqdm(range(len(extrinsics)), desc=f" {TextColors.CYAN}Evaluate{TextColors.RESET}"):
            t_start = time.time()
            (rgb, depth, normal, opacity, d2n, confidence, importance, count, _) = renderer.render_view(i)
            valid_mask = torch.ones(*render_resolution).bool().to(self.device)
            depth_voxel = depth[0].clone()
            depth_voxel[depth_voxel < 0.001] = 10000 
            depth_voxel = torch.clamp(depth_voxel, min=depth_range[0], max=depth_range[1])
            visible_mask = voxel_map.cal_visible_mask(extrinsics[i], intrinsics[i], depth_voxel).to(self.device)
            unexp_mask = voxel_map.unexplored_mask.to(self.device)
            explore_util[i] = torch.sum(visible_mask & unexp_mask) / len(voxel_map.voxel_centers)
            if self.mode == 1:
                score = torch.sum(visible_mask * occ_probs * sem_probs)
                semantic_util[i] = score / len(voxel_map.voxel_centers)
            confidences = confidence[0]
            outrange_mask = depth[0] > depth_range[1]
            confidences[outrange_mask] = 1.0
            uncertainty = 1 - confidences
            dist_aware = uncertainty * depth[0] / depth_range[1]
            exploit_util[i] = torch.mean(dist_aware)
            t_utility += time.time() - t_start

        exploit_util[torch.isnan(exploit_util)] = 0.0
        explore_util[torch.isnan(explore_util)] = 0.0
        semantic_util[torch.isnan(semantic_util)] = 0.0
        if torch.max(explore_util) > 0: explore_util /= torch.max(explore_util)
        if torch.max(semantic_util) > 0: semantic_util /= torch.max(semantic_util)

        alpha = self.explore_weight
        beta = self.semantic_weight if self.mode == 1 else 0.0
        gamma = 3.0 if self.state == 2 else 0.0 
        if self.state == 2: alpha = 0.0 
        
        utility = alpha * explore_util + beta * semantic_util + exploit_util + gamma * nav_util
        if self.mode == 1: print(f"   [Plan] MaxSemGain: {torch.max(semantic_util):.4f} (State={self.state})")
        return utility.cpu(), t_utility

