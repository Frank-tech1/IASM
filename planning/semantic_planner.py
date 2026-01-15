from planning.plan_base import PlannerBase
from vlm_utils.detector import SemanticBrain # 我们上一轮写的视觉大脑
from vlm_utils.voice_ear import VoiceEar     # 刚刚写的耳朵

class VoiceInterventionPlanner(PlannerBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 初始化双脑
        self.brain = SemanticBrain() # GPU
        self.ear = VoiceEar()        # CPU
        
        self.override_target = None
        self.intervention_steps = 0

    def plan(self, current_map, simulator, recorder):
        # 1. 每一帧都稍微听一下（或者按键触发，避免阻塞）
        # 这里的实现是阻塞式的，实际建议用多线程，或者每隔N帧听一次
        command = self.ear.listen_once(time_limit=2) 

        # 2. 如果听到指令，进行语义干涉
        if command and len(command) > 2:
            print(f"⚠️ Voice Intervention: {command}")
            
            # 获取当前画面
            current_rgb = simulator.get_current_rgb() # 需自行封装获取图像的方法
            
            # 让 VLM 找目标
            bbox = self.brain.detect_object(current_rgb, command)
            
            if bbox:
                print(f"🎯 Target Found at {bbox}")
                # 计算 3D 坐标
                cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
                depth = simulator.get_depth_at(cx, cy)
                self.override_target = self.unproject(cx, cy, depth, simulator.camera_pose)
                self.intervention_steps = 20 # 锁定目标跑 20 帧
            else:
                print("❌ I heard you, but I can't see it yet.")

        # 3. 状态机逻辑
        if self.intervention_steps > 0 and self.override_target is not None:
            # --- 干涉模式 ---
            self.intervention_steps -= 1
            # 生成去往 override_target 的路径
            return self.path_finder.plan(simulator.current_pose, self.override_target)
        else:
            # --- 自动探索模式 (默认) ---
            return super().plan(current_map, simulator, recorder)

