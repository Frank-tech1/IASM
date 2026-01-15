import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import warnings
import re
import json

warnings.filterwarnings("ignore")

class SemanticBrain:
    def __init__(self, model_path="Qwen/Qwen2-VL-7B-Instruct", device="cuda"):
        print(f"[Brain] Loading VLM: {model_path} (High Sensitivity Mode)...")
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=512*28*28)
            print("[Brain] ✅ Vision Online")
        except Exception as e:
            print(f"[Brain] ❌ Load Failed: {e}")
            raise e

    def detect_object(self, image_input, target_text):
        """
        全知感知模式：
        1. 优先寻找 target_text。
        2. 如果没有框但语义存在，返回中心框 (Soft Lock)。
        3. [未来扩展] 返回所有可见物体的列表，用于建立回溯地图。
        """
        if not isinstance(image_input, Image.Image):
            image = Image.fromarray(image_input)
        else:
            image = image_input

        # 🔴 [Prompt 升级] 强制高灵敏度，并要求描述
        # 我们问两个问题：1. 有没有？ 2. 在哪里？
        prompt_text = (
            f"Look carefully. Is there a '{target_text}' in this image? "
            f"Even if it is small, far away, or partial, say 'YES'. "
            f"If YES, provide the bounding box [ymin, xmin, ymax, xmax]. "
            f"Also describe what else you see briefly."
        )
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            # 增加 token 长度，允许模型多“思考”一会儿
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # 🔴 [调试] 打印 VLM 的心声
        if len(output_text.strip()) > 0:
            print(f"   👁️ [VLM Thought] {output_text[:100].replace(chr(10), ' ')}...", flush=True)

        # --- 解析逻辑 (三层漏斗) ---

        # 1. 尝试提取标准坐标框
        numbers = re.findall(r'\d+(?:\.\d+)?', output_text)
        nums = [float(x) for x in numbers]
        
        if len(nums) >= 4:
            # 取最后4个数字作为坐标 (防止前面有日期等数字干扰)
            coords = nums[-4:]
            final_coords = [val / 1000.0 if val > 1.0 else val for val in coords]
            y1, x1, y2, x2 = final_coords
            ymin, ymax = sorted([y1, y2])
            xmin, xmax = sorted([x1, x2])
            
            # 只有当框大到一定程度才认为是有效框 (防止噪点)
            if (ymax - ymin) * (xmax - xmin) > 0.001:
                return [xmin, ymin, xmax, ymax], "HARD_LOCK"

        # 2. [软锁定 - Soft Lock] 
        # 如果没有坐标，但模型说了 "YES" 或者 提到了目标名字，说明它看见了！
        # 这种情况下，我们不能放过，返回一个“中心视野框”，骗机器人往中间走
        target_keywords = target_text.lower().split()
        response_lower = output_text.lower()
        
        # 判定：是否包含 "yes" 且包含目标物体名
        is_positive = "yes" in response_lower or any(k in response_lower for k in target_keywords if len(k)>2)
        
        if is_positive:
            print(f"   ⚠️ [Soft Lock] VLM saw '{target_text}' but gave no coords. Moving forward to check!", flush=True)
            # 返回屏幕中心的一个虚构框 [0.4, 0.4, 0.6, 0.6]
            return [0.4, 0.4, 0.6, 0.6], "SOFT_LOCK"

        return None, "NONE"

    def parse_movement_command(self, text_command):
        # 保持不变
        system_prompt = "Output JSON: {\"mode\": \"adjust\"|\"stop\", \"delta_pose\": {\"x\":0.0, \"y\":0.0, \"z\":0.0, \"yaw\":0.0, \"pitch\":0.0}}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_command}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            start = output_text.find('{')
            end = output_text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(output_text[start:end].replace("'", '"'))
        except: pass
        return None

