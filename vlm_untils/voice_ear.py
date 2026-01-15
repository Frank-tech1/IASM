import speech_recognition as sr
import whisper
import warnings
import re
import time

# 过滤 FP16 警告
warnings.filterwarnings("ignore")

class VoiceEar:
    def __init__(self, model_size="small", device_index=6): 
        print(f"\n[Ear] 正在初始化 (Model: {model_size})...")
        self.model = whisper.load_model(model_size, device="cpu")
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(device_index=device_index)
        
        # ⚠️ 调高阈值：防止把底噪当成 "Go to the sofa"
        # 建议值：安静房间 300-500，有底噪/电流声 800-1500
        self.recognizer.energy_threshold = 1000 
        self.recognizer.dynamic_energy_threshold = False 
        self.recognizer.pause_threshold = 0.8 
        
        print(f"[Ear] 就绪! (麦克风 ID: {device_index})")

    def listen_once(self):
        """
        监听一次并返回结果（给机器人用的接口）
        """
        with self.mic as source:
            try:
                # 监听 (只听5秒，防止卡死)
                audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=5)
                
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                # 提示词引导
                prompt_text = "Commands: Find the bed. Look for the chair. Go to the sofa. Search for the table."
                
                result = self.model.transcribe(
                    "temp.wav", 
                    fp16=False, 
                    language='en',
                    initial_prompt=prompt_text,
                    no_speech_threshold=0.6 # 增加静音过滤
                )
                
                text = result["text"].strip().lower()
                
                # --- 核心纠错逻辑 ---
                corrections = {
                    "bat": "bed", "bad": "bed", "bet": "bed",
                    "find a": "find the"
                }
                for wrong, right in corrections.items():
                    if wrong in text:
                        text = text.replace(wrong, right)
                        
                # 关键词过滤
                keywords = ['find', 'go', 'look', 'search', 'chair', 'bed', 'sofa', 'table', 'kitchen']
                clean_text = re.sub(r'[^\w\s]', '', text)
                
                if len(clean_text) > 2 and any(w in clean_text for w in keywords):
                    return clean_text
                
                return None # 没听到有效指令返回 None

            except sr.WaitTimeoutError:
                return None
            except Exception:
                return None

# === 调试专用模块 ===
if __name__ == "__main__":
    ear = VoiceEar()
    print(f"✅ 调试模式启动 (阈值: {ear.recognizer.energy_threshold})")
    print("🎤 请不断说话测试 (按 Ctrl+C 退出)...")
    
    while True:
        try:
            # 这里我们手动循环调用 listen_once
            print("Listening...", end="\r")
            result = ear.listen_once()
            
            if result:
                # 只有听到有效结果才打印绿色
                print(f"👂 听到: \033[92m'{result}'\033[0m             ")
            else:
                # 没听到就打印个点，证明还在跑，没死机
                pass 
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n🛑 退出")
            break

