# Brain-on-Wheels
# 🚗 Active Inference Car — Simulation + Real-World WiFi Control

This project is a hybrid **simulation + real-robot control system** built using:

- **PyGame** for visualization  
- **Box2D** for physics  
- **PyTorch** for an **Active Inference RSSM (Recurrent State Space Model)**  
- **ESP32/ESP8266** for real-world sensing + motor control over WiFi  

The virtual car learns how its world behaves, predicts future sensory states, and chooses actions that minimize expected surprise — essentially a tiny Bayesian self-driving agent.  
Once trained, the same model can control a **physical robot** via WiFi.

---

## 🌟 Features

### 🧠 1. Recurrent State-Space Model (RSSM)

The system uses an simplified RSSM:

- Action embedding  
- Concatenation with observations  
- GRU-based latent transition  
- Neural decoder predicting next sensory state  

This model builds an internal predictive representation of the environment.

---

### 🤖 2. Active Inference Action Selection

Instead of reinforcement learning, the car uses a **planning-based Active Inference** controller:

1. For each candidate first-action  
2. Imagines future sensory states using the RSSM  
3. Evaluates a multi-term cost function  
4. Selects the action with lowest expected free energy  

The cost encourages:

- Avoiding obstacles  
- Maintaining clearance  
- Moving forward  
- Keeping balanced left/right spacing  
- Avoiding excessive action switching  
- Avoiding unnecessary stopping  

---

### 🌍 3. Infinite Procedural World

The environment is chunk-based and loads dynamically as the car moves.

Each chunk contains:

- Random rectangles  
- Random circles  
- Random physical damping/friction values  

Every run generates a completely new environment.

---

### 🚘 4. Car Physics + Sensors

The vehicle is a Box2D dynamic body with:

- Forward/backward linear velocity control  
- Left/right turning  
- Four ray-cast sensors (front, back, left, right)  
- Sensor normalization to [0, 1]  

These sensors feed into the RSSM and control system.

---

### 📡 5. Real Robot WiFi Integration

Switch to **real mode** at any point:

- Reads ultrasonic sensor values from an ESP32  
- Sends motor commands (`0=fwd, 1=bwd, 2=left, 3=right, 4=stop`)  
- Applies the same Active Inference logic to control the real robot  

This is sim-to-real transfer in action.

---

### 🎮 6. Training Modes

#### **Collect Mode**
- Car takes random actions  
- Fills replay buffer  
- RSSM trains until avg loss < threshold  

#### **Active Mode**
- Model predicts future  
- Active Inference selects actions  
- You may optionally switch to real robot control  

---

## ⚙️ Controls

| Key | Action |
|-----|--------|
| **R** | Reset simulation |
| **S** | Switch to real-car WiFi mode (Active Mode only) |

---


