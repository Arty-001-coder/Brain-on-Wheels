import pygame, random, sys, math, time, socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from Box2D import (
    b2World, b2PolygonShape, b2CircleShape,
    b2_dynamicBody, b2_staticBody,
    b2ContactListener, b2RayCastCallback
)

# =========================================================
# CONFIGURATION
# ==================================================pyt=======
WIDTH, HEIGHT = 1800, 1000
FPS = 60
PPM = 20.0
TIME_STEP = 1.0 / FPS
VEL_ITERS, POS_ITERS = 6, 2
MAX_SENSOR_DIST = 10.0
DEVICE = "cpu"
PLANNING_HORIZON = 3

# =========================================================
# COLORS
# =========================================================
GRAY = (30, 30, 30)
RED = (200, 50, 50)
BLUE = (50, 100, 255)
WHITE = (240, 240, 240)
YELLOW = (255, 255, 100)
GREEN = (100, 255, 100)
PURPLE = (180, 50, 180)

# =========================================================
# PYGAME INIT
# =========================================================
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Active Inference Car - WiFi Integration")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)

# =========================================================
# WORLD + HELPERS
# =========================================================
world = b2World(gravity=(0, 0))
CHUNK_SIZE = 800
OBSTACLES_PER_CHUNK = 12
world_chunks = {}
chunk_damping = {}

# =========================================================
# COLLISION + SENSOR CLASSES
# =========================================================
class CollisionListener(b2ContactListener):
    def __init__(self): super().__init__(); self.collisions = 0
    def BeginContact(self, contact):
        a, b = contact.fixtureA.body, contact.fixtureB.body
        if a.userData == "car" or b.userData == "car":
            self.collisions += 1

class SensorRayCastCallback(b2RayCastCallback):
    def __init__(self): super().__init__(); self.hit = False; self.distance = 1.0
    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body.userData == "obstacle":
            self.hit, self.distance = True, fraction
            return fraction
        return -1.0

collision_listener = CollisionListener()
world.contactListener = collision_listener

# =========================================================
# WORLD GENERATION
# =========================================================
def generate_chunk(world, cx, cy):
    seed_val = random.randint(0, 9999999)
    random.seed(seed_val)
    obstacles = []
    damping = random.uniform(0.5, 3.0)
    chunk_damping[(cx, cy)] = damping
    for _ in range(OBSTACLES_PER_CHUNK):
        shape_type = random.choice(["rect", "circle"])
        ox = cx * CHUNK_SIZE + random.randint(0, CHUNK_SIZE)
        oy = cy * CHUNK_SIZE + random.randint(0, CHUNK_SIZE)
        if shape_type == "rect":
            w, h = random.randint(40, 100), random.randint(40, 100)
            body = world.CreateStaticBody(position=(ox/PPM, oy/PPM), userData="obstacle")
            body.CreateFixture(shape=b2PolygonShape(box=(w/(2*PPM), h/(2*PPM))))
            obstacles.append(("rect", body, w, h))
        else:
            r = random.randint(20, 60)
            body = world.CreateStaticBody(position=(ox/PPM, oy/PPM), userData="obstacle")
            body.CreateFixture(shape=b2CircleShape(radius=r/PPM))
            obstacles.append(("circle", body, r))
    return obstacles

def get_visible_chunks(car):
    cx = int(car.body.position.x * PPM // CHUNK_SIZE)
    cy = int(car.body.position.y * PPM // CHUNK_SIZE)
    return [(cx+dx, cy+dy) for dx in range(-1, 2) for dy in range(-1, 2)]

def ensure_chunks_loaded(world, car):
    for (cx, cy) in get_visible_chunks(car):
        if (cx, cy) not in world_chunks:
            world_chunks[(cx, cy)] = generate_chunk(world, cx, cy)

# =========================================================
# CAR CLASS
# =========================================================
class Car:
    def __init__(self, world, x, y):
        self.body = world.CreateDynamicBody(position=(x/PPM, y/PPM), angle=0, userData="car")
        self.body.CreatePolygonFixture(box=(1.25, 0.625))
        self.speed = 7.0
        self.turn_speed = 3.0
        self.sensor_distances = [MAX_SENSOR_DIST]*4
        self.prev_action = 4

    def control(self, action):
        lin_vel, ang_vel = (0, 0), 0.0
        if action == 0:
            dir_vec = self.body.GetWorldVector((0, 1))
            lin_vel = (dir_vec[0]*self.speed, dir_vec[1]*self.speed)
        elif action == 1:
            dir_vec = self.body.GetWorldVector((0, -1))
            lin_vel = (dir_vec[0]*self.speed, dir_vec[1]*self.speed)
        elif action == 2: ang_vel = self.turn_speed
        elif action == 3: ang_vel = -self.turn_speed
        self.body.linearVelocity, self.body.angularVelocity = lin_vel, ang_vel
        self.prev_action = action

    def cast_sensors(self, world, offset_x, offset_y):
        sensor_angles = [0, math.pi, math.pi/2 - math.radians(45), -math.pi/2 + math.radians(45)]
        readings = []
        for ang in sensor_angles:
            dir_world = self.body.GetWorldVector((math.sin(ang), math.cos(ang)))
            start = self.body.position
            end = (start[0]+dir_world[0]*MAX_SENSOR_DIST, start[1]+dir_world[1]*MAX_SENSOR_DIST)
            cb = SensorRayCastCallback(); world.RayCast(cb, start, end)
            dist = cb.distance*MAX_SENSOR_DIST if cb.hit else MAX_SENSOR_DIST
            readings.append(dist)
            sx1, sy1 = start[0]*PPM-offset_x, HEIGHT-(start[1]*PPM-offset_y)
            sx2, sy2 = end[0]*PPM-offset_x, HEIGHT-(end[1]*PPM-offset_y)
            color = GREEN if cb.hit else YELLOW
            pygame.draw.line(screen, color, (sx1, sy1), (sx2, sy2), 2)
        self.sensor_distances = readings
        return readings

    def get_state(self):
        sensors_norm = np.array(self.sensor_distances, dtype=np.float32) / MAX_SENSOR_DIST
        prev_action_onehot = np.zeros(5, dtype=np.float32)
        prev_action_onehot[self.prev_action] = 1.0
        return np.concatenate([sensors_norm, prev_action_onehot])

    def draw(self, offset_x, offset_y):
        v = [(vx*PPM, vy*PPM) for vx, vy in self.body.fixtures[0].shape.vertices]
        transformed = []
        for vx, vy in v:
            wx, wy = self.body.GetWorldPoint((vx/PPM, vy/PPM))
            sx, sy = wx*PPM-offset_x, HEIGHT-(wy*PPM-offset_y)
            transformed.append((sx, sy))
        pygame.draw.polygon(screen, BLUE, transformed)

# =========================================================
# IMPROVED RSSM MODEL
# =========================================================
class ImprovedRSSM(nn.Module):
    def __init__(self, obs_dim=9, action_dim=5, hidden_dim=256, action_emb=32):
        super().__init__()
        self.action_emb = nn.Sequential(
            nn.Linear(action_dim, action_emb), nn.ReLU(),
            nn.Linear(action_emb, action_emb), nn.ReLU()
        )
        self.gru = nn.GRUCell(obs_dim + action_emb, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4), nn.ReLU(),
            nn.Linear(hidden_dim//4, obs_dim)
        )
    def forward_step(self, s_t, a_t, h_t):
        a_e = self.action_emb(a_t)
        h_next = self.gru(torch.cat([s_t, a_e], dim=-1), h_t)
        s_next_pred = self.decoder(h_next)
        return h_next, s_next_pred
    def rollout(self, s_t, actions, h_t):
        states = []; h = h_t; s = s_t
        for a in actions:
            h, s = self.forward_step(s, a, h)
            states.append(s)
        return states, h

# =========================================================
# REPLAY BUFFER
# =========================================================
class ReplayBuffer:
    def __init__(self, cap=10000): self.buf = deque(maxlen=cap)
    def add(self, s, a, s_next): self.buf.append((s, a, s_next))
    def sample(self, n=64):
        b = random.sample(self.buf, min(n, len(self.buf)))
        s, a, sn = zip(*b)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(np.array(a), dtype=torch.float32),
                torch.tensor(np.array(sn), dtype=torch.float32))

# =========================================================
# TRAJECTORY COST + POLICY
# =========================================================
def compute_trajectory_cost(predicted_states, actions):
    total_cost = 0.0
    for i, s_pred in enumerate(predicted_states):
        sensors = s_pred[0, :4]
        min_dist = torch.min(sensors)
        obstacle_cost = 10.0 * torch.exp(-5.0 * min_dist)
        clearance_cost = 5.0 * torch.exp(-3.0 * torch.mean(sensors))
        forward_reward = -2.0 * sensors[0]
        balance_cost = torch.abs(sensors[2] - sensors[3])
        discount = 0.9 ** i
        total_cost += discount * (obstacle_cost + clearance_cost + forward_reward + balance_cost)
    action_changes = sum(torch.argmax(actions[i]) != torch.argmax(actions[i+1]) for i in range(len(actions)-1))
    stop_penalty = sum(torch.argmax(a) == 4 for a in actions)
    total_cost += 2.0 * action_changes + stop_penalty
    return total_cost

def choose_action_active_inference(model, state, h_t):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    best_action, best_cost = 4, float('inf')
    for first_action in range(5):
        seq = []
        for t in range(PLANNING_HORIZON):
            a = torch.zeros(1, 5)
            if t == 0 or first_action in [2, 3]: a[0, first_action] = 1.0
            else: a[0, 0] = 1.0
            seq.append(a)
        with torch.no_grad(): predicted_states, _ = model.rollout(state_t, seq, h_t)
        cost = compute_trajectory_cost(predicted_states, seq)
        if cost < best_cost: best_cost, best_action = cost, first_action
    return best_action

# =========================================================
# REAL CAR WIFI INTERFACE
# =========================================================
class RealCarInterface:
    def __init__(self, esp32_ip="10.147.205.2", esp32_port=8080):
        self.esp32_ip = esp32_ip
        self.esp32_port = esp32_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(1.0)
        print(f"🔗 Connecting to ESP32 at {esp32_ip}:{esp32_port} ...")
        try:
            self.sock.connect((esp32_ip, esp32_port))
            print("✅ Connected to ESP32 over Wi-Fi")
        except Exception as e:
            print("❌ Connection failed:", e)
            self.sock = None

    def read_sensors(self):
        if not self.sock:
            return [MAX_SENSOR_DIST * PPM] * 4

        try:
            data = self.sock.recv(128).decode().strip()
            if not data:
                return [MAX_SENSOR_DIST * PPM] * 4

            # ESP8266 sends JSON format: {"front":300.00,"back":300.00,"left":300.00,"right":300.00}
            import json
            try:
                sensor_data = json.loads(data)
                front = sensor_data.get('front', MAX_SENSOR_DIST * PPM)
                back = sensor_data.get('back', MAX_SENSOR_DIST * PPM)
                left = sensor_data.get('left', MAX_SENSOR_DIST * PPM)
                right = sensor_data.get('right', MAX_SENSOR_DIST * PPM)
                return [front, back, left, right]
            except json.JSONDecodeError:
                # Fallback: try old format F:300.0,B:310.0,L:290.0,R:280.0
                readings = {"F": MAX_SENSOR_DIST * PPM, "B": MAX_SENSOR_DIST * PPM,
                            "L": MAX_SENSOR_DIST * PPM, "R": MAX_SENSOR_DIST * PPM}
                parts = data.split(",")
                for p in parts:
                    if ":" in p:
                        key, val = p.split(":")
                        key = key.strip().upper()
                        try:
                            val = float(val.strip())
                            if key in readings:
                                readings[key] = val
                        except ValueError:
                            pass  # Skip invalid values
                vals = [readings["F"], readings["B"], readings["L"], readings["R"]]
                return np.clip(vals, 0, MAX_SENSOR_DIST * PPM)

        except Exception as e:
            print("⚠️ Socket read error:", e)
            return [MAX_SENSOR_DIST * PPM] * 4

    def send_action(self, action):
        """Send action to ESP32: 0=fwd, 1=bwd, 2=left, 3=right, 4=stop"""
        if not self.sock:
            return

        try:
            cmd = str(action) + '\n'
            self.sock.sendall(cmd.encode())
        except Exception as e:
            print(f"⚠️ Send action error: {e}")


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    global world, world_chunks
    model = ImprovedRSSM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    replay = ReplayBuffer()
    h_t = torch.zeros(1, 256)
    car = Car(world, WIDTH/2, HEIGHT/2)
    mode, real_mode = "collect", False
    train_loss, avg_loss = None, 1.0
    loss_history = deque(maxlen=100)
    real_interface = None

    print("Press R to reset simulation")

    while True:
        dt = clock.tick(FPS) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    print("🔁 Resetting simulation...")
                    world_chunks.clear()
                    world = b2World(gravity=(0, 0))
                    world.contactListener = collision_listener
                    replay = ReplayBuffer()
                    model = ImprovedRSSM().to(DEVICE)
                    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
                    car = Car(world, WIDTH/2, HEIGHT/2)
                    mode, real_mode = "collect", False
                    h_t = torch.zeros(1, 256)
                    continue

                if e.key == pygame.K_s and mode == "active" and not real_mode:
                    print("🔗 Switching to REAL CAR (Wi-Fi) control...")
                    real_interface = RealCarInterface("10.147.205.2", 8080)
                    real_mode = True
                    continue

        if real_mode:
            # === REAL CAR CONTROL ===
            sensor_vals = real_interface.read_sensors()
            sensors_norm = (np.array(sensor_vals, dtype=np.float32)/100.0 )
            prev_action_onehot = np.zeros(5, dtype=np.float32)
            state = np.concatenate([sensors_norm, prev_action_onehot])
            action = choose_action_active_inference(model, state, h_t)
            real_interface.send_action(action)
            continue

        # === SIMULATION ===
        offset_x = car.body.position.x * PPM - WIDTH / 2
        offset_y = car.body.position.y * PPM - HEIGHT / 2
        car.cast_sensors(world, offset_x, offset_y)
        state = car.get_state()

        if mode == "collect":
            action = random.randint(0, 4)
        else:
            action = choose_action_active_inference(model, state, h_t)

        car.control(action)
        world.Step(TIME_STEP, VEL_ITERS, POS_ITERS)
        car.cast_sensors(world, offset_x, offset_y)
        next_state = car.get_state()

        action_onehot = np.zeros(5); action_onehot[action] = 1.0
        replay.add(state, action_onehot, next_state)

        if len(replay.buf) > 512:
            for _ in range(4):
                s, a, sn = replay.sample(64)
                h = torch.zeros(64, 256)
                _, pred = model.forward_step(s, a, h)
                loss = F.mse_loss(pred, sn)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); loss_history.append(loss.item())
            avg_loss = sum(loss_history)/len(loss_history)
            if avg_loss < 0.005 and mode == "collect":
                print(f"✅ Switching to ACTIVE mode (avg loss={avg_loss:.6f})")
                mode = "active"

        # === DRAWING ===
        screen.fill(GRAY)
        ensure_chunks_loaded(world, car)
        for (cx, cy) in get_visible_chunks(car):
            for data in world_chunks.get((cx, cy), []):
                if data[0] == "rect":
                    _, b, w, h = data
                    bx, by = b.position
                    sx, sy = bx*PPM-offset_x, HEIGHT-(by*PPM-offset_y)
                    pygame.draw.rect(screen, RED, pygame.Rect(sx-w/2, sy-h/2, w, h))
                else:
                    _, b, r = data
                    bx, by = b.position
                    sx, sy = bx*PPM-offset_x, HEIGHT-(by*PPM-offset_y)
                    pygame.draw.circle(screen, PURPLE, (int(sx), int(sy)), int(r))

        car.draw(offset_x, offset_y)
        txts = [
            f"Mode: {mode.upper()} {'[REAL]' if real_mode else ''}",
            f"Replay: {len(replay.buf)}",
            f"Loss: {loss_history[-1]:.6f}" if len(loss_history) else "Loss: ---",
            f"Avg Loss: {avg_loss:.6f}"
        ]
        for i, t in enumerate(txts):
            screen.blit(font.render(t, True, WHITE), (10, 10 + 25*i))

        # Display bottom instructions
        screen.blit(font.render("R: Reset Simulation ", True, (150, 150, 150)), (10, HEIGHT - 30))
        pygame.display.flip()

# =========================================================
if __name__ == "__main__":
    main()