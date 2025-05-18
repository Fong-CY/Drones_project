import numpy as np
import matplotlib.pyplot as plt
import heapq

# === 初始化參數 ===
def initialize_parameters(grid_size=5):
    B = 100
    speed = 1.0
    energy_rate = 2

    # 建立 grid_size x grid_size 的節點 (共 grid_size^2 個)
    coordinates = []
    for x in range(grid_size):
        for y in range(grid_size):
            if x == 0 and y == 0:
                continue  # 跳過 (0,0)，避免重複
            coordinates.append([x, y])
    coordinates = np.array([[0, 0]] + coordinates)  # 把基地台 (0,0) 放在最前面

    S = len(coordinates) - 1  # 節點數（不含基地）

    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    distances = (distances + distances.T) / 2
    t = distances / speed   # 飛行時間矩陣
    b = distances * energy_rate # 每段的能耗矩陣

    # 權重一開始全為 0
    weights = np.zeros(S + 1)
    weights[0] = 0  # 基地不巡邏

    return B, t, b, S, coordinates, weights


# === A* ===
def a_star_path(start, goal, t, h_func):
    open_set = [(h_func(start, goal), 0, start, [start])]
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        if current in visited:
            continue
        visited.add(current)

        for neighbor in range(len(t)):
            if neighbor in visited or neighbor == current:
                continue
            cost = t[current][neighbor]
            heapq.heappush(open_set, (
                g + cost + h_func(neighbor, goal),
                g + cost,
                neighbor,
                path + [neighbor]
            ))

    return None  # 找不到路徑

# === Heuristic(歐式距離) ===
def heuristic(n1, n2, coordinates):
    x1, y1 = coordinates[n1]
    x2, y2 = coordinates[n2]
    return np.linalg.norm([x1 - x2, y1 - y2])

# === 單輪排程（會被執行兩次）===
def single_patrol_round(B, t, b, S, weights, num_drones, max_time, alpha=0.5, require_all_visit=False, verbose=False, coordinates=None):
    drones = [{
        "battery": B,
        "current_node": 0,
        "visited_nodes": [0],
        "total_time": 0
    } for _ in range(num_drones)]

    patrol_counts = np.zeros(S + 1)
    time_since_last_visit = np.zeros(S + 1)
    visited_once = set()

    max_steps = 100  # 可以調整
    step = 0

    while any(drone["total_time"] < max_time for drone in drones) and step < max_steps:
        step += 1
        time_since_last_visit += 1

        for i, drone in enumerate(drones):
            if drone["total_time"] >= max_time:
                continue

            current_node = drone["current_node"]

            priorities = []
            for node in range(S + 1):
                if node == current_node:
                    priorities.append(-1)
                    continue
                if b[current_node][node] > drone["battery"]:
                    priorities.append(-1)
                    continue

                if require_all_visit and node not in visited_once:
                    priority = 10000 - t[current_node][node]  # 強迫拜訪未拜訪點
                else:
                    priority = (weights[node] / (t[current_node][node] + 1e-6)) + alpha * time_since_last_visit[node]

                priorities.append(priority)

            target_node = np.argmax(priorities)

            #if priorities[target_node] > -0.5:  # 只要有可飛目標就出發
            #    drone["battery"] -= b[current_node][target_node]
            #    drone["total_time"] += t[current_node][target_node]
            #    drone["visited_nodes"].append(target_node)
            #    patrol_counts[target_node] += 1
            #    time_since_last_visit[target_node] = 0
            #    visited_once.add(target_node)
            #    drone["current_node"] = target_node

            #    if verbose:
            #        print(f"[Step {step}] Drone {i+1} flew {current_node} → {target_node} | Battery: {drone['battery']} | Time: {drone['total_time']}")
            #else:
            #    if current_node != 0:
            #        # 回基地充電
            #        drone["total_time"] += t[current_node][0]
            #        drone["visited_nodes"].append(0)
            #        drone["battery"] = B
            #        drone["current_node"] = 0

            #        if verbose:
            #            print(f"[Step {step}] Drone {i+1} returned to base from {current_node} → 0 | Battery recharged | Time: {drone['total_time']}")
            #    else:
            #        if verbose:
            #            print(f"[Step {step}] Drone {i+1} is idle at base.")

            if priorities[target_node] > -0.5:
                # 使用 A* 規劃完整路徑
                path = a_star_path(current_node, target_node, t, lambda a, b: heuristic(a, b, coordinates))

                if path is None:
                    continue  # 找不到合法路徑

                for step_node in path[1:]:  # 忽略目前位置
                    drone["battery"] -= b[drone["current_node"]][step_node]
                    drone["total_time"] += t[drone["current_node"]][step_node]
                    drone["visited_nodes"].append(step_node)
                    patrol_counts[step_node] += 1
                    time_since_last_visit[step_node] = 0
                    visited_once.add(step_node)
                    drone["current_node"] = step_node

                    if drone["battery"] <= 0 or drone["total_time"] >= max_time:
                        break


    if verbose:
        print("\n=== Patrol Summary ===")
        for i, drone in enumerate(drones):
            print(f"Drone {i+1} Path: {drone['visited_nodes']}, Time: {drone['total_time']}")

    return drones, patrol_counts, time_since_last_visit

# === 權重更新 ===
def update_weights_randomly(S):
    weights = np.random.uniform(1, 5, S + 1)
    weights[0] = 0  # 基地台無權重
    return weights

# === 視覺化 ===

# === 單獨可視化每台無人機的巡邏路徑 ===
def visualize_individual_drone_schedules(coordinates, drones_round, patrol_counts, weights, time_since_last_visit):
    for idx, drone in enumerate(drones_round):
        # 繪製節點
        plt.figure(figsize=(8, 8))
        for i, (x, y) in enumerate(coordinates):
            if i == 0:
                plt.scatter(x, y, color='red', label='Base Station (0)', zorder=3)
                plt.text(
                    x, y,
                    f'Node {i} (Base)\nCoord: ({x:.1f}, {y:.1f})\nWeight: {weights[i]:.1f}\nVisits: {int(patrol_counts[i])}\nLast: {int(time_since_last_visit[i])}',
                    fontsize=12, color='black', zorder=4
                )
            else:
                plt.scatter(x, y, color='blue', zorder=3)
                plt.text(
                    x, y,
                    f'Node {i}\nCoord: ({x:.1f}, {y:.1f})\nWeight: {weights[i]:.1f}\nVisits: {int(patrol_counts[i])}\nLast: {int(time_since_last_visit[i])}',
                    fontsize=10, color='black', zorder=4
                )

        # 繪製無人機的路徑
        path = drone["visited_nodes"]
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            x_start, y_start = coordinates[start]
            x_end, y_end = coordinates[end]
            plt.plot(
                [x_start, x_end],
                [y_start, y_end],
                'g-' if start == 0 or end == 0 else 'b-',
                alpha=0.7,
                linewidth=1
            )

            # 繪製箭頭
            dx = x_end - x_start
            dy = y_end - y_start
            arrow_start_x = x_start + 0.2 * dx
            arrow_start_y = y_start + 0.2 * dy
            arrow_color = 'green' if start == 0 or end == 0 else 'blue'
            plt.arrow(
                arrow_start_x, arrow_start_y,
                dx * 0.8, dy * 0.8,
                head_width=0.08, head_length=0.08,
                fc=arrow_color, ec=arrow_color,
                length_includes_head=True,
                alpha=0.9,
                zorder=3
            )

            # 加上步數文字（放在箭頭中間偏上）
            step_number = i + 1
            text_x = x_start + 0.5 * dx
            text_y = y_start + 0.5 * dy + 0.05
            plt.text(
                text_x, text_y,
                str(step_number),
                fontsize=9, color='black',
                zorder=5
            )

        plt.title(f"Drone {idx + 1} Patrol Schedule", fontsize=16)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

def analyze_mission_paths(drones):
    for drone_idx, drone in enumerate(drones):
        print(f"\nDrone {drone_idx + 1} Missions:")
        path = drone["visited_nodes"]
        current_mission = []
        mission_count = 1
        
        for node in path:
            current_mission.append(node)
            # 當到達基地台(0)且不是路徑的起點時，表示一個任務完成
            if node == 0 and len(current_mission) > 1:
                # 計算這次任務訪問的獨特節點
                unique_nodes = set(current_mission[1:-1])  # 不包括起點和終點的0
                print(f"Mission {mission_count}:")
                print(f"  Full path: {' -> '.join(map(str, current_mission))}")
                print(f"  Unique nodes visited: {sorted(unique_nodes)}")
                print(f"  Path length: {len(current_mission)} nodes\n")
                
                # 重置為新任務
                current_mission = [0]
                mission_count += 1
            
        # 處理最後一個可能未完成的任務
        if len(current_mission) > 1:
            unique_nodes = set(current_mission[1:])
            print(f"Final Mission (might be incomplete):")
            print(f"  Full path: {' -> '.join(map(str, current_mission))}")
            print(f"  Unique nodes visited: {sorted(unique_nodes)}")
            print(f"  Path length: {len(current_mission)} nodes\n")
            
# 使用示例
def print_mission_analysis(drones):
    print("\n=== Mission Path Analysis ===")
    analyze_mission_paths(drones)
    
    # 添加總體統計
    print("=== Overall Statistics ===")
    for drone_idx, drone in enumerate(drones):
        path = drone["visited_nodes"]
        missions = sum(1 for i in range(1, len(path)) if path[i] == 0)
        print(f"\nDrone {drone_idx + 1}:")
        print(f"  Total missions: {missions}")
        print(f"  Total nodes visited: {len(path)}")
        print(f"  Average nodes per mission: {(len(path)-1) / max(1, missions):.2f}")

# === 主流程 ===
if __name__ == "__main__":
    # 初始參數
    B, t, b, S, coordinates, weights = initialize_parameters(grid_size=5)

    num_drones = 1
    max_time = 100
    total_rounds = 5

    #print("=== 第一輪（探索階段） ===")
    #drones_round1, patrol_counts1, time_since_last_visit1 = single_patrol_round(
    #    B, t, b, S, weights, num_drones, max_time, require_all_visit=True
    #)

    # 第一輪可視化與分析
    #visualize_individual_drone_schedules(coordinates, drones_round1, patrol_counts1, weights, time_since_last_visit1)
    #print_mission_analysis(drones_round1)

    # 更新節點權重
    #weights = update_weights_randomly(S)
    #print("\n更新後節點權重：", weights)

    #print("\n=== 第二輪（依照權重優先巡邏） ===")
    #drones_round2, patrol_counts2, time_since_last_visit2 = single_patrol_round(
    #    B, t, b, S, weights, num_drones, max_time, require_all_visit=True
    #)

    # 合併巡邏統計
    #total_patrols = patrol_counts1 + patrol_counts2

    # 第二輪可視化與分析
    #visualize_individual_drone_schedules(coordinates, drones_round2, total_patrols, weights, time_since_last_visit2)
    #print_mission_analysis(drones_round2)
    
    # 總巡邏次數統計
    cumulative_patrol_counts = np.zeros(S + 1)

    for round_num in range(1, total_rounds + 1):
        print(f"\n=== 第 {round_num} 輪 ===")

        # 每輪強制完整拜訪所有節點
        drones_round, patrol_counts, time_since_last_visit = single_patrol_round(
            B, t, b, S, weights, num_drones, max_time,
            require_all_visit=True,
            verbose=False,
            coordinates=coordinates
        )

        # 更新總巡邏統計
        cumulative_patrol_counts += patrol_counts

        # 視覺化
        visualize_individual_drone_schedules(coordinates, drones_round, cumulative_patrol_counts, weights, time_since_last_visit)
        print_mission_analysis(drones_round)

        # 更新權重（下一輪使用）
        weights = update_weights_randomly(S)
        print("更新後節點權重：", weights)