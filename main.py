'''
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

# === 單輪排程 ===
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
    total_rounds = 3

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
'''

import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import time

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

    # 預先計算距離矩陣（向量化操作）
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    distances = (distances + distances.T) / 2  # 確保對稱性
    t = distances / speed   # 飛行時間矩陣
    b = distances * energy_rate # 每段的能耗矩陣

    # 權重一開始全為 0
    weights = np.zeros(S + 1)
    weights[0] = 0  # 基地不巡邏

    return B, t, b, S, coordinates, weights


# === 優化的A*演算法 ===
def a_star_path(start, goal, t, h_func, b=None, battery=float('inf')):
    """
    優化的A*路徑演算法
    
    參數:
    - start: 起始節點
    - goal: 目標節點
    - t: 時間矩陣
    - h_func: 啟發式函數
    - b: 能耗矩陣（可選）
    - battery: 電池剩餘量（可選）
    """
    if start == goal:
        return [start]
        
    # 初始化開放集、關閉集和來源節點追蹤
    open_set = {start}
    closed_set = set()
    
    # g[n] 存儲從起點到節點n的成本
    g_cost = defaultdict(lambda: float('inf'))
    g_cost[start] = 0
    
    # f[n] = g[n] + h(n) 存儲總估計成本
    f_cost = defaultdict(lambda: float('inf'))
    f_cost[start] = h_func(start, goal)
    
    # 用於重建路徑
    came_from = {}
    
    # 優先隊列，存儲 (f_cost, node_id)
    queue = [(f_cost[start], start)]
    
    while queue:
        # 獲取f值最小的節點
        _, current = heapq.heappop(queue)
        
        if current not in open_set:
            continue
            
        if current == goal:
            # 重建路徑
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # 反轉路徑
            
        open_set.remove(current)
        closed_set.add(current)
        
        # 檢查所有相鄰節點
        for neighbor in range(len(t[current])):
            # 跳過已經評估過的節點或當前節點
            if neighbor in closed_set or neighbor == current:
                continue
                
            # 計算到鄰居的成本
            tentative_g = g_cost[current] + t[current][neighbor]
            
            # 檢查電池約束（如果提供）
            if b is not None and b[current][neighbor] > battery:
                continue
                
            # 如果找到更好的路徑
            if tentative_g < g_cost[neighbor]:
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g
                f_cost[neighbor] = tentative_g + h_func(neighbor, goal)
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                    heapq.heappush(queue, (f_cost[neighbor], neighbor))
    
    return None  # 找不到路徑


# === 改進的啟發式函數 ===
def heuristic(n1, n2, coordinates):
    x1, y1 = coordinates[n1]
    x2, y2 = coordinates[n2]
    return np.linalg.norm([x1 - x2, y1 - y2])


# === 計算能源效率路徑 ===
def get_energy_efficient_path(current_node, target_node, t, b, battery, coordinates):
    """計算考慮能源效率的最佳路徑"""
    
    # 定義考慮能源的啟發式函數
    def energy_heuristic(n1, n2):
        return heuristic(n1, n2, coordinates) * 1.5  # 略微高估以確保路徑有效
    
    # 使用A*找到路徑
    path = a_star_path(current_node, target_node, t, energy_heuristic, b, battery)
    return path


# === 改進的單輪排程 ===
def single_patrol_round(B, t, b, S, weights, num_drones, max_time, alpha=0.5, 
                        beta=0.3, require_all_visit=False, verbose=False, coordinates=None):
    """
    改進的單輪巡邏排程
    
    參數:
    - B: 電池容量
    - t: 時間矩陣
    - b: 能耗矩陣
    - S: 節點數（不含基地）
    - weights: 節點權重
    - num_drones: 無人機數量
    - max_time: 最大巡邏時間
    - alpha: 時間因子權重
    - beta: 能源效率權重
    - require_all_visit: 是否要求訪問所有節點
    - verbose: 是否輸出詳細信息
    - coordinates: 節點坐標
    """
    
    # 無人機狀態初始化
    drones = [{
        "battery": B,
        "current_node": 0,
        "visited_nodes": [0],
        "total_time": 0,
        "path_cache": {}  # 路徑快取
    } for _ in range(num_drones)]

    # 追蹤統計
    patrol_counts = np.zeros(S + 1)
    time_since_last_visit = np.zeros(S + 1)
    visited_once = set()
    
    # 前處理：計算最短能量路徑
    if verbose:
        print("預計算路徑中...")
        
    # 動態規劃表格
    max_steps = 100  
    step = 0
    
    start_time = time.time()
    
    while any(drone["total_time"] < max_time for drone in drones) and step < max_steps:
        step += 1
        time_since_last_visit += 1
        
        for i, drone in enumerate(drones):
            if drone["total_time"] >= max_time:
                continue
                
            current_node = drone["current_node"]
            current_battery = drone["battery"]
            
            # 如果電池電量低於一定閾值，優先考慮返回基地
            low_battery_threshold = 0.3 * B
            if current_battery < low_battery_threshold and current_node != 0:
                # 檢查是否有足夠電量返回基地
                if current_battery >= b[current_node][0] * 1.1:  # 加10%安全餘量
                    # 直接返回基地
                    drone["battery"] -= b[current_node][0]
                    drone["total_time"] += t[current_node][0]
                    drone["visited_nodes"].append(0)
                    drone["current_node"] = 0
                    drone["battery"] = B  # 充電
                    
                    if verbose:
                        print(f"[Step {step}] Drone {i+1} 返回基地充電，從節點 {current_node} → 0 | 電量: {B} | 時間: {drone['total_time']:.2f}")
                    
                    continue
            
            # 計算每個節點的優先級
            priorities = []
            reachable_nodes = []
            
            for node in range(S + 1):
                if node == current_node:
                    priorities.append(-1)
                    continue
                    
                # 檢查能源約束
                if b[current_node][node] > current_battery:
                    priorities.append(-1)
                    continue
                    
                # 計算優先級
                if require_all_visit and node not in visited_once and node != 0:
                    # 強制拜訪未訪問的節點
                    priority = 1000 + weights[node] - t[current_node][node]
                else:
                    # 正常優先級計算
                    time_factor = alpha * time_since_last_visit[node]
                    weight_factor = weights[node] / (t[current_node][node] + 1e-6)
                    energy_factor = -beta * (b[current_node][node] / B)  # 能源效率因子
                    
                    priority = weight_factor + time_factor + energy_factor
                
                priorities.append(priority)
                reachable_nodes.append(node)
            
            # 選擇目標節點
            if max(priorities) > -0.5:
                target_node = np.argmax(priorities)
                
                # 查看路徑快取
                cache_key = (current_node, target_node, int(current_battery))
                if cache_key in drone["path_cache"]:
                    path = drone["path_cache"][cache_key]
                else:
                    # 使用A*規劃考慮能源的路徑
                    path = get_energy_efficient_path(
                        current_node, target_node, t, b, current_battery, coordinates
                    )
                    # 儲存到快取
                    drone["path_cache"][cache_key] = path
                
                if path is None:
                    if verbose:
                        print(f"[Step {step}] Drone {i+1} 無法找到從 {current_node} 到 {target_node} 的有效路徑")
                    continue
                
                # 執行路徑
                for j in range(1, len(path)):
                    step_from = path[j-1]
                    step_to = path[j]
                    
                    # 更新無人機狀態
                    drone["battery"] -= b[step_from][step_to]
                    drone["total_time"] += t[step_from][step_to]
                    
                    # 只有在抵達新節點時計數
                    if step_to != current_node:
                        drone["visited_nodes"].append(step_to)
                        patrol_counts[step_to] += 1
                        time_since_last_visit[step_to] = 0
                        visited_once.add(step_to)
                    
                    drone["current_node"] = step_to
                    
                    # 檢查是否需要中斷
                    if drone["battery"] <= 0 or drone["total_time"] >= max_time:
                        break
                
                if verbose:
                    print(f"[Step {step}] Drone {i+1} 從 {current_node} 飛向 {drone['current_node']} | 電量: {drone['battery']:.2f} | 時間: {drone['total_time']:.2f}")
            
            elif current_node != 0:
                # 回基地充電
                if verbose:
                    print(f"[Step {step}] Drone {i+1} 沒有可達目標，準備返回基地")
                
                # 檢查是否有足夠電量返回基地
                if current_battery >= b[current_node][0]:
                    drone["battery"] -= b[current_node][0]
                    drone["total_time"] += t[current_node][0]
                    drone["visited_nodes"].append(0)
                    drone["current_node"] = 0
                    drone["battery"] = B  # 充電
                    
                    if verbose:
                        print(f"[Step {step}] Drone {i+1} 返回基地充電，從節點 {current_node} → 0 | 電量: {B} | 時間: {drone['total_time']:.2f}")
                else:
                    if verbose:
                        print(f"[Step {step}] Drone {i+1} 在節點 {current_node} 電量耗盡，無法返回基地")
                    # 電量耗盡，強制結束巡邏
                    drone["total_time"] = max_time
            
            else:
                if verbose:
                    print(f"[Step {step}] Drone {i+1} 在基地待命")
    
    execution_time = time.time() - start_time
    if verbose:
        print(f"排程計算用時: {execution_time:.4f} 秒")
        print("\n=== 巡邏摘要 ===")
        for i, drone in enumerate(drones):
            total_nodes = len(drone["visited_nodes"])
            unique_nodes = len(set(drone["visited_nodes"]))
            print(f"無人機 {i+1}: 總訪問節點數: {total_nodes}, 訪問的不同節點數: {unique_nodes}, 總時間: {drone['total_time']:.2f}")
    
    return drones, patrol_counts, time_since_last_visit


# === 智能權重更新 ===
def update_weights_intelligently(S, patrol_counts, time_since_last_visit, previous_weights=None):
    """
    根據巡邏歷史智能地更新節點權重
    
    參數:
    - S: 節點數（不含基地）
    - patrol_counts: 當前巡邏計數
    - time_since_last_visit: 上次訪問後經過的時間
    - previous_weights: 先前的權重
    """
    # 基本權重 - 未被訪問的節點獲得較高權重
    weights = np.ones(S + 1) * 3.0
    
    # 加入隨機性
    weights += np.random.uniform(-0.5, 0.5, S + 1)
    
    # 訪問頻率調整 - 訪問越少，權重越高
    if np.max(patrol_counts) > 0:
        frequency_factor = 1.0 - patrol_counts / np.max(patrol_counts)
        weights += frequency_factor * 2
    
    # 時間因素 - 長時間未訪問的節點權重增加
    if np.max(time_since_last_visit) > 0:
        time_factor = time_since_last_visit / np.max(time_since_last_visit)
        weights += time_factor * 2
    
    # 保留先前權重的部分影響（如果有）
    if previous_weights is not None:
        weights = 0.7 * weights + 0.3 * previous_weights
    
    # 基地台權重保持為0
    weights[0] = 0
    
    return weights


# === 視覺化 ===
def visualize_individual_drone_schedules(coordinates, drones_round, patrol_counts, weights, time_since_last_visit, round_num=None):
    """改進的無人機巡邏路徑視覺化"""
    
    # 提高圖像質量
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    for idx, drone in enumerate(drones_round):
        # 創建圖像和軸
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 繪製網格線，使圖像更清晰
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 計算用於顏色映射的最大訪問次數（避免除以零）
        max_patrol_count = max(1, np.max(patrol_counts))
        
        # 繪製節點
        for i, (x, y) in enumerate(coordinates):
            if i == 0:
                ax.scatter(x, y, color='red', s=100, label='基地台 (0)', zorder=3)
                ax.text(
                    x + 0.1, y + 0.1,
                    f'Node {i} (基地)\nCoordinate: ({x:.1f}, {y:.1f})\nWeight: {weights[i]:.1f}\n訪問次數: {int(patrol_counts[i])}\n上次訪問: {int(time_since_last_visit[i])}',
                    fontsize=8, color='black', zorder=4
                )
            else:
                # 使用熱力圖顏色編碼訪問頻率
                color_value = max(0, min(patrol_counts[i] / max_patrol_count, 1))
                node_color = plt.cm.YlOrRd(color_value)
                
                ax.scatter(x, y, color=node_color, s=80, zorder=3, edgecolor='black')
                ax.text(
                    x + 0.1, y + 0.1,
                    f'Node {i}\nCoordinate: ({x:.1f}, {y:.1f})\nWeight: {weights[i]:.2f}\n訪問次數: {int(patrol_counts[i])}\n上次訪問: {int(time_since_last_visit[i])}',
                    fontsize=8, color='black', zorder=4, 
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2')
                )

        # 繪製無人機的路徑
        path = drone["visited_nodes"]
        
        # 根據路徑順序使用漸變顏色
        cmap = plt.cm.viridis
        colors = [cmap(float(i)/(len(path)-1)) for i in range(len(path)-1)] if len(path) > 1 else []
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            x_start, y_start = coordinates[start]
            x_end, y_end = coordinates[end]
            
            # 使用漸變顏色
            ax.plot(
                [x_start, x_end],
                [y_start, y_end],
                '-', 
                color=colors[i],
                alpha=0.8,
                linewidth=1.5
            )

            # 繪製箭頭，顏色基于路徑順序
            dx = x_end - x_start
            dy = y_end - y_start
            arrow_start_x = x_start + 0.2 * dx
            arrow_start_y = y_start + 0.2 * dy
            ax.arrow(
                arrow_start_x, arrow_start_y,
                dx * 0.6, dy * 0.6,
                head_width=0.1, head_length=0.1,
                fc=colors[i], ec=colors[i],
                length_includes_head=True,
                alpha=0.9,
                zorder=3
            )

            # 加上步數文字（放在箭頭中間偏上）
            step_number = i + 1
            text_x = x_start + 0.5 * dx
            text_y = y_start + 0.5 * dy + 0.1
            ax.text(
                text_x, text_y,
                str(step_number),
                fontsize=8, color='black',
                weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'),
                zorder=5
            )

        # 使用藍色高亮顯示起點和終點
        if len(path) > 0:
            start_node = path[0]
            end_node = path[-1]
            x_start, y_start = coordinates[start_node]
            x_end, y_end = coordinates[end_node]
            ax.scatter(x_start, y_start, color='blue', s=100, zorder=4, marker='o')
            ax.scatter(x_end, y_end, color='green', s=100, zorder=4, marker='*')
        
        # 添加信息文本框
        info_text = f"無人機 {idx + 1} 統計:\n"
        info_text += f"總路徑長度: {len(path)-1} 步\n"
        info_text += f"訪問的不同節點數: {len(set(path))}\n"
        info_text += f"總巡邏時間: {drone['total_time']:.2f}"
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, 
                    bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))
        
        round_title = f" (第 {round_num} 輪)" if round_num is not None else ""
        plt.title(f"無人機 {idx + 1} 巡邏路徑{round_title}", fontsize=14, fontweight='bold')
        plt.xlabel("X 座標", fontsize=12)
        plt.ylabel("Y 座標", fontsize=12)
        
        # 添加顏色條，表示訪問頻率 - 修復方法
        norm = plt.Normalize(0, max_patrol_count)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
        sm.set_array([])  # 需要設置一個空數組
        cbar = plt.colorbar(sm, ax=ax)  # 明確指定ax參數
        cbar.set_label('訪問頻率', fontsize=10)
        
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


# === 改進的任務分析 ===
def analyze_mission_paths(drones):
    results = []
    
    for drone_idx, drone in enumerate(drones):
        drone_results = {"drone_idx": drone_idx, "missions": []}
        path = drone["visited_nodes"]
        current_mission = []
        mission_count = 1
        
        for node in path:
            current_mission.append(node)
            # 當到達基地台(0)且不是路徑的起點時，表示一個任務完成
            if node == 0 and len(current_mission) > 1:
                # 計算這次任務訪問的獨特節點
                unique_nodes = set(current_mission[1:-1])  # 不包括起點和終點的0
                
                mission_info = {
                    "mission_num": mission_count,
                    "path": current_mission.copy(),
                    "unique_nodes": sorted(list(unique_nodes)),
                    "length": len(current_mission)
                }
                drone_results["missions"].append(mission_info)
                
                # 重置為新任務
                current_mission = [0]
                mission_count += 1
            
        # 處理最後一個可能未完成的任務
        if len(current_mission) > 1:
            unique_nodes = set(current_mission[1:])
            mission_info = {
                "mission_num": mission_count,
                "path": current_mission.copy(),
                "unique_nodes": sorted(list(unique_nodes)),
                "length": len(current_mission),
                "incomplete": True
            }
            drone_results["missions"].append(mission_info)
        
        results.append(drone_results)
    
    return results


# === 改進的任務分析輸出 ===
def print_mission_analysis(drones):
    print("\n=== 巡邏任務分析 ===")
    
    analysis_results = analyze_mission_paths(drones)
    
    for drone_result in analysis_results:
        drone_idx = drone_result["drone_idx"]
        print(f"\n無人機 {drone_idx + 1} 任務:")
        
        for mission in drone_result["missions"]:
            incomplete_tag = " (未完成)" if mission.get("incomplete", False) else ""
            print(f"任務 {mission['mission_num']}{incomplete_tag}:")
            print(f"  完整路徑: {' -> '.join(map(str, mission['path']))}")
            print(f"  訪問的不同節點: {mission['unique_nodes']}")
            print(f"  路徑長度: {mission['length']} 節點\n")
    
    # 添加總體統計
    print("=== 總體統計 ===")
    for drone_idx, drone in enumerate(drones):
        path = drone["visited_nodes"]
        # 計算完成的任務數
        missions = sum(1 for i in range(1, len(path)) if path[i] == 0 and path[i-1] != 0)
        # 計算訪問的不同節點數
        unique_nodes = len(set(path))
        
        print(f"\n無人機 {drone_idx + 1}:")
        print(f"  總任務數: {missions}")
        print(f"  總訪問節點數: {len(path)}")
        print(f"  訪問的不同節點數: {unique_nodes}")
        print(f"  平均每任務節點數: {(len(path)-1) / max(1, missions):.2f}")
        print(f"  總巡邏時間: {drone['total_time']:.2f}")


# === 巡邏效率評估 ===
def evaluate_patrol_efficiency(drones, S, max_time):
    """
    評估巡邏效率
    
    返回:
    - coverage_ratio: 覆蓋率（0-1）
    - time_efficiency: 時間效率（0-1）
    - visit_balance: 訪問平衡性（0-1）
    """
    # 計算所有節點的訪問次數
    all_visits = {}
    for i in range(1, S+1):  # 跳過基地
        all_visits[i] = 0
    
    total_time_used = 0
    
    for drone in drones:
        # 統計每個節點訪問次數
        for node in drone["visited_nodes"]:
            if node > 0:  # 忽略基地
                all_visits[node] = all_visits.get(node, 0) + 1
        
        total_time_used += drone["total_time"]
    
    # 計算覆蓋率
    covered_nodes = sum(1 for count in all_visits.values() if count > 0)
    coverage_ratio = covered_nodes / S
    
    # 計算時間效率
    theoretical_max_time = len(drones) * max_time
    time_efficiency = min(1.0, total_time_used / theoretical_max_time)
    
    # 計算訪問平衡性
    if covered_nodes > 0:
        visit_counts = np.array(list(all_visits.values()))
        visit_counts = visit_counts[visit_counts > 0]  # 只考慮訪問過的節點
        visit_balance = 1.0 - (np.std(visit_counts) / (np.mean(visit_counts) + 1e-6))
        visit_balance = max(0, min(1, visit_balance))  # 限制在0-1之間
    else:
        visit_balance = 0
    
    return coverage_ratio, time_efficiency, visit_balance


# === 主流程 ===
if __name__ == "__main__":
    # 初始參數
    B, t, b, S, coordinates, weights = initialize_parameters(grid_size=5)

    num_drones = 1  # 無人機數量
    max_time = 100  # 最大巡邏時間
    total_rounds = 3  # 總巡邏輪數
    
    print(f"模擬 {num_drones} 台無人機在 {S+1} 個節點上的巡邏")
    print(f"電池容量: {B}, 最大巡邏時間: {max_time}, 總輪數: {total_rounds}")
    
    # 總巡邏次數統計
    cumulative_patrol_counts = np.zeros(S + 1)
    previous_weights = None
    
    total_start_time = time.time()
    
    for round_num in range(1, total_rounds + 1):
        print(f"\n=== 第 {round_num} 輪 ===")
        round_start_time = time.time()
        
        # 每輪強制完整拜訪所有節點
        drones_round, patrol_counts, time_since_last_visit = single_patrol_round(
            B, t, b, S, weights, num_drones, max_time,
            require_all_visit=(round_num == 1),  # 第一輪強制拜訪所有節點
            verbose=False,
            coordinates=coordinates,
            beta=0.2 + (round_num * 0.1)  # 隨著輪數增加，提高能源效率的重要性
        )
        
        round_time = time.time() - round_start_time
        
        # 更新總巡邏統計
        cumulative_patrol_counts += patrol_counts
        
        # 評估巡邏效率
        coverage, time_eff, balance = evaluate_patrol_efficiency(drones_round, S, max_time)
        
        print(f"第 {round_num} 輪計算時間: {round_time:.2f} 秒")
        print(f"覆蓋率: {coverage:.2f}, 時間效率: {time_eff:.2f}, 平衡性: {balance:.2f}")
        
        # 視覺化
        visualize_individual_drone_schedules(coordinates, drones_round, cumulative_patrol_counts, weights, time_since_last_visit, round_num)
        print_mission_analysis(drones_round)
        
        # 更新權重（下一輪使用）
        previous_weights = weights.copy()
        weights = update_weights_intelligently(S, cumulative_patrol_counts, time_since_last_visit, previous_weights)
        print("更新後節點權重：", np.round(weights, 2))
    
    total_time = time.time() - total_start_time
    print(f"\n總計算時間: {total_time:.2f} 秒")
    print("模擬完成！")