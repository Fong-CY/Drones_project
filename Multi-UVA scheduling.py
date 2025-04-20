import numpy as np
import matplotlib.pyplot as plt

# === 初始化參數 ===
def initialize_parameters():
    B = 100  # 無人機電池容量
    speed = 1.0  # 無人機速度（km/min）
    max_distance = 3  # 節點之間的最大距離
    energy_rate = 2  # 每公里能耗
    S = 20  # 傳感器節點數量（不包括基站）

    # 隨機生成節點的座標
    coordinates = np.random.rand(S + 1, 2) * max_distance  # 節點座標 (x, y)
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)  # 計算歐幾里得距離矩陣
    distances = (distances + distances.T) / 2
    # 計算飛行時間和能耗
    t = distances / speed  # 飛行時間矩陣
    b = distances * energy_rate  # 每段的能耗矩陣

    # 隨機生成節點的權重（模擬節點重要性）
    weights = np.random.uniform(1, 5, S + 1)
    weights[0] = 0  # 基站無需訪問

    return B, t, b, S, coordinates, weights

# === 多無人機巡邏排程演算法 ===
def multi_drone_patrol_schedule(B, t, b, S, weights, num_drones=2, max_time=100, alpha=0.5):
    drones = [{"battery": B, "current_node": 0, "visited_nodes": [0], "total_time": 0} for _ in range(num_drones)]
    patrol_counts = np.zeros(S + 1)  # 每個節點的巡邏次數
    total_score = 0  # 巡邏總效益
    time_since_last_visit = np.zeros(S + 1)  # 每個節點的未訪問時間

    while any(drone["total_time"] < max_time for drone in drones):
        # 更新每個節點的未訪問時間
        time_since_last_visit += 1

        for drone in drones:
            if drone["total_time"] >= max_time:
                continue

            # 計算每個節點的優先級
            current_node = drone["current_node"]
            priorities = [
                (weights[node] / t[current_node][node] + alpha * time_since_last_visit[node])
                if current_node != node and b[current_node][node] <= drone["battery"] else -1
                for node in range(S + 1)
            ]

            # 找到優先級最高的節點
            target_node = np.argmax(priorities)

            # 如果有可訪問的節點，進行訪問
            if priorities[target_node] > 0:
                drone["battery"] -= b[current_node][target_node]
                drone["total_time"] += t[current_node][target_node]
                drone["visited_nodes"].append(target_node)
                patrol_counts[target_node] += 1  # 增加巡邏次數
                total_score += weights[target_node]  # 增加總得分
                time_since_last_visit[target_node] = 0  # 重置未訪問時間
                drone["current_node"] = target_node
            else:
                # 電量不足，回基站充電
                if current_node != 0:
                    drone["total_time"] += t[current_node][0]
                    drone["visited_nodes"].append(0)
                    drone["battery"] = B
                    drone["current_node"] = 0

    return drones, patrol_counts, total_score, time_since_last_visit

# === 資料視覺化 ===
# === 單獨可視化每台無人機的巡邏路徑 ===
def visualize_individual_drone_schedules(coordinates, drones, patrol_counts, weights, time_since_last_visit):
    for idx, drone in enumerate(drones):
        plt.figure(figsize=(8, 8))
        # 繪製節點
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
            plt.plot(
                [coordinates[start][0], coordinates[end][0]],
                [coordinates[start][1], coordinates[end][1]],
                'g-' if start == 0 or end == 0 else 'b-', alpha=0.7, linewidth = 1
            )

        plt.title(f"Drone {idx + 1} Patrol Schedule", fontsize=16)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.legend()
        plt.grid(True)
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

# === 主程式執行 ===
if __name__ == "__main__":
    # 初始化參數
    B, t, b, S, coordinates, weights = initialize_parameters()

    # 執行多無人機排程
    num_drones = 3  # 設置無人機數量
    drones, patrol_counts, total_score, time_since_last_visit = multi_drone_patrol_schedule(
        B, t, b, S, weights, num_drones=num_drones, max_time=100, alpha=0.5
    )

    # 輸出結果
    for idx, drone in enumerate(drones):
        print(f"Drone {idx + 1}:")
        print("  Visit Path:", drone["visited_nodes"])
        print("  Total Time:", drone["total_time"])
        print()

    print("Patrol Counts:", patrol_counts)
    print("Node Weights:", weights)
    print("Time Since Last Visit:", time_since_last_visit)
    print("Total Score:", total_score)

    # 為每台無人機單獨視覺化
    visualize_individual_drone_schedules(coordinates, drones, patrol_counts, weights, time_since_last_visit)

    # 分析任務路徑
    print_mission_analysis(drones)
