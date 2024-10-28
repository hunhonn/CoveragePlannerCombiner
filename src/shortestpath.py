import cv2
import numpy as np
import heapq

# Paths to the image and waypoints files
WAYPOINTS_PATH1 = "../data/waypoints1.txt"
WAYPOINTS_PATH2 = "../data/waypoints2.txt"
OUTPUT_PATH = "../data/output_path.txt"
PARAMETER_FILE_PATH = "../config/params.config"

def load_parameters():
    global image_path, robot_width, robot_height, open_kernel_width, open_kernel_height
    global dilate_kernel_width, dilate_kernel_height, sweep_step, show_cells
    global mouse_select_start, start_x, start_y, subdivisions, manual_orientation

    # Load parameters from config file
    with open(PARAMETER_FILE_PATH, 'r') as f:
        for line in f:
            param = line.strip().split()
            if not param:
                continue
            
            if param[0] == "IMAGE_PATH":
                image_path = param[1]
            elif param[0] == "ROBOT_SIZE":
                robot_width, robot_height = map(int, param[1:3])
            elif param[0] == "MORPH_SIZE":
                open_kernel_width, open_kernel_height = map(int, param[1:3])
            elif param[0] == "OBSTACLE_INFLATION":
                dilate_kernel_width, dilate_kernel_height = map(int, param[1:3])
            elif param[0] == "SWEEP_STEP":
                sweep_step = int(param[1])
            elif param[0] == "SHOW_CELLS":
                show_cells = param[1].lower() == 'true'
            elif param[0] == "MOUSE_SELECT_START":
                mouse_select_start = param[1].lower() == 'true'
            elif param[0] == "START_POS":
                start_x, start_y = map(int, param[1:3])
            elif param[0] == "PATH_SUBDIVISION":
                subdivisions = int(param[1])
            elif param[0] == "MANUAL_ORIENTATION":
                manual_orientation = param[1].lower() == 'true'

def read_waypoints(file_path):
    waypoints = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split())
            waypoints.append((x, y))
    return waypoints

def get_navigatable_regions(img):
    if img is None:
        print("Error: Unable to load image.")
        return []
    
    white_points = cv2.findNonZero(img)
    return [(pt[0][0], pt[0][1]) for pt in white_points] if white_points is not None else []

def preprocess_image(img):
    print("Read map\nPre-Processing map image")

    # Convert to grayscale and binarize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_ = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

    # Apply morphological operations
    print("--Applying morphological operations onto image--")
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (robot_width, robot_height))
    img_ = cv2.morphologyEx(img_, cv2.MORPH_ERODE, erode_kernel)
    print("Erosion Kernel for robot size applied")

    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_width, open_kernel_height))
    img_ = cv2.morphologyEx(img_, cv2.MORPH_OPEN, open_kernel)
    print("Open Kernel applied")

    # Inflate obstacles
    print("--Inverting the image to apply dilation on black walls--")
    img_ = cv2.bitwise_not(img_)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_width, dilate_kernel_height))
    img_ = cv2.dilate(img_, dilation_kernel)
    img_ = cv2.bitwise_not(img_)
    print("Dilation applied to inflate the walls")

    return img_

def plot_image(img, waypoints1, waypoints2, path):
    if img is None:
        print("Error: Unable to load image.")
        return

    # Plot waypoints from both files
    for wp in range(len(waypoints1)):
        #cv2.circle(img, wp, 1, (0, 0, 255), -1)  # Red for waypoints1
        cv2.line(img, waypoints1[wp-1], waypoints1[wp], (0, 0, 255), 1)
    for wp in range(len(waypoints2)):
        #cv2.circle(img, wp, 1, (0, 255, 0), -1)  # Green for waypoints2
        cv2.line(img, waypoints2[wp-1], waypoints2[wp], (255, 0, 0), 1)

    # Draw lines for the path
    if path:
        for i in range(1, len(path)):
            cv2.line(img, path[i-1], path[i], (255, 0, 0), 1)  # Blue line for path

    # Display the combined image
    cv2.imshow("Image with Waypoints and Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def a_star_search(navigable_regions, start, goal):
    navigable_set = set(convert_to_tuples(navigable_regions))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

    open_list = []
    heapq.heappush(open_list, (0, start))
    
    g_cost = {start: 0}
    came_from = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if neighbor in navigable_set:
                tentative_g_cost = g_cost[current] + 1
                if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_cost, neighbor))
                    came_from[neighbor] = current

    return None

def convert_to_tuples(navigable_regions):
    return [(int(point[0]), int(point[1])) for point in navigable_regions]

def save_path_to_file(path, filename=OUTPUT_PATH):
    with open(filename, "w") as file:
        for point in path:
            file.write(f"{point[0]} {point[1]}\n")
    print(f"Path saved to {filename}")

if __name__ == "__main__":
    load_parameters()
    img = cv2.imread(image_path)
    processed_img = preprocess_image(img)

    waypoints1 = read_waypoints(WAYPOINTS_PATH1)
    waypoints2 = read_waypoints(WAYPOINTS_PATH2)
    navigatable_regions = get_navigatable_regions(processed_img)

    start = waypoints1[-1]
    goal = waypoints2[0]

    if start not in navigatable_regions or goal not in navigatable_regions:
        print("Error: Start or goal is not in navigable regions.")
    else:
        path = a_star_search(navigatable_regions, start, goal)
        if path:
            save_path_to_file(path)
            plot_image(img, waypoints1, waypoints2, path)
        else:
            print("No path found.")
