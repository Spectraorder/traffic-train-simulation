import pygame
import sys
import random
import time
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Constants for the display
WINDOW_WIDTH, WINDOW_HEIGHT = 620, 620
GRID_SIZE = 3
GRID_SCALE = 0.8  # 80% of the screen size

# Calculate the grid dimensions based on scale
CELL_SIZE = int((WINDOW_WIDTH * GRID_SCALE) // GRID_SIZE)
GRID_OFFSET = (WINDOW_WIDTH - (CELL_SIZE * GRID_SIZE)) // 2  # Offset to center the grid

# Size of the car
CAR_SIZE = 10  # Size of the car to be smaller than the cell
CAR_SPEED = 1  # Pixels per frame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)

# Font settings
pygame.font.init()  # Initialize the font module
font = pygame.font.SysFont(None, 24)  # Create a font object from the system fonts

# Set up the display
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Grid Simulation with Intersection-Based Turning")

# Define global variables for data logging
traffic_data = []
start_logging_time = None
logging_active = True
class TrafficLight:
    def __init__(self, change_interval=1):
        self.current_light = 0  # Start with 'left'
        self.num_stages = 4  # Total directions
        self.change_interval = change_interval
        self.last_change_time = time.time()
        self.force_red = False  # Add a flag to force the light red

    def update(self):
        current_time = time.time()
        if current_time - self.last_change_time >= self.change_interval and not self.force_red:
            self.current_light = (self.current_light + 1) % self.num_stages
            self.last_change_time = current_time

    def is_green(self, direction):
        if self.force_red:
            return False
        direction_map = {'left': 0, 'down': 1, 'right': 2, 'up': 3}
        return direction_map[direction] == self.current_light

    def set_force_red(self, state):
        self.force_red = state


traffic_lights = {(i, j): TrafficLight() for i in range(1, GRID_SIZE) for j in range(1, GRID_SIZE)}


# Car data structures
def generate_car():
    """Generates a car with a random starting position and destination."""
    start_line_choice = random.choice(['horizontal', 'vertical'])
    if start_line_choice == 'horizontal':
        y = random.randint(0, GRID_SIZE - 1)
        start_pixel_x = random.randint(GRID_OFFSET, WINDOW_WIDTH - GRID_OFFSET - CAR_SIZE)
        start_pixel_y = GRID_OFFSET + y * CELL_SIZE - CAR_SIZE // 2
    else:
        x = random.randint(0, GRID_SIZE - 1)
        start_pixel_y = random.randint(GRID_OFFSET, WINDOW_HEIGHT - GRID_OFFSET - CAR_SIZE)
        start_pixel_x = GRID_OFFSET + x * CELL_SIZE - CAR_SIZE // 2

    end_line_choice = random.choice(['horizontal', 'vertical'])
    if end_line_choice == 'horizontal':
        y = random.randint(0, GRID_SIZE - 1)
        destination_pixel_x = random.randint(GRID_OFFSET, WINDOW_WIDTH - GRID_OFFSET - CAR_SIZE)
        destination_pixel_y = GRID_OFFSET + y * CELL_SIZE - CAR_SIZE // 2
    else:
        x = random.randint(0, GRID_SIZE - 1)
        destination_pixel_y = random.randint(GRID_OFFSET, WINDOW_HEIGHT - GRID_OFFSET - CAR_SIZE)
        destination_pixel_x = GRID_OFFSET + x * CELL_SIZE - CAR_SIZE // 2

    car_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    start_time = random.uniform(0, 4)  # Random start time between 0 and 4 seconds

    return {
        'pixel_position': [start_pixel_x, start_pixel_y],
        'destination': [destination_pixel_x, destination_pixel_y],
        'color': car_color,
        'start_time': start_time,
        'active': False,
        'move_stage': 'vertical' if start_line_choice == 'vertical' else 'horizontal'
    }


num_cars = 5
cars = [generate_car() for _ in range(num_cars)]

# Define the train's properties
train = {
    'position': [GRID_OFFSET + CELL_SIZE, GRID_OFFSET + CELL_SIZE],
    'size': (CAR_SIZE, CAR_SIZE * 3),  # Length and width of the train
    'color': (100, 100, 200),  # Distinguishable color
    'speed': 2,  # Speed of the train
    'direction': 'up',  # Initial direction
    'cycle': ['right', 'down', 'left', 'up']  # Cycling directions
}


def draw_grid():
    """Draws a 3x3 grid on the pygame window, scaled and centered."""
    for x in range(GRID_OFFSET, WINDOW_WIDTH - GRID_OFFSET + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, GRID_OFFSET), (x, WINDOW_HEIGHT - GRID_OFFSET))
    for y in range(GRID_OFFSET, WINDOW_HEIGHT - GRID_OFFSET + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (GRID_OFFSET, y), (WINDOW_WIDTH - GRID_OFFSET, y))


def move_car(car):
    """Moves the car towards its destination by evaluating the optimal path at every intersection,
       and prevents the car from getting stuck by ensuring there's always a direction to move towards."""
    # Calculate grid position
    current_x, current_y = car['pixel_position']
    dest_x, dest_y = car['destination']
    grid_x = (current_x - GRID_OFFSET + CAR_SIZE // 2) // CELL_SIZE
    grid_y = (current_y - GRID_OFFSET + CAR_SIZE // 2) // CELL_SIZE

    # Calculate if the current position is an intersection
    is_at_intersection = (current_x - GRID_OFFSET + CAR_SIZE // 2) % CELL_SIZE == 0 and \
                         (current_y - GRID_OFFSET + CAR_SIZE // 2) % CELL_SIZE == 0

    # Distance to destination on each axis
    distance_x = abs(dest_x - current_x)
    distance_y = abs(dest_y - current_y)

    can_move = True

    # Move according to the current stage and available distance
    if car['move_stage'] == 'horizontal':
        if is_at_intersection:
            if (grid_x, grid_y) in traffic_lights:
                light = traffic_lights[(grid_x, grid_y)]
                direction = 'left' if dest_x < current_x else 'right'
                if light.is_green(direction) and not light.force_red:
                    can_move = True
                else:
                    can_move = False
        if can_move:
            if distance_x > 0:
                car['pixel_position'][0] += CAR_SPEED if dest_x > current_x else -CAR_SPEED
            elif distance_x == 0 and distance_y > 0:
                # Change direction if stuck and the other direction has distance to cover
                car['move_stage'] = 'vertical'

    if car['move_stage'] == 'vertical':
        if is_at_intersection:
            if (grid_x, grid_y) in traffic_lights:
                light = traffic_lights[(grid_x, grid_y)]
                direction = 'up' if dest_y < current_y else 'down'
                if light.is_green(direction) and not light.force_red:
                    can_move = True
                else:
                    can_move = False
        if can_move:
            if distance_y > 0:
                car['pixel_position'][1] += CAR_SPEED if dest_y > current_y else -CAR_SPEED
            elif distance_y == 0 and distance_x > 0:
                # Change direction if stuck and the other direction has distance to cover
                car['move_stage'] = 'horizontal'

    # Check if the car reached its destination
    if current_x == dest_x and current_y == dest_y:
        car['active'] = False


def manage_cars(current_time):
    """Checks each car's status; regenerates a car if it has reached its destination."""
    global cars  # To modify the global list of cars
    for i, car in enumerate(cars):
        if car['active']:
            move_car(car)
            if not car['active']:  # Car has reached its destination and becomes inactive
                cars[i] = generate_car()  # Replace it with a new car
        else:
            # Activate the car if its start time has passed
            if current_time >= car['start_time']:
                car['active'] = True


def draw_cars():
    """Draws only active cars and their destinations."""
    for car in cars:
        if car['active']:
            pixel_x, pixel_y = car['pixel_position']
            dest_x, dest_y = car['destination']
            # Draw the car and its destination
            pygame.draw.rect(screen, car['color'], (pixel_x, pixel_y, CAR_SIZE, CAR_SIZE))
            pygame.draw.circle(screen, car['color'], (dest_x + CAR_SIZE // 2, dest_y + CAR_SIZE // 2), CAR_SIZE // 2)


def draw_traffic_lights():
    for (x, y), light in traffic_lights.items():
        # Calculate the center position of each intersection
        center_x = GRID_OFFSET + x * CELL_SIZE
        center_y = GRID_OFFSET + y * CELL_SIZE
        radius = CAR_SIZE // 2  # Radius for the traffic light indicator

        # Determine the offset for the green light circle based on the light's current direction
        if light.current_light == 0:  # 'left'
            circle_x = center_x - CELL_SIZE // 6
            circle_y = center_y
        elif light.current_light == 1:  # 'down'
            circle_x = center_x
            circle_y = center_y + CELL_SIZE // 6
        elif light.current_light == 2:  # 'right'
            circle_x = center_x + CELL_SIZE // 6
            circle_y = center_y
        elif light.current_light == 3:  # 'up'
            circle_x = center_x
            circle_y = center_y - CELL_SIZE // 6

        # Draw the green circle for the traffic light
        if light.force_red:
            pygame.draw.circle(screen, (255, 0, 0), (circle_x, circle_y), radius)
        else:
            pygame.draw.circle(screen, (0, 255, 0), (circle_x, circle_y), radius)


def update_train():
    # Index for direction in the cycle
    direction_index = train['cycle'].index(train['direction'])

    # Move the train based on its current direction
    if train['direction'] == 'right':
        train['position'][0] += train['speed']
        train['size'] = (CAR_SIZE * 3, CAR_SIZE)
        if train['position'][0] >= GRID_OFFSET + CELL_SIZE * 2:
            direction_index = (direction_index + 1) % len(train['cycle'])  # Move to next direction in the cycle
    elif train['direction'] == 'down':
        train['position'][1] += train['speed']
        train['size'] = (CAR_SIZE, CAR_SIZE * 3)
        if train['position'][1] >= GRID_OFFSET + CELL_SIZE * 2:
            direction_index = (direction_index + 1) % len(train['cycle'])
    elif train['direction'] == 'left':
        train['position'][0] -= train['speed']
        train['size'] = (CAR_SIZE * 3, CAR_SIZE)
        if train['position'][0] <= GRID_OFFSET + CELL_SIZE:
            direction_index = (direction_index + 1) % len(train['cycle'])
    elif train['direction'] == 'up':
        train['position'][1] -= train['speed']
        train['size'] = (CAR_SIZE, CAR_SIZE * 3)
        if train['position'][1] <= GRID_OFFSET + CELL_SIZE:
            direction_index = (direction_index + 1) % len(train['cycle'])

    # Update the direction from the cycle
    train['direction'] = train['cycle'][direction_index]


def euclidean_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def update_nearest_traffic_light():
    train_center_x = train['position'][0] + train['size'][0] // 2
    train_center_y = train['position'][1] + train['size'][1] // 2
    min_distance = float('inf')
    nearest_light = None

    # Reset all lights from forced red state before updating the nearest one
    for light in traffic_lights.values():
        light.set_force_red(False)

    # Iterate through all traffic lights to find the closest one
    for (x, y), light in traffic_lights.items():
        light_center_x = GRID_OFFSET + x * CELL_SIZE
        light_center_y = GRID_OFFSET + y * CELL_SIZE
        distance = euclidean_distance(train_center_x, train_center_y, light_center_x, light_center_y)

        if distance < min_distance:
            min_distance = distance
            nearest_light = light

    # Set the nearest traffic light to forced red
    if nearest_light:
        nearest_light.set_force_red(True)


def draw_train():
    pygame.draw.rect(screen, train['color'],
                     (train['position'][0], train['position'][1], train['size'][0], train['size'][1]))


def display_time(current_time):
    """Displays the current time on the screen."""
    time_text = f"Time: {current_time:.1f}s"
    text_surface = font.render(time_text, True, WHITE)
    screen.blit(text_surface, (10, 10))  # Position the text in the top left corner


def update_traffic_lights():
    for light in traffic_lights.values():
        light.update()

def main():
    global start_logging_time, logging_active
    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    running = True
    start_logging_time = time.time()

    while running:
        current_ticks = pygame.time.get_ticks()
        current_time = (current_ticks - start_ticks) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if logging_active and (time.time() - start_logging_time) > 15:
            logging_active = False
            plt.plot([td[0] for td in traffic_data], [td[1] for td in traffic_data], marker='o', linestyle='-')
            plt.title('Traffic Data Over Time')
            plt.xlabel('Time (s)')
            plt.ylabel('Number of Cars Passing a Point')
            plt.show()
        screen.fill(BLACK)
        draw_grid()
        manage_cars(current_time)
        update_traffic_lights()  # Regular update for normal light cycling
        update_nearest_traffic_light()  # Check and update the nearest light to the train
        draw_traffic_lights()
        update_train()  # Update train's position
        draw_train()
        draw_cars()
        display_time(current_time)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
