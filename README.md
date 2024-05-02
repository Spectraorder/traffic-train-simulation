# Traffic Simulation Project

This Traffic Simulation project is developed using Pygame to model and visualize traffic flow on a grid-based map featuring interactive cars and a train. The simulation includes dynamic traffic lights that influence car movements, a train that moves cyclically through a central path, and cars that navigate through the grid based on traffic light signals.

## Features

- **Dynamic Traffic Lights**: Traffic lights change state at set intervals, controlling the flow of car traffic at intersections.
- **Interactive Train Movement**: A train moves cyclically on a designated path and affects nearby traffic light states.
- **Car Navigation**: Cars are generated randomly at the grid edges with specific destinations and move according to the traffic light status.
- **Data Logging and Visualization**: The simulation logs traffic data for a set duration (15 seconds) and visualizes this data using Matplotlib to assess traffic patterns and system performance.

## Requirements

To run this simulation, you will need:

- Python 3.x
- Pygame
- Matplotlib

## Usage

The graphical interface will display the simulation, where you can observe the traffic dynamics influenced by the traffic lights and train movements.

## How It Works

- **Traffic Light Logic**: Each traffic light operates independently with a timer to switch between green and red states, affecting the car movements across intersections.
- **Train Mechanics**: The train moves in a continuous loop, temporarily forcing nearby traffic lights to red as it passes, ensuring cars stop for the train.
- **Car Movements**: Cars activate based on a timer and move towards their destination, respecting traffic lights and stopping if necessary.