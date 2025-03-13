# Diet Optimizer

A Streamlit web application for optimizing diet based on nutritional constraints.

## Overview

The Diet Optimizer helps users find balanced food combinations that meet nutritional requirements while minimizing cost. It uses linear programming optimization through PuLP to solve this complex constraint satisfaction problem.

## Features

- Set minimum and maximum constraints for various nutrients
- Make constraints "soft" to allow for violations with penalties
- Adjust weights for each soft constraint to prioritize certain nutrients
- Automatic "Find Balanced Solution" feature to optimize constraint violations
- Visualization of food quantities and nutrient values

## Dependencies

The app requires several Python packages:
- streamlit
- pandas
- numpy
- pulp
- matplotlib
- plus additional supporting packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run diet_optimizer_streamlit.py
   ```

## Usage

1. Select which nutritional constraints should be "soft" (can be violated with a penalty)
2. Adjust the weights for each soft constraint to prioritize certain nutrients
3. Click "Run Optimization" to find a solution, or "Find Balanced Solution" to let the app search for the most balanced set of constraint violations

## License

[MIT License](LICENSE) 