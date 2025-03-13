#!/usr/bin/env python3
"""
Diet Optimizer - Streamlit App

A web-based interface for the diet optimization model using Streamlit.
Allows users to toggle constraints and find balanced solutions.
"""

import time
from statistics import stdev

import matplotlib.pyplot as plt
import pandas as pd
import pulp
import streamlit as st

# Set the CBC solver path to use the one installed via Homebrew
if pulp.apis.COIN_CMD().available():
    solver = pulp.apis.COIN_CMD(path="/opt/homebrew/bin/cbc")
else:
    # Fallback to default solver
    solver = pulp.apis.PULP_CBC_CMD()


class DietOptimizer:
    """Diet Optimizer model logic separated from UI"""

    def __init__(self):
        """Initialize the Diet Optimizer with data"""
        # Initialize data structures
        self.setup_data()

    def setup_data(self):
        """Setup the food and nutrient data"""
        # Define food items with their data
        self.items = [
            "Cassava (dry)",
            "Maize meal",
            "Millet (white)",
            "Potatoes (Irish)",
            "Rice",
            "Sorghum (brown)",
            "Sorghum flour",
            "Fish (dry)",
            "Meat (beef)",
            "Meat (goat)",
            "Milk (fresh)",
            "Sugar (brown)",
            "Beans (fava, dry)",
            "Cowpeas",
            "Groundnuts",
            "Sesame",
            "Okra (dry)",
            "Chicken egg",
            "Tomato",
        ]

        # Maximum consumption in 100g units
        self.max_consumption = {
            "Cassava (dry)": 57.12,
            "Maize meal": 46.41,
            "Millet (white)": 39.27,
            "Potatoes (Irish)": 67.83,
            "Rice": 42.84,
            "Sorghum (brown)": 42.84,
            "Sorghum flour": 35.7,
            "Fish (dry)": 17.85,
            "Meat (beef)": 14.28,
            "Meat (goat)": 14.28,
            "Milk (fresh)": 31.4,
            "Sugar (brown)": 7.14,
            "Beans (fava, dry)": 14.28,
            "Cowpeas": 14.28,
            "Groundnuts": 3.57,
            "Sesame": 3.57,
            "Okra (dry)": 42.84,
            "Chicken egg": 3.5,
            "Tomato": 30.0,
        }

        # Cost per 100g
        self.cost_per_100g = {
            "Cassava (dry)": 43,
            "Maize meal": 50,
            "Millet (white)": 60,
            "Potatoes (Irish)": 117,
            "Rice": 75,
            "Sorghum (brown)": 29,
            "Sorghum flour": 50,
            "Fish (dry)": 1200,
            "Meat (beef)": 300,
            "Meat (goat)": 400,
            "Milk (fresh)": 60,
            "Sugar (brown)": 50,
            "Beans (fava, dry)": 200,
            "Cowpeas": 86,
            "Groundnuts": 120,
            "Sesame": 97,
            "Okra (dry)": 200,
            "Chicken egg": 400,
            "Tomato": 300,
        }

        # Nutrient content per 100g
        self.nutrient_content = {
            "Energy": {
                "Cassava (dry)": 347,
                "Maize meal": 362,
                "Millet (white)": 79,
                "Potatoes (Irish)": 93,
                "Rice": 361.63,
                "Sorghum (brown)": 339,
                "Sorghum flour": 339,
                "Fish (dry)": 290.32,
                "Meat (beef)": 132,
                "Meat (goat)": 103,
                "Milk (fresh)": 66,
                "Sugar (brown)": 376,
                "Beans (fava, dry)": 102,
                "Cowpeas": 336,
                "Groundnuts": 567,
                "Sesame": 573,
                "Okra (dry)": 31,
                "Chicken egg": 139,
                "Tomato": 22,
            },
            "Protein": {
                "Cassava (dry)": 2.1,
                "Maize meal": 8.1,
                "Millet (white)": 1.7,
                "Potatoes (Irish)": 0,
                "Rice": 6.7,
                "Sorghum (brown)": 11.3,
                "Sorghum flour": 11.3,
                "Fish (dry)": 56.56,
                "Meat (beef)": 19.7,
                "Meat (goat)": 20.6,
                "Milk (fresh)": 3.2,
                "Sugar (brown)": 0,
                "Beans (fava, dry)": 7.6,
                "Cowpeas": 23.5,
                "Groundnuts": 25.8,
                "Sesame": 17.7,
                "Okra (dry)": 2,
                "Chicken egg": 12.6,
                "Tomato": 1.01,
            },
            "Fat": {
                "Cassava (dry)": 0.6,
                "Maize meal": 3.6,
                "Millet (white)": 0.1,
                "Potatoes (Irish)": 0.1,
                "Rice": 0.6,
                "Sorghum (brown)": 3.3,
                "Sorghum flour": 3.3,
                "Fish (dry)": 5.44,
                "Meat (beef)": 5.3,
                "Meat (goat)": 2.3,
                "Milk (fresh)": 3.9,
                "Sugar (brown)": 0,
                "Beans (fava, dry)": 0.4,
                "Cowpeas": 1.3,
                "Groundnuts": 49.2,
                "Sesame": 49.7,
                "Okra (dry)": 0.5,
                "Chicken egg": 9.5,
                "Tomato": 0.18,
            },
            "VitaminA": {
                "Cassava (dry)": 0,
                "Maize meal": 23.5,
                "Millet (white)": 0.2,
                "Potatoes (Irish)": 0,
                "Rice": 0,
                "Sorghum (brown)": 6,
                "Sorghum flour": 7,
                "Fish (dry)": 5.52,
                "Meat (beef)": 0,
                "Meat (goat)": 0,
                "Milk (fresh)": 55,
                "Sugar (brown)": 0,
                "Beans (fava, dry)": 0,
                "Cowpeas": 5,
                "Groundnuts": 0,
                "Sesame": 1,
                "Okra (dry)": 19,
                "Chicken egg": 160,
                "Tomato": 51.97,
            },
            "VitaminC": {
                "Cassava (dry)": 5,
                "Maize meal": 0,
                "Millet (white)": 11,
                "Potatoes (Irish)": 18.6,
                "Rice": 0,
                "Sorghum (brown)": 0,
                "Sorghum flour": 0,
                "Fish (dry)": 0.24,
                "Meat (beef)": 0,
                "Meat (goat)": 0,
                "Milk (fresh)": 1,
                "Sugar (brown)": 0,
                "Beans (fava, dry)": 1,
                "Cowpeas": 0,
                "Groundnuts": 0,
                "Sesame": 0,
                "Okra (dry)": 21.1,
                "Chicken egg": 0,
                "Tomato": 29.64,
            },
            "FolicAcid": {
                "Cassava (dry)": 47,
                "Maize meal": 25,
                "Millet (white)": 8,
                "Potatoes (Irish)": 75,
                "Rice": 6,
                "Sorghum (brown)": 11,
                "Sorghum flour": 14,
                "Fish (dry)": 28.32,
                "Meat (beef)": 5,
                "Meat (goat)": 0,
                "Milk (fresh)": 6,
                "Sugar (brown)": 1,
                "Beans (fava, dry)": 35,
                "Cowpeas": 549,
                "Groundnuts": 126,
                "Sesame": 97,
                "Okra (dry)": 88,
                "Chicken egg": 47,
                "Tomato": 21.25,
            },
            "VitaminB12": {
                "Cassava (dry)": 0,
                "Maize meal": 0,
                "Millet (white)": 0,
                "Potatoes (Irish)": 0,
                "Rice": 0,
                "Sorghum (brown)": 0,
                "Sorghum flour": 0,
                "Fish (dry)": 6.27,
                "Meat (beef)": 1.47,
                "Meat (goat)": 1.13,
                "Milk (fresh)": 0.4,
                "Sugar (brown)": 0,
                "Beans (fava, dry)": 0,
                "Cowpeas": 0,
                "Groundnuts": 0,
                "Sesame": 0,
                "Okra (dry)": 0,
                "Chicken egg": 0.9,
                "Tomato": 0,
            },
            "IronAbs": {
                "Cassava (dry)": 0.08,
                "Maize meal": 0.18,
                "Millet (white)": 0.02,
                "Potatoes (Irish)": 0.4,
                "Rice": 0.03,
                "Sorghum (brown)": 4.1,
                "Sorghum flour": 4.1,
                "Fish (dry)": 0.45,
                "Meat (beef)": 0.33,
                "Meat (goat)": 2.8,
                "Milk (fresh)": 0.01,
                "Sugar (brown)": 0.1,
                "Beans (fava, dry)": 1.4,
                "Cowpeas": 6.6,
                "Groundnuts": 4.6,
                "Sesame": 14.6,
                "Okra (dry)": 0.8,
                "Chicken egg": 0.45,
                "Tomato": 0.03,
            },
        }

        # Define the constraints with their limit values
        self.constraints = {
            "Energy_min": {"nutrient": "Energy", "type": "min", "value": 22400},
            "Energy_max": {"nutrient": "Energy", "type": "max", "value": 28000},
            "Protein_min": {"nutrient": "Protein", "type": "min", "value": 455},
            "Fat_min": {"nutrient": "Fat", "type": "min", "value": 560},
            "Fat_max": {"nutrient": "Fat", "type": "max", "value": 980},
            "VitaminA_min": {"nutrient": "VitaminA", "type": "min", "value": 4200},
            "VitaminA_max": {"nutrient": "VitaminA", "type": "max", "value": 23100},
            "VitaminC_min": {"nutrient": "VitaminC", "type": "min", "value": 315},
            "VitaminC_max": {"nutrient": "VitaminC", "type": "max", "value": 6594},
            "FolicAcid_min": {"nutrient": "FolicAcid", "type": "min", "value": 2800},
            "VitaminB12_min": {"nutrient": "VitaminB12", "type": "min", "value": 16.8},
            "IronAbs_min": {"nutrient": "IronAbs", "type": "min", "value": 9.59},
            "IronAbs_max": {"nutrient": "IronAbs", "type": "max", "value": 4389},
            "Budget_max": {"type": "budget", "value": 5000},
        }

    def solve_model(self, soft_constraints, constraint_weights):
        """
        Run the optimization model with the given constraint settings

        Args:
            soft_constraints: Dict of constraint names that should be soft
            constraint_weights: Dict of weights for each soft constraint

        Returns:
            Dictionary with model results or None if model failed
        """
        # Setup optimization model
        model = pulp.LpProblem("Diet_Optimization", pulp.LpMinimize)

        # Decision variables - food quantities
        x = {
            item: pulp.LpVariable(
                item,
                lowBound=0,
                upBound=self.max_consumption[item],
                cat=pulp.LpContinuous,
            )
            for item in self.items
        }

        # Create shortfall variables for soft constraints
        shortfall_vars = {}
        for constraint_name in self.constraints:
            is_soft = constraint_name in soft_constraints

            if is_soft and self.constraints[constraint_name]["type"] == "min":
                shortfall_vars[constraint_name] = pulp.LpVariable(
                    f"shortfall_{constraint_name}", lowBound=0, cat=pulp.LpContinuous
                )
            elif is_soft and self.constraints[constraint_name]["type"] == "max":
                shortfall_vars[constraint_name] = pulp.LpVariable(
                    f"shortfall_{constraint_name}", lowBound=0, cat=pulp.LpContinuous
                )

        # Initialize weighted objective (sum of weighted shortfalls)
        weighted_objective = pulp.LpAffineExpression()
        for c_name, var in shortfall_vars.items():
            weight = constraint_weights.get(c_name, 1)
            weighted_objective += weight * var

        # Add constraints
        for constraint_name, constraint_info in self.constraints.items():
            is_soft = constraint_name in soft_constraints

            # Budget constraint is always hard
            if constraint_name == "Budget_max":
                total_cost = pulp.lpSum(
                    self.cost_per_100g[i] * x[i] for i in self.items
                )
                model += (total_cost <= constraint_info["value"]), constraint_name
                continue

            # Regular nutrient constraints
            nutrient = constraint_info["nutrient"]
            nutrient_sum = pulp.lpSum(
                self.nutrient_content[nutrient][i] * x[i] for i in self.items
            )

            if constraint_info["type"] == "min":
                if is_soft:
                    # Soft minimum: sum(nutrient) + shortfall >= value
                    model += (
                        nutrient_sum + shortfall_vars[constraint_name]
                        >= constraint_info["value"]
                    ), constraint_name
                else:
                    # Hard minimum: sum(nutrient) >= value
                    model += nutrient_sum >= constraint_info["value"], constraint_name
            else:  # max constraint
                if is_soft:
                    # Soft maximum: sum(nutrient) - shortfall <= value
                    model += (
                        nutrient_sum - shortfall_vars[constraint_name]
                        <= constraint_info["value"]
                    ), constraint_name
                else:
                    # Hard maximum: sum(nutrient) <= value
                    model += nutrient_sum <= constraint_info["value"], constraint_name

        # Set objective: minimize weighted sum of all shortfalls
        model.setObjective(weighted_objective)

        # Solve the model
        try:
            # Set timeout to avoid hanging
            model.solve(solver)
        except Exception as e:
            st.error(f"Error solving model: {str(e)}")
            return None

        # Process results
        results = {
            "status": model.status,
            "x": {i: x[i].value() for i in self.items},
            "shortfalls": {c: var.value() for c, var in shortfall_vars.items()},
            "food_quantities": {i: x[i].value() for i in self.items},
            "objective": pulp.value(model.objective),
        }

        # Calculate achieved values for each nutrient
        results["achieved_values"] = {}
        for nutrient in self.nutrient_content:
            achieved = sum(
                self.nutrient_content[nutrient][i] * x[i].value() for i in self.items
            )
            results["achieved_values"][nutrient] = achieved

        # Calculate total cost
        results["total_cost"] = sum(
            self.cost_per_100g[i] * x[i].value() for i in self.items
        )

        return results

    def calculate_balance_score(self, result):
        """Calculate a score for how balanced the constraint violations are.
        Lower is better - means more even distribution of constraint violations."""
        if not result or not result.get("shortfalls"):
            return 0  # No shortfalls, perfect solution

        normalized_shortfalls = []
        for constraint_name, shortfall in result["shortfalls"].items():
            constraint_info = self.constraints[constraint_name]
            target_value = constraint_info["value"]
            if target_value > 0 and shortfall is not None:  # Add safety check for None
                # Normalize as percentage of target value
                normalized = (shortfall / target_value) * 100
                normalized_shortfalls.append(normalized)

        # Safety checks for empty or None values
        if not normalized_shortfalls:
            return 0  # No valid shortfalls

        normalized_shortfalls = [n for n in normalized_shortfalls if n is not None]
        if not normalized_shortfalls:
            return 0  # All shortfalls were None

        # Calculate statistics on normalized shortfalls
        avg_normalized = sum(normalized_shortfalls) / len(normalized_shortfalls)
        max_normalized = max(normalized_shortfalls)
        min_normalized = min(normalized_shortfalls)

        # Standard deviation (measure of spread)
        if len(normalized_shortfalls) > 1:
            try:
                stdev_value = stdev(normalized_shortfalls)
            except:
                stdev_value = max_normalized  # Fallback if stdev fails
        else:
            stdev_value = 0

        # Calculate the range of shortfalls (max - min)
        shortfall_range = max_normalized - min_normalized

        # Calculate sum of pairwise differences between shortfalls
        pairwise_diff_sum = 0
        count = len(normalized_shortfalls)
        if count > 1:
            num_pairs = 0
            for i in range(count):
                for j in range(i + 1, count):
                    pairwise_diff_sum += abs(
                        normalized_shortfalls[i] - normalized_shortfalls[j]
                    )
                    num_pairs += 1
            pairwise_diff_sum /= num_pairs

        # Apply a non-linear penalty to the shortfall range and standard deviation
        range_penalty = (
            shortfall_range**2 / 100
        )  # Divide by 100 to keep reasonable scale
        stdev_penalty = stdev_value**2 / 100  # Square to penalize higher standard devs

        # Combined score with emphasis on evenness
        return (
            (pairwise_diff_sum * 0.4)
            + (range_penalty * 0.4)
            + (stdev_penalty * 0.1)
            + (avg_normalized * 0.1)
        )

    def find_balanced_solution(self, status_callback=None):
        """Find the most balanced solution by trying different constraint combinations

        Args:
            status_callback: Optional function to update UI with progress

        Returns:
            Dictionary with best solution details
        """
        # Track best solution
        best_solution = None
        best_score = float("inf")
        best_config = {}
        best_constraint_set = set()
        best_shortfalls = {}
        best_balance_details = None

        # First try with all constraints as hard
        if status_callback:
            status_callback("Trying all constraints as hard...")

        all_hard_result = self.solve_model({}, {})

        if all_hard_result and all_hard_result["status"] == pulp.LpStatusOptimal:
            if status_callback:
                status_callback("Found feasible solution with all constraints as hard!")
            return {
                "solution": all_hard_result,
                "score": 0,
                "config": {},
                "balance_details": None,
            }

        # Next try with all constraints (except Budget) soft
        eligible_constraints = [
            constraint for constraint in self.constraints if constraint != "Budget_max"
        ]

        # Sort constraints by type
        min_constraints = [
            c for c in eligible_constraints if self.constraints[c]["type"] == "min"
        ]
        max_constraints = [
            c for c in eligible_constraints if self.constraints[c]["type"] == "max"
        ]

        # Track when last improvement was found
        start_time = time.time()
        last_improvement_time = start_time
        solutions_found = 0

        # Maximum search time
        max_search_time = 60  # seconds

        # Try with only MIN constraints as soft (as per user's observation)
        if status_callback:
            status_callback("Trying minimum constraints as soft (user suggested)...")

        # Define some weight combinations to try for min constraints, including user's suggestion
        min_weight_combinations = [
            # User's suggested weights (for all min constraints)
            [1, 1, 1, 2, 8, 1, 1, 1],  # Updated best weights found by user
            [1, 1, 1, 3, 7, 2, 2, 1],  # Previous suggestion
            # Some alternative weight distributions to try
            [1] * len(min_constraints),  # All equal weights
            [2] * len(min_constraints),  # All weight 2
            [5] * len(min_constraints),  # All weight 5
            [1, 2, 3, 4, 5, 6, 7, 8],  # Increasing weights
            [8, 7, 6, 5, 4, 3, 2, 1],  # Decreasing weights
            [1, 5, 1, 5, 1, 5, 1, 5],  # Alternating
        ]

        # Try each weight combination for min constraints
        for weights_list in min_weight_combinations:
            # Create weight dictionary
            weights = {}
            for i, constraint in enumerate(min_constraints):
                if i < len(weights_list):
                    weights[constraint] = weights_list[i]
                else:
                    weights[constraint] = 1  # Default weight

            # Run model with only min constraints as soft
            result = self.solve_model(min_constraints, weights)

            if result and result["status"] == pulp.LpStatusOptimal:
                score = self.calculate_balance_score(result)

                if score < best_score:
                    best_score = score
                    best_solution = result
                    best_config = {
                        c: (True, weights.get(c, 1)) for c in min_constraints
                    }
                    best_shortfalls = result.get("shortfalls", {})
                    best_constraint_set = set(min_constraints)
                    solutions_found += 1
                    last_improvement_time = time.time()

                    if status_callback:
                        status_callback(
                            f"Found better solution with min constraints. Score: {best_score:.2f}"
                        )

                    # Calculate balance details
                    if best_shortfalls:
                        shortfall_percentages = []
                        for c, s in best_shortfalls.items():
                            target = self.constraints[c]["value"]
                            if target > 0:
                                pct = (s / target) * 100
                                shortfall_percentages.append(pct)

                        if shortfall_percentages:
                            max_pct = max(shortfall_percentages)
                            avg_pct = sum(shortfall_percentages) / len(
                                shortfall_percentages
                            )
                            stdev_pct = (
                                stdev(shortfall_percentages)
                                if len(shortfall_percentages) > 1
                                else 0
                            )

                            best_balance_details = {
                                "max_pct": max_pct,
                                "avg_pct": avg_pct,
                                "stdev_pct": stdev_pct,
                                "num_shortfalls": len(shortfall_percentages),
                            }

        # Try with all constraints soft as a fallback
        if status_callback:
            status_callback("Trying all constraints as soft (fallback)...")

        all_soft_weights = {c: 1 for c in eligible_constraints}
        all_soft_result = self.solve_model(eligible_constraints, all_soft_weights)

        if all_soft_result and all_soft_result["status"] == pulp.LpStatusOptimal:
            all_soft_score = self.calculate_balance_score(all_soft_result)

            if all_soft_score < best_score:
                best_solution = all_soft_result
                best_score = all_soft_score
                best_config = {c: (True, 1) for c in eligible_constraints}
                best_constraint_set = set(eligible_constraints)
                best_shortfalls = best_solution.get("shortfalls", {})
                solutions_found += 1

                if status_callback:
                    status_callback(
                        f"Found better solution with all soft constraints. Score: {best_score:.2f}"
                    )

        # Only continue exploration if we haven't found good solutions yet
        if best_score > 10 or solutions_found < 2:
            # Generate constraint combinations to try
            constraint_combinations = []

            # Try various combinations
            import itertools

            # Try some max constraints combinations
            constraint_combinations.append(max_constraints)

            # Try some mixed combinations (few min + few max)
            for num_min in range(1, min(4, len(min_constraints) + 1)):
                for num_max in range(1, min(4, len(max_constraints) + 1)):
                    for min_combo in itertools.combinations(min_constraints, num_min):
                        for max_combo in itertools.combinations(
                            max_constraints, num_max
                        ):
                            constraint_combinations.append(
                                list(min_combo) + list(max_combo)
                            )
                            # Limit to avoid too many combinations
                            if len(constraint_combinations) >= 20:
                                break
                        if len(constraint_combinations) >= 20:
                            break
                    if len(constraint_combinations) >= 20:
                        break
                if len(constraint_combinations) >= 20:
                    break

            # Testing different constraint combinations
            if status_callback:
                status_callback(
                    f"Testing {len(constraint_combinations)} additional combinations..."
                )

            for i, constraints_to_soften in enumerate(constraint_combinations):
                current_time = time.time()

                # Check if time limit exceeded
                if current_time - start_time > max_search_time:
                    if status_callback:
                        status_callback(
                            f"Time limit ({max_search_time}s) reached. Using best solution."
                        )
                    break

                # Try different weight distributions
                weight_distributions = []

                # Equal weights
                weight_distributions.append({c: 1 for c in constraints_to_soften})
                weight_distributions.append({c: 3 for c in constraints_to_soften})

                # Try each weight distribution
                for weights in weight_distributions:
                    result = self.solve_model(constraints_to_soften, weights)

                    if result and result["status"] == pulp.LpStatusOptimal:
                        score = self.calculate_balance_score(result)

                        if score < best_score:
                            best_score = score
                            best_solution = result
                            best_config = {
                                c: (True, weights.get(c, 1))
                                for c in constraints_to_soften
                            }
                            best_shortfalls = result.get("shortfalls", {})
                            best_constraint_set = set(constraints_to_soften)
                            solutions_found += 1
                            last_improvement_time = time.time()

                            # Calculate balance details
                            if best_shortfalls:
                                shortfall_percentages = []
                                for c, s in best_shortfalls.items():
                                    target = self.constraints[c]["value"]
                                    if target > 0:
                                        pct = (s / target) * 100
                                        shortfall_percentages.append(pct)

                                if shortfall_percentages:
                                    max_pct = max(shortfall_percentages)
                                    avg_pct = sum(shortfall_percentages) / len(
                                        shortfall_percentages
                                    )
                                    stdev_pct = (
                                        stdev(shortfall_percentages)
                                        if len(shortfall_percentages) > 1
                                        else 0
                                    )

                                    best_balance_details = {
                                        "max_pct": max_pct,
                                        "avg_pct": avg_pct,
                                        "stdev_pct": stdev_pct,
                                        "num_shortfalls": len(shortfall_percentages),
                                    }

                            if status_callback:
                                constraint_names = [
                                    self.constraints[c]["nutrient"]
                                    + " "
                                    + self.constraints[c]["type"]
                                    for c in constraints_to_soften
                                ]
                                status_callback(
                                    f"Found better solution with {len(constraints_to_soften)} soft constraints:\n"
                                    f"{', '.join(constraint_names)}\n"
                                    f"Balance score: {best_score:.2f}"
                                )

        return {
            "solution": best_solution,
            "score": best_score,
            "config": best_config,
            "balance_details": best_balance_details,
        }


# Streamlit UI
def main():
    st.set_page_config(page_title="Diet Optimizer", page_icon="🥗", layout="wide")

    st.title("Diet Optimizer")
    st.write("Optimize your diet based on nutritional constraints")

    # Initialize the diet optimizer
    if "diet_optimizer" not in st.session_state:
        st.session_state.diet_optimizer = DietOptimizer()
        st.session_state.results = None
        st.session_state.status = ""
        # Initialize session state for constraint configurations
        st.session_state.best_config = {}
        st.session_state.apply_best_config = False

    # Store best configuration from find_balanced_solution
    if "best_config" not in st.session_state:
        st.session_state.best_config = {}

    # Flag to indicate if we should apply the best configuration
    if "apply_best_config" not in st.session_state:
        st.session_state.apply_best_config = False

    optimizer = st.session_state.diet_optimizer

    # Create sidebar for controls and results area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Constraints")
        st.write(
            "Select which constraints should be soft (can be violated with penalty)"
        )

        # Organize constraints by type (min/max)
        min_constraints = {}
        max_constraints = {}

        for name, info in optimizer.constraints.items():
            if name == "Budget_max":
                continue  # Skip budget, always hard

            if info["type"] == "min":
                min_constraints[name] = info
            else:
                max_constraints[name] = info

        # Function to initialize constraint widget values
        def get_initial_soft_state(constraint_name):
            # Always check best_config, not just when apply_best_config is True
            # This ensures checkboxes remain checked even when modifying weights
            if constraint_name in st.session_state.best_config:
                return st.session_state.best_config[constraint_name][0]  # is_soft
            return False

        def get_initial_weight(constraint_name):
            # Always check best_config, not just when apply_best_config is True
            if constraint_name in st.session_state.best_config:
                return st.session_state.best_config[constraint_name][1]  # weight
            return 1

        # Add function to track when a checkbox or slider is changed
        def on_constraint_change():
            # Update best_config whenever constraints change, to preserve state across renders
            updated_config = {}
            for name in optimizer.constraints:
                if name == "Budget_max":
                    continue  # Skip budget

                is_soft = st.session_state.get(f"soft_{name}", False)
                weight = st.session_state.get(f"weight_{name}", 1)

                if is_soft:
                    updated_config[name] = (True, weight)

            # Update the best_config to reflect current UI state
            st.session_state.best_config = updated_config

        # Collect all soft constraints
        soft_constraints = {}

        # Create a more consistent layout for constraints
        st.subheader("Minimum Constraints")

        for name, info in min_constraints.items():
            nutrient = info["nutrient"]
            value = info["value"]

            # Create a container for each constraint to keep layout stable
            with st.container():
                cols = st.columns([3, 2])

                # Checkbox in first column
                with cols[0]:
                    is_soft = st.checkbox(
                        f"{nutrient} ≥ {value}",
                        key=f"soft_{name}",
                        value=get_initial_soft_state(name),
                        on_change=on_constraint_change,
                    )

                # Slider in second column (always visible, but only active when is_soft=True)
                with cols[1]:
                    weight = st.slider(
                        "Weight",
                        1,
                        10,
                        value=get_initial_weight(name),
                        key=f"weight_{name}",
                        disabled=not is_soft,
                        on_change=on_constraint_change,
                    )

                # If soft, add to constraints dict
                if is_soft:
                    soft_constraints[name] = True

        st.subheader("Maximum Constraints")

        for name, info in max_constraints.items():
            nutrient = info["nutrient"]
            value = info["value"]

            # Create a container for each constraint to keep layout stable
            with st.container():
                cols = st.columns([3, 2])

                # Checkbox in first column
                with cols[0]:
                    is_soft = st.checkbox(
                        f"{nutrient} ≤ {value}",
                        key=f"soft_{name}",
                        value=get_initial_soft_state(name),
                        on_change=on_constraint_change,
                    )

                # Slider in second column (always visible, but only active when is_soft=True)
                with cols[1]:
                    weight = st.slider(
                        "Weight",
                        1,
                        10,
                        value=get_initial_weight(name),
                        key=f"weight_{name}",
                        disabled=not is_soft,
                        on_change=on_constraint_change,
                    )

                # If soft, add to constraints dict
                if is_soft:
                    soft_constraints[name] = True

        st.subheader("Budget Constraint")
        with st.container():
            cols = st.columns([3, 2])
            with cols[0]:
                st.write("Budget ≤ 5000 (always hard)")

        # After we've applied the best config, reset the flag
        if st.session_state.apply_best_config:
            st.session_state.apply_best_config = False
            # Don't clear best_config, so users can keep modifying the solution
            # Instead, keep the config in place for continued modification

        # Action buttons
        st.markdown("---")
        col_buttons = st.columns(2)

        with col_buttons[0]:
            if st.button("Run Optimization", use_container_width=True):
                # Get constraint weights
                weights = {}
                for name in soft_constraints:
                    weights[name] = st.session_state.get(f"weight_{name}", 1)

                # Run the model
                with st.spinner("Running optimization..."):
                    results = optimizer.solve_model(soft_constraints, weights)
                    if results and results["status"] == pulp.LpStatusOptimal:
                        # Calculate balance score for any optimization run, not just "Find Balanced"
                        st.session_state.balance_score = (
                            optimizer.calculate_balance_score(results)
                        )

                        # Also calculate balance details for any optimization
                        if "shortfalls" in results and results["shortfalls"]:
                            shortfall_percentages = []
                            for c, s in results["shortfalls"].items():
                                target = optimizer.constraints[c]["value"]
                                if target > 0:
                                    pct = (s / target) * 100
                                    shortfall_percentages.append(pct)

                            if shortfall_percentages:
                                max_pct = max(shortfall_percentages)
                                avg_pct = sum(shortfall_percentages) / len(
                                    shortfall_percentages
                                )
                                stdev_pct = (
                                    stdev(shortfall_percentages)
                                    if len(shortfall_percentages) > 1
                                    else 0
                                )

                                st.session_state.balance_details = {
                                    "max_pct": max_pct,
                                    "avg_pct": avg_pct,
                                    "stdev_pct": stdev_pct,
                                    "num_shortfalls": len(shortfall_percentages),
                                }

                        # Update the current "best" configuration to reflect current settings
                        st.session_state.best_config = {
                            name: (True, weights.get(name, 1))
                            for name in soft_constraints
                        }

                    st.session_state.results = results

        with col_buttons[1]:
            if st.button("Find Balanced Solution", use_container_width=True):
                status_placeholder = st.empty()

                def update_status(message):
                    status_placeholder.write(message)
                    st.session_state.status = message

                with st.spinner("Searching for balanced solution..."):
                    balanced_result = optimizer.find_balanced_solution(update_status)

                    if balanced_result and balanced_result["solution"]:
                        # Store the best configuration
                        st.session_state.best_config = balanced_result["config"]
                        # Set flag to apply this configuration on next render
                        st.session_state.apply_best_config = True
                        # Store solution results
                        st.session_state.results = balanced_result["solution"]
                        # Add balance details
                        st.session_state.balance_details = balanced_result[
                            "balance_details"
                        ]
                        st.session_state.balance_score = balanced_result["score"]
                        # Force a rerun to apply the new configuration
                        st.rerun()
                    else:
                        st.error("Could not find a balanced solution")

        # Show status if available
        if st.session_state.status:
            st.info(st.session_state.status)

    # Display results in right column
    with col2:
        st.header("Results")

        if st.session_state.results:
            results = st.session_state.results

            if results["status"] == pulp.LpStatusOptimal:
                st.success("Optimal solution found!")

                # Show balance details if available
                if "balance_score" in st.session_state:
                    st.write(f"Balance score: {st.session_state.balance_score:.2f}")

                    if (
                        "balance_details" in st.session_state
                        and st.session_state.balance_details
                    ):
                        details = st.session_state.balance_details
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Max Shortfall", f"{details['max_pct']:.1f}%")
                        with metric_cols[1]:
                            st.metric("Avg Shortfall", f"{details['avg_pct']:.1f}%")
                        with metric_cols[2]:
                            st.metric(
                                "Shortfall Std Dev", f"{details['stdev_pct']:.1f}%"
                            )

                st.metric("Total Cost", f"{results['total_cost']:.2f}")

                # Food Quantities
                st.subheader("Food Quantities")
                food_df = pd.DataFrame(
                    {
                        "Food": [
                            i for i, v in results["food_quantities"].items() if v > 1e-6
                        ],
                        "Quantity (100g)": [
                            v for i, v in results["food_quantities"].items() if v > 1e-6
                        ],
                    }
                )
                if not food_df.empty:
                    st.dataframe(food_df.set_index("Food"), use_container_width=True)

                    # Food quantities chart
                    fig, ax = plt.subplots(figsize=(10, max(6, len(food_df) * 0.4)))
                    bars = ax.barh(food_df["Food"], food_df["Quantity (100g)"])
                    ax.set_xlabel("Quantity (100g units)")
                    ax.set_title("Food Quantities")

                    # Add values on the bars
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(
                            width + 0.1,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.2f}",
                            ha="left",
                            va="center",
                        )

                    st.pyplot(fig)
                    plt.close(fig)  # Close to avoid memory leak
                else:
                    st.write("No food items selected")

                # Nutrient Values
                st.subheader("Nutrient Achieved Values")
                nutrient_df = pd.DataFrame(
                    {
                        "Nutrient": list(results["achieved_values"].keys()),
                        "Achieved Value": list(results["achieved_values"].values()),
                    }
                )
                st.dataframe(
                    nutrient_df.set_index("Nutrient"), use_container_width=True
                )

                # Shortfalls if any
                if "shortfalls" in results and results["shortfalls"]:
                    st.subheader("Constraint Shortfalls")
                    shortfalls = []
                    for c_name, shortfall in results["shortfalls"].items():
                        if shortfall > 1e-6:  # Only show non-zero shortfalls
                            info = optimizer.constraints[c_name]
                            nutrient = (
                                info["nutrient"] if "nutrient" in info else "Budget"
                            )
                            type_str = info["type"]
                            value = info["value"]
                            perc = (shortfall / value) * 100 if value > 0 else 0
                            shortfalls.append(
                                {
                                    "Constraint": f"{nutrient} {type_str}",
                                    "Target": value,
                                    "Shortfall": shortfall,
                                    "% of Target": f"{perc:.1f}%",
                                }
                            )

                    if shortfalls:
                        shortfall_df = pd.DataFrame(shortfalls)
                        st.dataframe(
                            shortfall_df.set_index("Constraint"),
                            use_container_width=True,
                        )

                        # Shortfalls chart
                        fig, ax = plt.subplots(
                            figsize=(10, max(6, len(shortfalls) * 0.4))
                        )
                        bars = ax.barh(
                            [s["Constraint"] for s in shortfalls],
                            [
                                float(s["% of Target"].replace("%", ""))
                                for s in shortfalls
                            ],
                        )
                        ax.set_xlabel("Shortfall (% of target)")
                        ax.set_title("Constraint Shortfalls")

                        # Add values on the bars
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(
                                width + 0.1,
                                bar.get_y() + bar.get_height() / 2,
                                f"{width:.1f}%",
                                ha="left",
                                va="center",
                            )

                        st.pyplot(fig)
                        plt.close(fig)  # Close to avoid memory leak
                    else:
                        st.write("No shortfalls (all constraints satisfied)")
            else:
                st.error(f"No solution found: {pulp.LpStatus[results['status']]}")
                st.write("Try making more constraints soft or adjusting their weights.")
        else:
            st.info("Run optimization or find a balanced solution to see results here.")


if __name__ == "__main__":
    main()
