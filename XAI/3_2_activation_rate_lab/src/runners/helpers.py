import numpy as np
import os
import json

_POSSIBLE_BOUND_FNS = [">=", "<=", ">", "<"]
_BOUNDS_FN_MAP = {
    ">=": (lambda x, y: x >= y),
    ">": (lambda x, y: x > y),
    "<=": (lambda x, y: x <= y),
    "<": (lambda x, y: x < y),
}
_POSSIBLE_VARIABLES = [f"V{i}" for i in range(8)]

_RWARE_ACTION_MAP = {
    0: "noop",
    1: "forward",
    2: "left",
    3: "right",
    4: "load",
}
_RWARE_LAYERS = ["Shelves", "Good shelves", "Agents", "Agents load", "Goal"]

ROOT = os.getcwd()


def _does_rule_activate(atoms, rule) -> bool:
    equal = True
    for body_atom in rule:
        # If already grounded just check
        grounded = np.all([v not in body_atom for v in _POSSIBLE_VARIABLES])
        if grounded and body_atom not in atoms:
            equal = False
            break
        elif not grounded:
            # Skip bounds
            if np.any(np.array([bound in body_atom for bound in _POSSIBLE_BOUND_FNS])):
                continue
            variable = ""
            for var in _POSSIBLE_VARIABLES:
                if var in body_atom:
                    variable = var
                    break
            # Now find the bound
            for body_atom2 in rule:
                if body_atom2.startswith(variable):
                    # Find the bound type /fn and value
                    bound_fn = None
                    for bound_type in _POSSIBLE_BOUND_FNS:
                        if bound_type in body_atom2:
                            bound_fn = _BOUNDS_FN_MAP[bound_type]
                            bound = int(body_atom2.split(bound_type)[1])
                            break
            # Now we have grounded a variable: we need just one in the atoms
            possibilities = [
                body_atom.replace(variable, str(x))
                for x in range(80)
                if bound_fn(x, bound)
            ]

            # All false
            try:
                if np.all(~np.array([p in atoms for p in possibilities])):
                    equal = False
                    break
            except:
                print("REMOVE THIS")

    return equal


def rware_obs_to_atoms(obs=np.ones((2)), sensor_range=3, adversary=False):
    atoms = []
    carried_shelves = set()
    agent_positions = np.where(obs[2] > 0.0)
    positions = zip(
        [y.item() for y in agent_positions[0]],
        [x.item() for x in agent_positions[1]],
    )
    for y, x in positions:
        carrying = obs[3, y, x] > 0.0
        if carrying:
            carried_shelves.add((y, x))
        if not carrying:
            carrying_status = "n"
        elif carrying and obs[1, y, x] > 0.0:
            carrying_status = "g"
        else:
            carrying_status = "b"
        # Ego
        if x == y and x == sensor_range:
            atoms.append(f"ego({carrying_status})")
        else:
            # Not ego
            direction_y = "north" if y - sensor_range < 0 else "south"
            direction_x = "west" if x - sensor_range < 0 else "east"
            atoms.append(f"agent({direction_x}, {abs(x - sensor_range)})")
            atoms.append(f"agent({direction_y}, {abs(y - sensor_range)})")

    good_shelves_positions = np.where(obs[1] > 0.0)
    positions = zip(
        [y.item() for y in good_shelves_positions[0]],
        [x.item() for x in good_shelves_positions[1]],
    )
    for y, x in positions:
        # The second should not happen
        if (y, x) in carried_shelves or (y == x and x == 0):
            continue

        direction_y = "north" if y - sensor_range < 0 else "south"
        direction_x = "west" if x - sensor_range < 0 else "east"
        atoms.append(f"good_shelf({direction_x}, {abs(x - sensor_range)})")
        atoms.append(f"good_shelf({direction_y}, {abs(y - sensor_range)})")

    goals_positions = np.where(obs[4] > 0.0)
    positions = zip(
        [y.item() for y in goals_positions[0]],
        [x.item() for x in goals_positions[1]],
    )
    for y, x in positions:
        direction_y = "north" if y - sensor_range < 0 else "south"
        direction_x = "west" if x - sensor_range < 0 else "east"
        atoms.append(f"goal({direction_x}, {abs(x - sensor_range)})")
        atoms.append(f"goal({direction_y}, {abs(y - sensor_range)})")
    return atoms


def get_theory_dict(steps, index, n_agents, action_map):
    """Return theory dict for `env` after training `steps` steps in the form: [{action: theory}]_i
    where theory is a list of sets containing the atoms (one set for rule in the theory) and i is the agent index.
    That is, a list of dictionary containing, for the agent at index i, the theory mapped to each action"""
    print(f">>>>>>>>>>>>>> LOAD theory:RWARE:{index}:{steps} <<<<<<<<<<<<<<")

    theories = []
    file_path = os.path.join(ROOT, "rware_theories.json")
    with open(file_path, "r") as f:
        data_json = json.load(f)

    for agent_idx in range(n_agents):
        theory = dict()
        for action in action_map.values():
            if action == "noop":
                continue
            action_theory = []
            theory_set = data_json[str(index)][str(steps)][str(agent_idx)][action]
            for rule in theory_set:
                if rule == "":
                    action_theory.append(set(["[NEVER]"]))
                    continue
                assert rule.startswith(action), (
                    f"{rule} [needed to start with {action}]"
                )
                body_set = set([
                    r.lstrip(" ")
                    .rstrip(" ")
                    .rstrip(".")
                    .rstrip(" ")
                    .rstrip(".")
                    .rstrip(" ")
                    for r in rule.lstrip(f"{action} :- ").split(";")
                ])
                for at in body_set:
                    assert not at.endswith("."), (
                        f"{at} || {rule} || {action} || {agent_idx} || {index}"
                    )
                action_theory.append(body_set)

            theory[action] = action_theory
        theories.append(theory)
    return theories
