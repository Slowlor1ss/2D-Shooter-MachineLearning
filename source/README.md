<br>

# TODO:

- Add graphs from stats
- --hit walls does not seem to get reset--
- --keep agents alive--
- Relook at ofspring to keep currently 20%
- adjust weights of bad and good
- add in ammo and (food) maybe?
- save brains and load brains implementation
- too close to enemy = bad
- more enemies? maybe helps them to realize that they should shoot the enemy
- grow with age
- possibly add fitness as incentive
- looking at enemies give positiveQ
- based on deltatime issues frame rate independent issues
- check move speed 
- check if lookin at and hitting works correctly
- clean up all the added bullet code
- poistive when avoiding wall?
- positive when moving forward and seeing a enemy
- standing still bad
- we need multiple see enemy sensors or distance to enemies sensors
- Add food and start training with food then add walls with food, then remove foor add an enemy...

0.094   -0.230  -0.785  -0.423  -0.695  0.954   -0.776  0.799
0.070   -0.652  0.589   -0.584  0.814   -0.300  -0.197  -0.825
0.418   -0.016  0.347   -0.527  0.604   -0.741  0.153   0.645
-0.576  -0.512  -0.180  -0.449  -1.000  0.366   0.611   0.515
0.629   -0.962  0.831   0.601   -0.657  -0.449  0.368   0.861
0.157   0.108   -0.796  0.305   0.125   0.841   -0.382  0.532
-0.356  0.901   -0.388  -0.570  0.075   0.558   -0.255  0.641
-0.539  0.094   -0.346  0.088   -0.829  -0.886  0.369   0.241
0.730   0.218   -0.586  0.875   0.560   -0.608  0.645   0.751
-0.938  0.526   0.437   0.572   0.749   0.488   0.956   -0.998

<br>

Thanks for the clarification! Given these notes, we can refine the inputs and outputs to match the constraints of your setup. This makes the environment partially observable, where the agent doesn't have direct information on the enemy’s position until they’re within its field of view (FOV). Here’s an updated design based on those constraints.

### 1. Inputs (Observations)
Without absolute positions and with information only within the agent's FOV, the inputs will focus on the agent's **orientation, FOV sensors**, and **distance to visible objects** (like the enemy or obstacles).

#### Suggested Inputs:
1. **Agent’s Orientation**:
   - **Angle**: A single scalar value representing the agent’s current orientation in the 2D plane (e.g., 0 to 360 degrees or -π to π). This allows the agent to keep track of its own facing direction.

2. **FOV Sensors**:
   - **Enemy Detection**:
     - **Enemy Seen**: A binary flag (1 or 0) indicating whether the enemy is visible within the FOV.
     - **Distance to Enemy**: A single scalar value representing the distance to the enemy when visible (or max distance if not visible).
   
   - **Obstacle Detection**:
     - **Obstacle Proximity in FOV**: A set of distance values from several sensors across the FOV. For example, if you divide the FOV into **3 or 5 segments**, each segment’s sensor would give the distance to the closest obstacle in that segment.
   
     These inputs give the agent a spatial "map" of obstacles within its FOV and the ability to detect the enemy only when directly in view.

3. **Agent’s Speed**:
   - **Current Speed**: A scalar value representing the agent’s current speed. This helps the agent modulate its movement and respond dynamically based on whether it needs to speed up, slow down, or turn.

#### Input Summary:
If we break down the FOV into segments and include agent status data, the inputs might look like this:
   - **Orientation**: 1 value (agent’s facing angle).
   - **Enemy Detection**: 1 binary value (enemy seen) + 1 distance value (if enemy is visible).
   - **Obstacle Proximity**: 3–5 distance values from FOV segments.
   - **Current Speed**: 1 value.

   Total: **7–9 inputs** (depending on the number of FOV segments for obstacle detection).

### 2. Outputs (Actions)
Since you control movement via **angle and speed**, the outputs need to let the agent adjust its heading, speed, and take an action to attack the enemy when aligned.

#### Suggested Outputs:
1. **Movement**:
   - **Rotation Angle**: A continuous action that adjusts the agent’s orientation (e.g., an angular velocity or a target angle adjustment, like ± some degrees).
   - **Speed**: A continuous value that sets the agent’s speed, allowing it to increase, decrease, or maintain its velocity.

2. **Attack**:
   - **Shoot**: A binary action (1 or 0) that triggers firing if the enemy is within the agent’s line of sight and FOV.

#### Output Summary:
The outputs could look like:
   - **Rotation**: 1 continuous value (to control angular adjustment).
   - **Speed**: 1 continuous value (for forward or backward movement).
   - **Shoot**: 1 binary value.

   Total: **3 outputs**.

### Example Summary
Based on these adjustments, your agent’s inputs and outputs might look like this:

- **Inputs (7–9 values)**:
  - Orientation: 1 value
  - Enemy Detection: 1 binary (enemy seen) + 1 distance
  - Obstacle Proximity: 3–5 values (distances in FOV segments)
  - Speed: 1 value

- **Outputs (3 values)**:
  - Rotation: 1 continuous value (to control angle)
  - Speed: 1 continuous value (to set velocity)
  - Shoot: 1 binary value

### Explanation of Behavior
With this setup:
- The agent can adjust its **angle and speed** to navigate toward or away from objects and try to close in on the enemy when visible.
- When the **enemy enters the FOV**, the agent receives the distance information, enabling it to pursue or attempt to align for a shot.
- The agent can **avoid obstacles** based on proximity sensors within the FOV, helping it navigate the environment more effectively.

This input-output structure should allow the agent to learn a policy that balances **navigation, obstacle avoidance, enemy pursuit, and shooting** in a partially observable environment.

<br>
	
[Back to top](#readme)
