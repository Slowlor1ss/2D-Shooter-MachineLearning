#include "stdafx.h"
#include "QBot2.h"
#include <projects/MachineLearning/App_MachineLearning.h>
#include "projects/Movement/SteeringBehaviors/SteeringAgent.h"

// for setting the precision in cout for floating points.
#include <iomanip>


QBot2::QBot2(Vector2 pos, float angle, float radius, float fov, int memorySize) : BaseAgent(radius, { 0.9f, 0.7f, 0.7f, 1.f }, Elite::AGENT_CATEGORY)
{
	SetPosition(pos);
	SetRotation(angle);
	m_Radius = radius;

	m_FOV = fov;
	m_AngleStep = fov / m_InputObstacleProxIndices.size();
	for (size_t i = 0; i < m_InputObstacleProxIndices.size(); i++)
	{
		m_ObstacleSensorsInRad.push_back(m_AngleStep * i);
	}

	for (int i = 0; i < m_MemorySize; ++i)
	{
		m_StateMatrixMemoryArr[i].Resize(1, m_NrOfInputs);
		m_ActionMatrixMemoryArr[i].Resize(1, m_NrOfOutputs);
	}

	// TODO: Initialize bot brain?

	//m_BotBrain.Randomize(-1.0f, 1.0f);
	//if (SettingsRL::m_TrainNavigation && SettingsRL::m_TrainShooting)
	//	m_BotBrain.parseFile("resources/Combined.txt");
	//else if (SettingsRL::m_TrainNavigation)
	//	m_BotBrain.parseFile("resources/Navigation.txt");
	//else if (SettingsRL::m_TrainShooting)
	//	m_BotBrain.parseFile("resources/Shooting.txt");

	/*if (m_UseBias) {
		m_BotBrain.SetRowAll(m_NrOfInputs, -10.0f);
	}*/

	//m_BotBrain.Print();

	
}

std::tuple<float, float, float> QBot2::SelectAction(const std::vector<float>& state)
{
	float epsilon = 0.1f;  // Exploration rate (10% chance to explore)
	float angleAdjustment, speed, shootFlag;

	if ((rand() / static_cast<float>(RAND_MAX)) < epsilon) 
	{
		// Exploration: select random actions
		angleAdjustment = randomFloat(-m_MaxAngleChange, m_MaxAngleChange); // Random rotation adjustment
		speed = randomFloat(-m_MaxSpeed, m_MaxSpeed); // Random speed
		shootFlag = rand() % 2;  // Random shoot (either 0 or 1)
	}
	else 
	{
		// Exploitation: select the action with the highest Q-value
		std::vector<float> qValues = QNetwork->Predict(state); // Q-values for each action
		int maxQIndex = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));

		// Decode maxQIndex to actions; assuming a discrete encoding for simplicity
		angleAdjustment = (maxQIndex - m_InputObstacleProxIndices[0]) * m_AngleStep; // DecodeAngleAdjustment
		speed = DecodeSpeed(maxQIndex);
		shootFlag = DecodeShoot(maxQIndex);
	}

	return std::make_tuple(angleAdjustment, speed, shootFlag);
}

std::tuple<float, float, float> QBot2::Predict(const std::vector<float>& state)
{
	// Calculate action matrix;
	m_StateMatrixMemoryArr[currentIndex].MatrixMultiply(m_BotBrain, m_ActionMatrixMemoryArr[currentIndex]);
	// Squash values to a range of 0 to 1
	m_ActionMatrixMemoryArr[currentIndex].Sigmoid();

	// Extract actions from matrix
	int r, QAngle, QSpeed, QShoot;
	m_ActionMatrixMemoryArr[currentIndex].Max(r, QAngle, m_OutputRotationIndices.front(), m_OutputRotationIndices.back()); // Maybe output instead of input
	//m_ActionMatrixMemoryArr[currentIndex].Max(r, cAngle2, m_NrOfMovementOutputs * (1 / 3.f), m_NrOfMovementOutputs * (2 / 3.f));
	m_ActionMatrixMemoryArr[currentIndex].Max(r, QSpeed, m_NrOfMovementOutputs * (2 / 3.f), m_NrOfMovementOutputs);
	m_ActionMatrixMemoryArr[currentIndex].Max(r, QShoot, m_NrOfMovementOutputs, m_NrOfOutputs); // last 2

	return std::tuple<float, float, float>();
}

void QBot2::Update(const std::vector<SteeringAgent*>& enemies, float deltaTime)
{
	// Go trough memory
	currentIndex = (currentIndex + 1) % m_MemorySize;
	// Purposefully above !m_Alive
	m_CurrentPos = GetPosition();
	m_Angle = GetRotation();

	if (!m_Alive)
		return;

	m_Age += deltaTime;
	m_StateMatrixMemoryArr[currentIndex].SetAll(0.0);

	const Vector2 dir(cos(m_Angle), sin(m_Angle));
	// TODO: maybe add speedStep back in if needed
	// This allows the bot to choose to move indifferent increments of speed
	//const float speedStep = m_MaxSpeed / (m_NrOfMovementInputs / 3.f);
	//const float angleStep = m_FOV / m_NrOfInputs;

	if (SettingsRL::m_TrainNavigation) //TODO: add spawn protection
		UpdateNavigation(dir, m_AngleStep, speedStep, deltaTime);

	if (SettingsRL::m_TrainShooting)
		UpdateEnemy(enemies, dir, m_AngleStep, speedStep);

	//Updates the bot / Makes the bot move, rotate, and shoot according to what it decides to do
	UpdateBot(enemies, dir, deltaTime);

	//Make the bot lose 0.1 health per second
	m_Health -= 0.1f * deltaTime;

	if (m_Health < 0) {
		// update the bot brain, they did something bad
		if constexpr (!SettingsRL::m_TrainShooting)
			Reinforcement(m_NegativeQBig, m_MemorySize);
		m_Alive = false;
		// TODO: could probably do closest enemy?
		// TODO: currently unused
		float closestPosSqrd{ FLT_MAX };
		for (SteeringAgent* enemy : enemies)
		{
			float distSqrd = GetPosition().DistanceSquared(enemy->GetPosition());
			if (closestPosSqrd > distSqrd)
				closestPosSqrd = distSqrd;
		}
		m_DistanceClosestEnemyAtDeathSqrd = closestPosSqrd;
		CalculateFitness();
	}
}

void QBot2::Render(float deltaTime, const std::vector<SteeringAgent*>& enemies)
{
	// This doesnt actually work as when were dead we stop rendering the agent
	// TODO: fix this
	Color color = m_DeadColor;
	if (m_Alive) {
		color = m_AliveColor;
	}
	SetBodyColor(color);
	DEBUGRENDERER2D->DrawSolidCircle(GetPosition(), m_Radius, { 0,0 }, m_BodyColor);

	// TODOL: see if we should use m_CurrentPos or not maybe not for more accurate drawing?
	const Vector2& location = GetPosition(); //create alias for get position
	Vector2 dir(cos(GetRotation()), sin(GetRotation()));
	Color rayColor = { 1, 0, 0 };
	DEBUGRENDERER2D->DrawSegment(location, location + m_MaxDistance * dir, rayColor);
}

void QBot2::UpdateBot(const std::vector<SteeringAgent*>& enemies, Vector2 dir, float deltaTime)
{
	// 1. Get Observations
	Vector2 position = m_CurrentPos;
	float orientation = m_Angle;
	bool enemyVisible = false;
	float enemyDistance = std::numeric_limits<float>::max();

	// Loop through enemies to see if one is within the FOV
	for (const SteeringAgent* enemy : enemies) 
	{
		Vector2 enemyPos = enemy->GetPosition();
		float distFromEnemySqr = position.DistanceSquared(enemyPos);

		// Within view range?
		if (distFromEnemySqr < Elite::Square(m_MaxDistance)) 
		{
			float angleToEnemy = AngleBetween(dir, enemyPos - position);
			bool isLookingAtEnemy = AreEqual(angleToEnemy, 0.f, 0.05f) && !m_IsEnemyBehindWall;

			if (isLookingAtEnemy) 
			{
				enemyVisible = true;
				enemyDistance = sqrt(distFromEnemySqr);
				break;
			}
		}
	}

	// TODO":: do it like this !!!!!
	// Obstacle proximity logic (simplified)
	//float obstacleDistances[3] = { /* calculate distances for FOV segments */ };

	// 2. Create State Vector
	//std::vector<float> state = { 
	//	orientation, static_cast<float>(enemyVisible), enemyDistance, 
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 0), 
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 1), 
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 2),
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 3),
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 4),
	//	m_StateMatrixMemoryArr[currentIndex].Get(0, 5) 
	//};

	// 3. Select Action
	// Pass state to the RL model to get action values
	float angleAdjustment, speed, shootFlag;
	std::tie(angleAdjustment, speed, shootFlag) = RLModel->SelectAction(m_StateMatrixMemoryArr);

	// 4. Execute Movement and Shooting
	m_Angle += angleAdjustment * deltaTime;
	Vector2 newDir(cos(m_Angle), sin(m_Angle));
	SetPositionBullet(position + newDir * speed * deltaTime, deltaTime);
	SetRotation(m_Angle);

	// TODO: Update shoot counter
	if (shootFlag > 0.5f && m_ShootCounter <= 0) 
	{
		if (enemyVisible) {
			// Hit detected, apply positive reward
			Reinforcement(m_PositiveQBig, m_MemorySize);
			++m_EnemiesHit;
		}
		else {
			// Missed, apply negative reward
			Reinforcement(m_NegativeQSmall, m_MemorySize);
			++m_EnemiesMisses;
		}
		m_ShootCounter = 10;
	}

	// 5. Reward and Update RL Model
	float reward = 0.0f;
	if (enemyVisible) {
		reward += m_PositiveQ; // reward for seeing enemy
	}
	else if (/* check if near obstacle */) {
		reward -= m_NegativeQSmall; // penalty for being near an obstacle
	}

	RLModel->Update(state, angleAdjustment, speed, shootFlag, reward); // Update Q-value or policy
}

void QBot2::UpdateNavigation(const Vector2& dir, const float& angleStep, const float& speedStep, float deltaTime)
{
	const Vector2& location = m_CurrentPos; //create alias for get position

	bool StayedAwayFromWalls{ true };
	bool NoWallsInFov{ true };
	for (const auto obstacle : m_vNavigationColliders)
	{
		Vector2 obstacleVector = obstacle->GetClosestPoint(location) - (location - dir * m_MaxDistance);
		const float dist = obstacle->DistancePointRect(location); // TODO; maybe use dst sqrd?
		if (dist > m_MaxDistance) {
			continue;
		}
		obstacleVector *= 1 / dist; // Normalize

		const float angle = AngleBetween(dir, obstacleVector);
		// Can we see the obstacle?
		if (angle > -m_FOV / 2 && angle < m_FOV / 2) 
		{
			NoWallsInFov = false;
			// Which sensor can see the wall/closest to wall (in case of multiple)
			int angleIndex = static_cast<int>(std::round(m_InputObstacleProxIndices[0] + (angle / angleStep))); // TODO: Test This! its quit eimportant for navigation...
			//int accelerationIndex = static_cast<int>(m_NrOfMovementInputs * (2 / 3.f) + ((angle + m_FOV / 2) / speedStep));

			//float invDist = CalculateInverseDistance(dist);
			//float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, angleIndex);
			//if (invDist > currentDist) // why do we do this check?
			//{
				m_StateMatrixMemoryArr[currentIndex].Set(0, angleIndex, dist);
				m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
			//}

			//float currentDist2 = m_StateMatrixMemoryArr[currentIndex].Get(0, accelerationIndex);
			//if (invDist > currentDist2)
			//	m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
		}

		//got close to a wall
		if (dist < 5.f + m_Radius) {
			StayedAwayFromWalls = false;
		}
		//hits a wall
		if (dist < 0.5f + m_Radius) {
			if ((m_Age - deltaTime) > deltaTime) // Add some spawn protection for the first frame
			{
				// TODO: this wallshit value probably needs to be lower like increase by 1 every second and then also change weight
				++m_WallsHit;
				m_Health -= 5.f * deltaTime;
				Reinforcement(m_NegativeQSmall, 50);
			}
			break;
		}
	}

	// TODO pass these variables from Population::Population
	// Is outside of wall's? If so get back or SUFFER
	constexpr float worldSize = 100;
	constexpr float blockSize{ 5.0f };
	constexpr float hBlockSize{ blockSize / 2.0f };
	if ((location.x < -worldSize - blockSize || location.x > worldSize + hBlockSize)
		|| (location.y < -worldSize - blockSize || location.y > worldSize + hBlockSize))
	{
		m_Health -= 10.f * deltaTime;
		m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	}


	//if (m_WallCheckCounter > 0) {
	//	--m_WallCheckCounter;
	//}

	//if (m_WallCheckCounter == 0)
	//{
	//	if (NoWallsInFov)
	//		Reinforcement(m_PositiveQ, 50);
	//	else if (StayedAwayFromWalls || NoWallsInFov)
	//		Reinforcement(m_PositiveQSmall, 50);

	//	m_WallCheckCounter = 100;
	//}

	//if (m_MoveAroundCounter > 0) {
	//	--m_MoveAroundCounter;
	//}

	////TODO: make it framerate independent
	////encourage exploring the whole map and nor rotation around one point
	//if (m_MoveAroundCounter == 0) {
	//	if (GetPosition().DistanceSquared(m_prevPos) < Square(30))
	//	{
	//		Reinforcement(m_NegativeQSmall, m_MemorySize);
	//		//m_Health -= 10.f;
	//	}

	//	m_prevPos = { GetPosition() };
	//	m_MoveAroundCounter = 2000;
	//}
}
