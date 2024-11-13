#include "stdafx.h"
#include "QBot2.h"
#include <projects/MachineLearning/App_MachineLearning.h>
#include "projects/Movement/SteeringBehaviors/SteeringAgent.h"

// for setting the precision in cout for floating points.
#include <iomanip>


QBot2::QBot2(Vector2 pos, float angle, float radius, float fov, int memorySize) : BaseAgent(radius, { 0.9f, 0.7f, 0.7f, 1.f }, Elite::AGENT_CATEGORY),
	m_StateMatrixMemoryArr(new FMatrix[m_MemorySize]),
	m_ActionMatrixMemoryArr(new FMatrix[m_MemorySize]),
	m_BotBrain(m_NrOfInputs, m_NrOfOutputs),
	m_DeltaBotBrain(m_NrOfInputs, m_NrOfOutputs),
	m_MemorySize(memorySize),
	m_AliveColor(0.9f, 0.7f, 0.7f, 1.f),
	m_DeadColor(.75f, 0.1f, .2f)
{
	m_CurrentPos = pos;
	m_PrevPos = pos;
	SetPosition(pos);
	SetRotation(angle);
	m_Radius = radius;

	m_FOV = fov;
	m_MaxAngleChange = m_FOV;
	//m_AngleStep = fov / m_InputObstacleProxIndices.size();
	//for (size_t i = 0; i < m_InputObstacleProxIndices.size(); i++)
	//{
	//	m_ObstacleSensorsInRad.push_back(m_AngleStep * i);
	//}

	if (m_InputObstacleProxIndices.size() < 2) {
		throw std::invalid_argument("Number of sensors must be at least 2 for symmetrical spacing.");
	}

	// Calculate the spacing between each sensor angle
	m_AngleStep = fov / (m_InputObstacleProxIndices.size() - 1);
	for (int i = 0; i < m_InputObstacleProxIndices.size(); ++i) {
		float angle = -fov / 2 + i * m_AngleStep;
		m_ObstacleSensorsInRad.push_back(angle);
	}

	//for (size_t i = 0; i < m_OutputRotationIndices.size(); i++)
	//{
	//	m_RotationsInRad.push_back(m_AngleStep * i);
	//}

	for (int i = 0; i < m_MemorySize; ++i)
	{
		m_StateMatrixMemoryArr[i].Resize(1, m_NrOfInputs);
		m_ActionMatrixMemoryArr[i].Resize(1, m_NrOfOutputs);
	}

	// TODO: Initialize bot brain?

	m_BotBrain.Randomize(-1.0f, 1.0f);
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

QBot2::~QBot2()
{
	delete[] m_ActionMatrixMemoryArr;
	m_ActionMatrixMemoryArr = nullptr;
	delete[] m_StateMatrixMemoryArr;
	m_StateMatrixMemoryArr = nullptr;
}

bool QBot2::IsAlive() const
{
	return m_Alive;
}

void QBot2::CalculateFitness()
{
	// TODO: add misses and distance from enemy (Possibly add optimal distance?)
	//			Add static bools so we calculate fitness based on what we're training
	if constexpr (SettingsRL::m_TrainNavigation && SettingsRL::m_TrainShooting)
		m_Fitness = m_Age + m_EnemiesHit * m_EnemiesHitWeight
		- m_EnemiesMisses * m_EnemiesMissedWeight
		- m_WallsHit * m_WallsHitWeight
		+ m_EnemiesSeen;
	else if constexpr (SettingsRL::m_TrainShooting && !SettingsRL::m_TrainNavigation)
		m_Fitness = m_Age + m_EnemiesHit - m_EnemiesMisses;
	else if constexpr (SettingsRL::m_TrainNavigation && !SettingsRL::m_TrainShooting)
		m_Fitness = m_FoodEaten + m_Age - m_WallsHit; // Maybe walls hit is not that good to use like this possibly need to add a weight?
}

void QBot2::Reset()
{
	m_Health = 100;
	//m_TimeOfDeath = 0;
	m_Alive = true;
	m_FoodEaten = 0;
	m_EnemiesHit = 0;
	m_EnemiesMisses = 0;
	m_WallsHit = 0;

	m_Age = 0;
	SetPosition(m_StartLocation);

	m_AliveColor = Color(0.9f, 0.7f, 0.7f, 1.f);
}

void QBot2::PrintInfo() const
{
	cout << "Died after " << std::setprecision(4) << m_Age << " seconds.\n";
	cout << "Fitness " << std::setprecision(4) << m_Fitness << "\n";
	cout << "Fitness Norm " << std::setprecision(4) << m_FitnessNormalized << "\n";
	if (SettingsRL::m_TrainMoveToItems) cout << "Ate " << std::setprecision(4) << m_FoodEaten << " food.\n";
	if (SettingsRL::m_TrainNavigation) cout << "Hit " << std::setprecision(4) << m_WallsHit << " walls.\n";
	if (SettingsRL::m_TrainShooting) cout << "Hit " << std::setprecision(4) << m_EnemiesHit << " enemies.\n";
	if (SettingsRL::m_TrainShooting) cout << "Missed " << std::setprecision(4) << m_EnemiesMisses << " enemies.\n";
}

std::tuple<float, float, float> QBot2::SelectAction() const
{
	constexpr float epsilon = 0.1f;  // Exploration rate (10% chance to explore)
	float angleAdjustment, speed, shootFlag;

	//if ((rand() / static_cast<float>(RAND_MAX)) < epsilon)
	//{
	//	// Exploration: select random actions
	//	angleAdjustment = randomFloat(-m_MaxAngleChange, m_MaxAngleChange); // Random rotation adjustment
	//	speed = randomFloat(-m_MaxSpeed, m_MaxSpeed); // Random speed
	//	shootFlag = rand() % 2;  // Random shoot (either 0 or 1)
	//}
	//else 
	{
		// Exploitation: select the action with the highest Q-value
		std::tie(angleAdjustment, speed, shootFlag) = Predict(); // Rename to Get actions from Q or something
		//int maxQIndex = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
		angleAdjustment = std::clamp(angleAdjustment, -m_MaxAngleChange, m_MaxAngleChange);
		speed = std::clamp(speed, -m_MaxSpeed, m_MaxSpeed);
		
		// Decode maxQIndex to actions; assuming a discrete encoding for simplicity
		//angleAdjustment = (maxQIndex - m_InputObstacleProxIndices[0]) * m_AngleStep; // DecodeAngleAdjustment
		//speed = DecodeSpeed(maxQIndex);
		//shootFlag = DecodeShoot(maxQIndex);
	}

	return std::make_tuple(angleAdjustment, speed, shootFlag);
}

std::tuple<float, float, float> QBot2::Predict() const
{
	// Calculate action matrix;
	m_StateMatrixMemoryArr[currentIndex].MatrixMultiply(m_BotBrain, m_ActionMatrixMemoryArr[currentIndex]);
	// Squash values to a range of 0 to 1
	m_ActionMatrixMemoryArr[currentIndex].Sigmoid();

	// Extract actions from matrix
	int r, c, QAngle;
	float QSpeed, QShoot;
	float angleAdjustment, speed, shootFlag;

	m_ActionMatrixMemoryArr[currentIndex].Max(r, c, m_OutputRotationIndices.front(), m_OutputRotationIndices.back()); // Maybe output instead of input
	angleAdjustment = m_ActionMatrixMemoryArr[currentIndex].Get(r, c);
	//m_ActionMatrixMemoryArr[currentIndex].Max(r, cAngle2, m_NrOfMovementOutputs * (1 / 3.f), m_NrOfMovementOutputs * (2 / 3.f));
	speed = m_ActionMatrixMemoryArr[currentIndex].Get(0, m_OutputSpeedIndex);
	shootFlag = m_ActionMatrixMemoryArr[currentIndex].Get(0, m_OutputShootIndex);

	//if (QAngle > m_RotationsInRad.size() - 1)
	//	QAngle = m_RotationsInRad.size() - 1;
	//else if (QAngle < 0)
	//	QAngle = 0;

	//angleAdjustment = m_RotationsInRad[QAngle];

	return { angleAdjustment, speed, shootFlag };
}

void QBot2::MutateMatrix(const float mutationRate, const float mutationAmplitude) const
{
	for (int c = 0; c < m_BotBrain.GetNrOfColumns(); ++c)
	{
		for (int r = 0; r < m_BotBrain.GetNrOfRows(); ++r)
		{
			if (randomFloat(0, 1) < mutationRate)
			{
				const float update = randomFloat(-mutationAmplitude, mutationAmplitude);
				const float currentVal = m_BotBrain.Get(r, c);
				m_BotBrain.Set(r, c, currentVal + update);
			}
		}
	}
}

//TODO: optimize instead of calling this a bunch of times update all at once every frame or something
void QBot2::Reinforcement(const float factor, const int memory) const
{
	// go back in time, and reinforce (or inhibit) the weights that led to the right/wrong decision.
	m_DeltaBotBrain.SetAllZero();

#pragma push_macro("disable_min")
#undef min
	const int min = std::min(m_MemorySize, memory);
#pragma pop_macro("disable_min")

	auto oneDivMem{ 1.f / m_MemorySize };

	for (int mi{ 0 }; mi < min; ++mi)
	{
		const auto timeFactor = 1 / (1 + Square(mi));

		const auto actualIndex = currentIndex - mi; //% (currentIndex+1);
		if (actualIndex < 0)
			return;

		int rMax{};
		int cMax{};
		m_ActionMatrixMemoryArr[actualIndex].Max(rMax, cMax);

		const auto scVal = m_StateMatrixMemoryArr[actualIndex].GetNrOfColumns();
		for (int c{ 0 }; c < scVal; ++c)
		{
			if (m_StateMatrixMemoryArr[actualIndex].Get(0, c) > 0)
			{
				m_DeltaBotBrain.Add(c, cMax, timeFactor * factor * scVal);

				int rcMax;
				do
				{
					rcMax = randomInt(m_DeltaBotBrain.GetNrOfColumns() - 1);
				} while (rcMax == cMax);

				m_DeltaBotBrain.Add(c, rcMax, -timeFactor * factor * scVal);
			}

			m_DeltaBotBrain.ScalarMultiply(oneDivMem);
			m_BotBrain.FastAdd(m_DeltaBotBrain);
		}
	}
}

void QBot2::Update(const std::vector<SteeringAgent*>& enemies, float deltaTime)
{
	// Go trough memory
	currentIndex = (currentIndex + 1) % m_MemorySize;
	// Purposefully above !m_Alive
	m_PrevPos = m_CurrentPos;
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

	if (SettingsRL::m_TrainNavigation) //TODO: add spawn protection and remove m_anglesetp as a varible we pass its a member now bitch
		UpdateNavigation(dir, m_AngleStep, deltaTime);

	if (SettingsRL::m_TrainShooting)
		UpdateEnemy(enemies, dir, m_AngleStep, deltaTime);

	//Updates the bot / Makes the bot move, rotate, and shoot according to what it decides to do
	UpdateBot(enemies, dir, deltaTime);

	//Make the bot lose 0.1 health per second
	m_Health -= 0.5f * deltaTime;

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
	//for (const SteeringAgent* enemy : enemies) 
	//{
	//	Vector2 enemyPos = enemy->GetPosition();
	//	float distFromEnemySqr = position.DistanceSquared(enemyPos);

	//	// Within view range?
	//	if (distFromEnemySqr < Elite::Square(m_MaxDistance)) 
	//	{
	//		float angleToEnemy = AngleBetween(dir, enemyPos - position);
	//		bool isLookingAtEnemy = AreEqual(angleToEnemy, 0.f, 0.05f) && !m_IsEnemyBehindWall;

	//		if (isLookingAtEnemy) 
	//		{
	//			enemyVisible = true;
	//			enemyDistance = sqrt(distFromEnemySqr);
	//			break;
	//		}
	//	}
	//}

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
	auto [angleAdjustment, speed, shootFlag] = SelectAction();

	m_CurrSpeed = speed;
	m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputSpeedIndex, speed);
	m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputRotationIndex, angleAdjustment);

	// 4. Execute Movement and Shooting
	m_Angle += angleAdjustment * deltaTime;
	const Vector2 newDir(cos(m_Angle), sin(m_Angle));
	SetPositionBullet(position + newDir * (speed * deltaTime), deltaTime);
	SetRotation(m_Angle);

	// TODO: Update shoot counter
	//if (shootFlag > 0.5f && m_ShootCounter <= 0) 
	//{
	//	if (enemyVisible) {
	//		// Hit detected, apply positive reward
	//		Reinforcement(m_PositiveQBig, m_MemorySize);
	//		++m_EnemiesHit;
	//	}
	//	else {
	//		// Missed, apply negative reward
	//		Reinforcement(m_NegativeQSmall, m_MemorySize);
	//		++m_EnemiesMisses;
	//	}
	//	m_ShootCounter = 10;
	//}

	//if (enemyVisible) {
	//	// Saw enemy, apply positive reward
	//	Reinforcement(m_PositiveQ, m_MemorySize);
	//}

	// 5. Reward and Update RL Model
	//float reward = 0.0f;
	//if (enemyVisible) {
	//	reward += m_PositiveQ; // reward for seeing enemy
	//}
	//else if (/* check if near obstacle */) {
	//	reward -= m_NegativeQSmall; // penalty for being near an obstacle
	//}

	//RLModel->Update(state, angleAdjustment, speed, shootFlag, reward); // Update Q-value or policy
}
#pragma optimize("", off)
void QBot2::UpdateNavigation(const Vector2& dir, const float& angleStep, float deltaTime)
{
	const Vector2& location = m_CurrentPos; //create alias for get position

	bool StayedAwayFromWalls{ true };
	bool NoWallsInFov{ true };
	for (const auto obstacle : m_vNavigationColliders)
	{
		Vector2 obstacleVector = obstacle->GetClosestPoint(location) - (location - dir * m_MaxDistance);
		const float dist = obstacle->DistancePointRect(location); // TODO; maybe use dst sqrd?
		//if (dist > m_MaxDistance) {
		//	continue;
		//}
		obstacleVector *= 1 / dist; // Normalize

		const float angle = AngleBetween(dir, obstacleVector);
		// Can we see the obstacle?
		if (dist < m_MaxDistance && angle > -m_FOV / 2 && angle < m_FOV / 2)
		{
			NoWallsInFov = false;

			int closestIndex = 0;
			double minDifference = std::abs(m_InputObstacleProxIndices[0] - angle);

			for (int i = 1; i < m_InputObstacleProxIndices.size(); ++i) {
				double difference = std::abs(m_InputObstacleProxIndices[i] - angle);
				if (difference < minDifference) {
					minDifference = difference;
					closestIndex = i;
				}
			}

			// Which sensor can see the wall/closest to wall (in case of multiple)
			//const int angleIndex = static_cast<int>(std::round(m_InputObstacleProxIndices[0] + (angle / angleStep))); // TODO: Test This! its quit eimportant for navigation...
			//int accelerationIndex = static_cast<int>(m_NrOfMovementInputs * (2 / 3.f) + ((angle + m_FOV / 2) / speedStep));

			//float invDist = CalculateInverseDistance(dist);
			//float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, angleIndex);
			//if (invDist > currentDist) // why do we do this check?
			//{
				m_StateMatrixMemoryArr[currentIndex].Set(0, closestIndex, dist);
				//m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
			//}

			//float currentDist2 = m_StateMatrixMemoryArr[currentIndex].Get(0, accelerationIndex);
			//if (invDist > currentDist2)
			//	m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
		}
		else
		{
			//const int angleIndex = static_cast<int>(std::round(m_InputObstacleProxIndices[0] + (angle / angleStep)));
			int closestIndex = 0;
			double minDifference = std::abs(m_InputObstacleProxIndices[0] - angle);

			for (int i = 1; i < m_InputObstacleProxIndices.size(); ++i) {
				double difference = std::abs(m_InputObstacleProxIndices[i] - angle);
				if (difference < minDifference) {
					minDifference = difference;
					closestIndex = i;
				}
			}
			m_StateMatrixMemoryArr[currentIndex].Set(0, closestIndex, FLT_MAX);
		}

		m_WallHitCooldown += deltaTime;
		if (m_WallHitCooldown > 1.f)
		{
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
					m_Health -= 1.f * deltaTime;
					Reinforcement(m_NegativeQSmall, 50);
				}
				break;
			}
			m_WallHitCooldown = 0;
		}

		m_distCooldown += deltaTime;
		if(m_distCooldown > 1.f)
		{
			float distance = abs(m_CurrentPos.DistanceSquared(m_PrevPos));
			if (distance < 0.06 * deltaTime)
			{
				Reinforcement(m_NegativeQ, m_MemorySize);
				m_Health -= 0.01f * deltaTime;
			}
			else
			{
				//Reinforcement(m_PositiveQ, m_MemorySize);
				m_Health += 0.01f * deltaTime;
			}
			m_distCooldown = 0;
		}
	}

	// TODO pass these variables from Population::Population
	// Is outside of wall's? If so get back or SUFFER
	//constexpr float worldSize = 100;
	//constexpr float blockSize{ 5.0f };
	//constexpr float hBlockSize{ blockSize / 2.0f };
	//if ((location.x < -worldSize - blockSize || location.x > worldSize + hBlockSize)
	//	|| (location.y < -worldSize - blockSize || location.y > worldSize + hBlockSize))
	//{
	//	m_Health -= 10.f * deltaTime;
	//	m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//}

	//move out of spawn area
	constexpr float worldSize = 25;
	constexpr float blockSize{ 5.0f };
	constexpr float hBlockSize{ blockSize / 2.0f };
	if ((location.x < -worldSize - blockSize || location.x > worldSize + hBlockSize)
		|| (location.y < -worldSize - blockSize || location.y > worldSize + hBlockSize))
	{
		m_Health += 0.3f * deltaTime;
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
#pragma optimize("", on)
void QBot2::UpdateEnemy(const std::vector<SteeringAgent*>& enemies, const Vector2 dir, const float angleStep, float deltaTime)
{
	//const Vector2& location = GetPosition(); //create alias for get position

	//for (const SteeringAgent* enemy : enemies)
	//{
	//	const Vector2 enemyLoc{ enemy->GetPosition() };
	//	Vector2 enemyVector = enemyLoc - (location - dir * 10);
	//	const float dist = (enemyLoc - location).Magnitude();
	//	if (dist > m_MaxDistance) 
	//	{
	//		return;
	//	}
	//	enemyVector *= 1 / dist;


	//	const float angle = AngleBetween(dir, enemyVector);
	//	if (angle > -m_FOV / 2 && angle < m_FOV / 2) {
	//		m_IsEnemyBehindWall = false;
	//		// First 4 should be outside walls and out shot is pretty long so we skip them (Probably should do this in a better way)
	//		for (size_t i{ 4 }; i < m_vNavigationColliders.size(); ++i)
	//		{
	//			if (m_vNavigationColliders[i]->Intersection(location, enemyLoc))
	//			{
	//				m_IsEnemyBehindWall = true;
	//				break;
	//			}
	//		}
	//		if (m_IsEnemyBehindWall)
	//			return;

	//		// TODO: See if this still works correctly now that itsin a for loop over all enemies maybe use "add" istead of "set"?
	//		//		 Probably best to have a look at how we do it with the food vector in UpdateFood
	//		int index = static_cast<int>((angle + m_FOV / 2) / angleStep);
	//		int accelerationIndex = static_cast<int>(m_NrOfMovementInputs * (2 / 3.f) + ((angle + m_FOV / 2) / speedStep));
	//		float invDist = CalculateInverseDistance(dist);
	//		float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, index);
	//		if (invDist > currentDist) 
	//		{
	//			m_StateMatrixMemoryArr[currentIndex].Set(0, index, invDist);
	//			m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
	//		}
	//	}
	//}

	if (SettingsRL::m_TrainShooting)
	{
		//TODO: Make shoot cooldown based on deltatime
		if (m_ShootCounter > 0) {
			--m_ShootCounter;
		}
		if (m_SeenCounter > 0) {
			m_SeenCounter -= deltaTime;
		}

		m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex, FLT_MAX);
		m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemySeenIndex, 0);

		for (const SteeringAgent* enemy : enemies)
		{
			Vector2 enemyPos = enemy->GetPosition();
			float distFromEnemySqr = GetPosition().DistanceSquared(enemyPos);
			// TODO: maybe put code in location of m_IsEnemyBehindWall but look in to AngleBetween differences see which one is right
			bool isLookingAtEnemy = AreEqual(AngleBetween(dir, enemyPos - m_CurrentPos), 0.f, 0.5f) && !m_IsEnemyBehindWall;

			// Looking at enemy
			if (distFromEnemySqr < Elite::Square(m_MaxDistance))
			{
				if (isLookingAtEnemy)
				{
					if (m_CurrSpeed > 0.1f)
					{
						Reinforcement(m_PositiveQ /** 10*/, m_MemorySize);
						m_Health += 0.01f * deltaTime;
					}
					//++m_EnemiesSeen;
					//m_Health += 0.01f;

					m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex, sqrt(distFromEnemySqr));
					m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemySeenIndex, 1);

					//TODO: pass a parameter selected agent trough update so we can print messages only for the selected agent
					//cout << "Saw enemy\n";
					//m_SeenCounter = 1;
				}
				else
				{
					// Empty for now...
				}
				//DEBUGRENDERER2D->DrawDirection(GetPosition(), dir, 1000, { 1,0,0 });
			}
		}
	}
}

void QBot2::UniformCrossover(QBot2* otherBrain)
{
	m_BotBrain.UniformCrossover(otherBrain->m_BotBrain);
}