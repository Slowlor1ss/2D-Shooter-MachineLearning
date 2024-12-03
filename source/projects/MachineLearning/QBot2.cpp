#include "stdafx.h"
#include "QBot2.h"
#include <projects/MachineLearning/App_MachineLearning.h>
#include "projects/Movement/SteeringBehaviors/SteeringAgent.h"

// for setting the precision in cout for floating points.
#include <iomanip>
//#define USE_RENFORCMENT_TD

QBot2::QBot2(Vector2 pos, float angle, float radius, float fov, int memorySize) : BaseAgent(radius, { 0.9f, 0.7f, 0.7f, 1.f }, Elite::AGENT_CATEGORY),
	m_MemorySize(memorySize),
	m_StateMatrixMemoryArr(new FMatrix<bfloat>[m_MemorySize]),
	m_ActionMatrixMemoryArr(new FMatrix<bfloat>[m_MemorySize]),
	m_BotBrain(m_NrOfInputs, m_NrOfOutputs),
	m_DeltaBotBrain(m_NrOfInputs, m_NrOfOutputs),
	m_AliveColor(0.9f, 0.7f, 0.7f, 1.f),
	m_DeadColor(.75f, 0.1f, .2f)
{
	currentIndex = 0;

	m_CurrentPos = pos;
	m_PrevPos = pos;
	m_FirstMemPos = pos;
	m_SecMemDist = pos;
	SetPosition(pos);
	SetRotation(angle);
	m_Radius = radius;

	m_FOV = fov;
	m_MaxAngleChange = m_FOV/4;
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

	// We use half of the fov for enemy detection as we only use a small number of sensors thsi will allow smaller gaps
	m_EnemyAngleStep = (fov/4) / (m_InputEnemyDistIndex.size() - 1);
	for (int i = 0; i < m_InputEnemyDistIndex.size(); ++i) {
		float angle = -(fov/4) / 2 + i * m_EnemyAngleStep;
		m_EnemySensorsInRad.push_back(angle);
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
	//m_ActionMatrixMemoryArr[1].Print();

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

	ResetPredictiveEnemyPos();
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
	// 
	//m_Fitness = m_Age + m_EnemiesHit * m_EnemiesHitWeight
	//	- m_EnemiesMisses * m_EnemiesMissedWeight
	//	- m_WallsHit * m_WallsHitWeight
	//	+ m_EnemiesSeen;

	m_Fitness = m_Age + m_EnemyPursuitValue + m_WallsAvoidedValue + m_ExplorationValue + m_RotationCostValue - (m_WallsHit * m_WallsHitWeight);
}

void QBot2::Reset()
{
	m_Health = 100;
	m_Alive = true;
	//m_EnemiesHit = 0;
	//m_EnemiesMisses = 0;
	m_WallsHit = 0;
	m_WallsAvoidedValue = 0;
	m_EnemyPursuitValue = 0;
	m_ExplorationValue = 0;
	m_RotationCostValue = 0;

	m_Age = 0;
	SetPosition(m_StartLocation);

	m_AliveColor = Color(0.9f, 0.7f, 0.7f, 1.f);

	ResetPredictiveEnemyPos();
	SetActive(true);
}

void QBot2::ResetPredictiveEnemyPos()
{
	// For predicitve enemy pos we set teh last one to safe max at the start
	for (size_t i = 0; i < m_InputEnemyDistIndex.size(); i++)
	{
		m_StateMatrixMemoryArr[m_MemorySize - 1].Set(0, m_InputEnemyDistIndex[i], SAFE_MAX);
	}
}

void QBot2::PrintInfo() const
{
	cout << "Died after " << std::setprecision(4) << m_Age << " seconds.\n";
	cout << "Fitness " << std::setprecision(4) << m_Fitness << "\n";
	cout << "Fitness Norm " << std::setprecision(4) << m_FitnessNormalized << "\n";
	cout << "m_EnemyPursuitValue " << std::setprecision(4) << m_EnemyPursuitValue << "\n";
	cout << "m_WallsAvoidedValue " << std::setprecision(4) << m_WallsAvoidedValue << "\n";
	cout << "m_ExplorationValue " << std::setprecision(4) << m_ExplorationValue << "\n";
	cout << "m_WallsHit * m_WallsHitWeight " << std::setprecision(4) << m_WallsHit * m_WallsHitWeight << "\n";
	cout << "m_RotationCostValue " << std::setprecision(4) << m_RotationCostValue << "\n";
	if (SettingsRL::m_TrainNavigation) cout << "Hit " << std::setprecision(4) << m_WallsHit << " walls.\n";
	//if (SettingsRL::m_TrainShooting) cout << "Hit " << std::setprecision(4) << m_EnemiesHit << " enemies.\n";
	//if (SettingsRL::m_TrainShooting) cout << "Missed " << std::setprecision(4) << m_EnemiesMisses << " enemies.\n";
}

std::tuple<float, float, float> QBot2::SelectAction() const
{
	constexpr float epsilon = 0.1f;  // Exploration rate (10% chance to explore)
	float angleAdjustment, speed, shootFlag;

	if ((rand() / static_cast<float>(RAND_MAX)) < epsilon)
	{
		// Exploration: select random actions
		angleAdjustment = randomFloat(-m_MaxAngleChange, m_MaxAngleChange); // Random rotation adjustment
		speed = randomFloat(/*-m_MaxSpeed*/0.f, m_MaxSpeed); // Random speed
		shootFlag = rand() % 2;  // Random shoot (either 0 or 1)
	}
	else 
	{
		// Exploitation: select the action with the highest Q-value
		std::tie(angleAdjustment, speed, shootFlag) = Predict(); // Rename to Get actions from Q or something
		//int maxQIndex = std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
		angleAdjustment = std::clamp(angleAdjustment, -m_MaxAngleChange, m_MaxAngleChange);
		speed = std::clamp(speed, /*-m_MaxSpeed*/0.f, m_MaxSpeed);
		if (isnan(speed))
		{
			m_BotBrain.Print();
			std::cout << "\n-----------------------------------\n";
			m_ActionMatrixMemoryArr[currentIndex].Print();
			__debugbreak();
		}
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

	m_ActionMatrixMemoryArr[currentIndex].Max(r, c, m_OutputRotationIndices.front(), m_OutputRotationIndices.back());
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
//void QBot2::Reinforcement(const float factor, const int memory) const
//{
//	// go back in time, and reinforce (or inhibit) the weights that led to the right/wrong decision.
//	m_DeltaBotBrain.SetAllZero();
//
//#pragma push_macro("disable_min")
//#undef min
//	const int min = std::min(m_MemorySize, memory);
//#pragma pop_macro("disable_min")
//
//	auto oneDivMem{ 1.f / m_MemorySize };
//
//	for (int mi{ 0 }; mi < min; ++mi)
//	{
//		const auto timeFactor = 1 / (1 + Square(mi));
//
//		const auto actualIndex = currentIndex - mi; //% (currentIndex+1);
//		if (actualIndex < 0)
//			return;
//
//		int rMax{};
//		int cMax{};
//		m_ActionMatrixMemoryArr[actualIndex].Max(rMax, cMax);
//
//		const auto scVal = m_StateMatrixMemoryArr[actualIndex].GetNrOfColumns();
//		for (int c{ 0 }; c < scVal; ++c)
//		{
//			if (m_StateMatrixMemoryArr[actualIndex].Get(0, c) > 0)
//			{
//				m_DeltaBotBrain.Add(c, cMax, timeFactor * factor * scVal);
//
//				int rcMax;
//				do
//				{
//					rcMax = randomInt(m_DeltaBotBrain.GetNrOfColumns() - 1);
//				} while (rcMax == cMax);
//
//				m_DeltaBotBrain.Add(c, rcMax, -timeFactor * factor * scVal);
//			}
//
//			m_DeltaBotBrain.ScalarMultiply(oneDivMem);
//			m_BotBrain.FastAdd(m_DeltaBotBrain);
//		}
//	}
//}

int randomIntExcluding(int exclude, int range) {
	int rand;
	do {
		rand = randomInt(range - 1);
	} while (rand == exclude);
	return rand;
}

#if !defined(USE_RENFORCMENT_TD)
void QBot2::Reinforcement(const float factor, const int memory) const
{
	// Initialize delta matrix to zero
	m_DeltaBotBrain.SetAllZero();

	// Determine the number of memory steps to use
	const int maxSteps = std::min(m_MemorySize, memory);
	const float memoryScale = 1.f / m_MemorySize;

	// Loop through memory steps
	for (int step = 0; step < maxSteps; ++step)
	{
		// Compute time decay factor
		const float timeFactor = 1.f / (1.f + Square(step));
		const int actualIndex = currentIndex - step;

		// Ensure the index is valid
		if (actualIndex < 0) break;

		// Find the most impactful action for this memory
		int rMax, cMax;
		m_ActionMatrixMemoryArr[actualIndex].Max(rMax, cMax);

		// Reinforce state-action pairs
		const int columnCount = m_StateMatrixMemoryArr[actualIndex].GetNrOfColumns();
		for (int col = 0; col < columnCount; ++col)
		{
			// Check if the state is active
			if (m_StateMatrixMemoryArr[actualIndex].Get(0, col) <= 0) continue;

			// Strengthen the correct action
			float delta = timeFactor * factor * columnCount;
			m_DeltaBotBrain.Add(col, cMax, delta);

			// Inhibit a random incorrect action
			int randomAction = randomIntExcluding(cMax, m_DeltaBotBrain.GetNrOfColumns());
			m_DeltaBotBrain.Add(col, randomAction, -delta);
		}
	}

	// Apply scaled updates to the bot brain
	m_DeltaBotBrain.ScalarMultiply(memoryScale);
	m_BotBrain.FastAdd(m_DeltaBotBrain);
}
#else
void QBot2::Reinforcement(const float immediateReward, const int memory) const
{
	// Ensure memory size is valid
	const int maxSteps = std::min(m_MemorySize, memory);
	const float gamma = 0.9f; // Discount factor
	const float alpha = 0.1f; // Learning rate

	for (int step = 0; step < std::min(maxSteps, currentIndex + 1); ++step)
	{
		const int actualIndex = currentIndex - step;
		if (currentIndex == 0)
			break; // Prevent underflow

		// Retrieve action and state data from memory
		int rMax, cMax;

		//std::cout << actualIndex << '\n';
		m_ActionMatrixMemoryArr[actualIndex].Max(rMax, cMax); // Action with highest Q-value

		const int stateCount = m_StateMatrixMemoryArr[actualIndex].GetNrOfColumns();

		// For each state
		for (int s = 0; s < stateCount; ++s)
		{
			if (m_StateMatrixMemoryArr[actualIndex].Get(0, s) <= 0) continue;

			// TD Error Calculation
			float qCurrent = m_BotBrain.Get(s, cMax); // Current Q-value
			float maxFutureQ = 0.0f;

			// Estimate maximum future Q-value
			if (step < maxSteps - 1) // Future states available
			{
				const int futureIndex = currentIndex - step - 1;
				if (futureIndex >= 0)
				{
					int futureRMax, futureCMax;
					m_ActionMatrixMemoryArr[futureIndex].Max(futureRMax, futureCMax);
					maxFutureQ = m_BotBrain.Get(s, futureCMax); // Future Q-value estimate
				}
			}

			// Temporal-Difference Error
			float tdError = immediateReward + gamma * maxFutureQ - qCurrent;

			// Update Q-value (TD Learning Update)
			float newQ = qCurrent + alpha * tdError;
			m_BotBrain.Set(s, cMax, newQ);

			// Debugging (Optional): Log TD update
			// std::cout << "TD Update: s=" << s << ", a=" << cMax << ", TD Error=" << tdError << ", New Q=" << newQ << std::endl;
		}
	}
}
#endif

void QBot2::Update(const std::vector<SteeringAgent*>& enemies, float deltaTime)
{
	// Go trough memory
	//currentIndex = (currentIndex + 1) % m_MemorySize;
	//// Purposefully above !m_Alive
	m_PrevPos = m_CurrentPos;
	m_CurrentPos = GetPosition();
	if (isnan(m_CurrentPos.x))
		__debugbreak();
	m_Angle = GetRotation();

	if (!m_Alive)
		return;

	m_Age += deltaTime;
	//m_StateMatrixMemoryArr[currentIndex].SetAll(0.0);
	m_StateMatrixMemoryArr[currentIndex].SetAllZero(); // why is currindex -1???

	const Vector2 dir(cos(m_Angle), sin(m_Angle));
	// TODO: maybe add speedStep back in if needed
	// This allows the bot to choose to move indifferent increments of speed
	//const float speedStep = m_MaxSpeed / (m_NrOfMovementInputs / 3.f);
	//const float angleStep = m_FOV / m_NrOfInputs;

	//if (SettingsRL::m_TrainNavigation) //TODO: add spawn protection and remove m_anglesetp as a varible we pass its a member now bitch
		UpdateNavigation(dir, m_AngleStep, deltaTime);

	//if (SettingsRL::m_TrainShooting)
		UpdateEnemy(enemies, dir, m_AngleStep, deltaTime);

	//Updates the bot / Makes the bot move, rotate, and shoot according to what it decides to do
	UpdateBot(enemies, dir, deltaTime);

	//Make the bot lose 0.1 health per second
	//m_Health -= 0.5f * deltaTime;

	if (m_Health < 0) {
		// update the bot brain, they did something bad
		if constexpr (!SettingsRL::m_TrainShooting)
			Reinforcement(m_NegativeQBig, m_MemorySize);
		m_Alive = false;
		SetActive(false);
		// TODO: could probably do closest enemy?
		// TODO: currently unused
		//float closestPosSqrd{ FLT_MAX };
		//for (SteeringAgent* enemy : enemies)
		//{
		//	float distSqrd = GetPosition().DistanceSquared(enemy->GetPosition());
		//	if (closestPosSqrd > distSqrd)
		//		closestPosSqrd = distSqrd;
		//}
		//m_DistanceClosestEnemyAtDeathSqrd = closestPosSqrd;
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
	//float orientation = m_Angle;
	//bool enemyVisible = false;
	//float enemyDistance = std::numeric_limits<float>::max();

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
	float angle = m_Angle + angleAdjustment * deltaTime;
	const Vector2 newDir(cos(angle), sin(angle));
	SetPositionBullet(position + newDir * (speed * deltaTime), deltaTime);
	SetRotation(angle);

	// Add a cost to rotating as we dont want out agents to be fotating for no reason for now
	// we should look in to this later as this might prewvent sertain sweeping techneques
	float angularVelocity = std::abs(angle - m_Angle);
	Reinforcement(angularVelocity * m_NegativeQSmall, 1);
	m_RotationCostValue += angularVelocity * m_NegativeQ;

	//v/ Update old values /v//
	 
	// Go trough memory
	currentIndex = (currentIndex + 1) % m_MemorySize;

	//m_PrevPos = m_CurrentPos;
	//m_CurrentPos = GetPosition();
	//m_Angle = GetRotation();

	if (currentIndex == 0)
	{
		m_FirstMemPos = GetPosition();
	}

	//

	UpdateQMatrix(deltaTime);

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
	for (size_t i = 0; i < m_InputObstacleProxIndices.size(); i++)
	{
		m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputObstacleProxIndices[i], SAFE_MAX);
	}

	m_WallSeen.reset();

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
			double minDifference = std::abs(m_ObstacleSensorsInRad[0] - angle);

			for (int i = 1; i < m_ObstacleSensorsInRad.size(); ++i) {
				double difference = std::abs(m_ObstacleSensorsInRad[i] - angle);
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

			// We keep the closest wall we've seen
			if (!m_WallSeen.has_value()
				|| m_WallSeen.value().distSqrd > Square(dist))
			{
				m_WallSeen.emplace(WallSeen{ obstacle, m_Angle, Square(dist), obstacle->CalculateNormal(m_CurrentPos) });
			}
		}
		else
		{
			//const int angleIndex = static_cast<int>(std::round(m_InputObstacleProxIndices[0] + (angle / angleStep)));
			/*int closestIndex = 0;
			double minDifference = std::abs(m_ObstacleSensorsInRad[0] - angle);

			for (int i = 1; i < m_ObstacleSensorsInRad.size(); ++i) {
				double difference = std::abs(m_ObstacleSensorsInRad[i] - angle);
				if (difference < minDifference) {
					minDifference = difference;
					closestIndex = i;
				}
			}
			m_StateMatrixMemoryArr[currentIndex].Set(0, closestIndex, FLT_MAX);*/
		}

		m_WallHitCooldown += deltaTime;
		if (m_WallHitCooldown > 1.f)
		{
			//got close to a wall
			if (dist < 5.f + m_Radius) {
				StayedAwayFromWalls = false;
			}
			//hits a wall
			if (dist < 0.5f + m_Radius) 
			{
				//if ((m_Age - deltaTime) > deltaTime) // Add some spawn protection for the first frame
				{
					// TODO: this walls hit value probably needs to be lower like increase by 1 every second and then also change weight
					++m_WallsHit;
					//m_Health -= 1.f * deltaTime;
					//m_Health -= 25.f;
					//Reinforcement(m_NegativeQ, 50);
				}
				break;
			}
			m_WallHitCooldown = 0;
		}

		//m_distCooldown += deltaTime;
		//if(m_distCooldown > 1.f)
		//{
		//	float distance = abs(m_CurrentPos.DistanceSquared(m_PrevPos));
		//	if (distance < 0.06 * deltaTime)
		//	{
		//		// Find a better way to encourage exploration
		//		Reinforcement(m_NegativeQ, m_MemorySize);
		//		m_Health -= 0.01f * deltaTime;
		//	}
		//	else
		//	{
		//		// Maybe do once? we dont want to overload the agent with renforcements
		//		//Reinforcement(m_PositiveQ, m_MemorySize);
		//		m_Health += 0.01f * deltaTime;
		//	}
		//	m_distCooldown = 0;
		//}
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
	//constexpr float worldSize = 25;
	//constexpr float blockSize{ 5.0f };
	//constexpr float hBlockSize{ blockSize / 2.0f };
	//if ((location.x < -worldSize - blockSize || location.x > worldSize + hBlockSize)
	//	|| (location.y < -worldSize - blockSize || location.y > worldSize + hBlockSize))
	//{
	//	m_Health += 0.3f * deltaTime;
	//	m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//}


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
void QBot2::UpdateEnemy(const std::vector<SteeringAgent*>& enemies, const Vector2 dir, const float, float deltaTime)
{
	m_EnemySeen.reset();
	if (SettingsRL::m_TrainShooting)
	{
		//TODO: Make shoot cooldown and based on deltatime
		//TODO: predict eney distance based on currentindex -s
		//m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex, FLT_MAX);
		for (size_t i = 0; i < m_InputEnemyDistIndex.size(); i++)
		{
			int prevIdx = (currentIndex-1) % m_MemorySize;
			if (prevIdx < 0) prevIdx = m_MemorySize + prevIdx; // Prevent underflow
			float prevDist = m_StateMatrixMemoryArr[prevIdx].Get(0, m_InputEnemyDistIndex[i]);
			if (!Elite::AreEqual(prevDist, SAFE_MAX) && !isless(prevDist, m_MaxDistance))
			{
				// We predict the distance of the enemy by default so the agaent can remmeber the position of the enemy
				m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex[i], prevDist+2.f);
			}
			else
			{
				m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex[i], SAFE_MAX);
			}
		}
		//m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemySeenIndex, 0);

		for (const SteeringAgent* enemy : enemies)
		{
			Vector2 enemyPos = enemy->GetPosition();
			float distFromEnemySqr = GetPosition().DistanceSquared(enemyPos);

			bool isEnemyBehindWall = false;
			// TODO: maybe put code in location of m_IsEnemyBehindWall but look in to AngleBetween differences see which one is right
			// First 4 should be outside walls and out shot is pretty long so we skip them (Probably should do this in a better way)
			for (size_t i{ 4 }; i < m_vNavigationColliders.size(); ++i)
			{
				if (m_vNavigationColliders[i]->Intersection(m_CurrentPos, enemyPos))
				{
					isEnemyBehindWall = true;
					break;
				}
			}
			bool isLookingAtEnemy = AreEqual(AngleBetween(dir, enemyPos - m_CurrentPos), 0.f, 0.5f) && !isEnemyBehindWall;


			// Looking at enemy
			if (distFromEnemySqr < Elite::Square(m_MaxDistance))
			{
				const float angle = AngleBetween(dir, enemyPos - m_CurrentPos);
				// Can we see the Enemy?
				if (angle > -m_FOV / 2 && angle < m_FOV / 2 && !isEnemyBehindWall)
				//if (isLookingAtEnemy)
				{
					// Update if theres a closer enemy
					//if (m_StateMatrixMemoryArr[currentIndex].Get(0, m_InputEnemyDistIndex) > distFromEnemySqr)
					//{
					//	//if (m_CurrSpeed > 0.1f)
					//	//{
					//	//	Reinforcement(m_PositiveQ /** 10*/, m_MemorySize);
					//	//	m_Health += 0.01f * deltaTime;
					//	//}
					//	//++m_EnemiesSeen;
					//	//m_Health += 0.01f;

					//	m_EnemySeen.emplace(EnemySeen{ enemy, distFromEnemySqr });

					//	m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemyDistIndex, sqrt(distFromEnemySqr));
					//	//m_StateMatrixMemoryArr[currentIndex].Set(0, m_InputEnemySeenIndex, 1);

					//	//TODO: pass a parameter selected agent trough update so we can print messages only for the selected agent
					//	//cout << "Saw enemy\n";
					//	//m_SeenCounter = 1;
					//}

						int closestIndex = 0;
						double minDifference = std::abs(m_EnemySensorsInRad[0] - angle);

						for (int i = 1; i < m_EnemySensorsInRad.size(); ++i) {
							double difference = std::abs(m_EnemySensorsInRad[i] - angle);
							if (difference < minDifference) {
								minDifference = difference;
								closestIndex = i;
							}
						}
						
						m_StateMatrixMemoryArr[currentIndex].Set(0, closestIndex, sqrt(distFromEnemySqr));

					// We keep the closest wall we've seen
					if (!m_EnemySeen.has_value()
						|| m_EnemySeen.value().distSqrd > distFromEnemySqr)
					{
						m_EnemySeen.emplace(EnemySeen{ enemy, distFromEnemySqr, isLookingAtEnemy });
					}
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

void QBot2::HandleWallQValue(float deltaTime)
{
	// If distance to wall is close and looking at wall renforce neg
// Sees wall?
	//bool closeToWall = false;
	//if (m_WallSeen.has_value() && !m_EnemySeen.has_value())
	//{
	//	WallSeen wallData = m_WallSeen.value();

	//	float minDistSqrd = Square(m_MaxDistance); //Square((1.5f + m_Radius)); // we dont want too get too close
	//	if (wallData.distSqrd < minDistSqrd)
	//	{
	//		//hits a wall
	//		if (wallData.distSqrd < Square(0.5f + m_Radius))
	//		{
	//			Reinforcement(m_NegativeQ, 50);
	//			//m_Health -= 5.;
	//		}

	//		closeToWall = true;
	//		float angle = GetRotation();
	//		const Vector2 newDir(cos(angle), sin(angle));
	//		Vector2 movementVec = m_CurrSpeed * newDir; //{ currPos - prevPos };

	//		Vector2 movementDir = movementVec.GetNormalized();
	//		Vector2 wallDir = wallData.wallNormal.GetNormalized();

	//		// Calculate alignment (cosine of the angle)
	//		float alignment = Dot(movementDir, wallDir);

	//		// Negative reinforcement if moving towards the wall
	//		// (Negative alignment means moving towards the wall)
	//		// We also add a speed requirement to encourage moving away from the wall 
	//		// and not just rotating around in place giving it a false sense of doing good 
	//		if (alignment < 0 || m_CurrSpeed < 1.f)
	//		{
	//			m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//			m_WallsAvoidedValue += (m_NegativeQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (-alignment) * 20;
	//			Reinforcement((m_NegativeQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (-alignment), 10); //TODO: maybe only memory size of 1
	//			m_Health -= -((m_NegativeQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (-alignment));
	//		}
	//		// Positive reinforcement if steering away
	//		// (Positive alignment means moving away)
	//		else
	//		{
	//			m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//			m_WallsAvoidedValue += (m_PositiveQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (alignment) * 20;
	//			Reinforcement((m_PositiveQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (alignment), 10); //TODO: maybe only memory size of 1
	//			m_Health += (m_PositiveQ * 10) * (1.0 - wallData.distSqrd / minDistSqrd) * (alignment);
	//		}
	//	}
	//	else
	//	{
	//		m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//	}

	//	// TODO: Add a strong negative reward for hitting the wall to reinforce the importance of avoidance
	//}
	//else
	//{
	//	m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//}

	// If distance to wall is close and looking at wall renforce neg
	// Sees wall?
	//bool closeToWall = false;
	//if (m_WallSeen.has_value())
	//{
	//	WallSeen wallData = m_WallSeen.value();
	//
	//	float minDistSqrd = Square(m_MaxDistance/10.f); //Square((1.5f + m_Radius)); // we dont want too get too close
	//	if (wallData.distSqrd < minDistSqrd)
	//	{
	//		//hits a wall
	//		if (wallData.distSqrd < Square(0.5f + m_Radius))
	//		{
	//			Reinforcement(m_NegativeQBig, 2);
	//			m_Health -= 1.*deltaTime;
	//		}
	//
	//		closeToWall = true;
	//		float angle = GetRotation();
	//		const Vector2 newDir(cos(angle), sin(angle));
	//		Vector2 movementVec = m_CurrSpeed * newDir; //{ currPos - prevPos };
	//
	//		Vector2 movementDir = movementVec.GetNormalized();
	//		Vector2 wallDir = wallData.wallNormal.GetNormalized();
	//
	//		// Calculate alignment (cosine of the angle)
	//		float alignment = Dot(movementDir, wallDir);
	//
	//		// Moving to wall
	//		if (alignment < 0 || m_CurrSpeed < 1.f)
	//		{
	//			m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//			m_WallsAvoidedValue -= 1.f * deltaTime;
	//			Reinforcement(m_NegativeQ, 2); //TODO: maybe only memory size of 1
	//			//m_Health -= 1.f* deltaTime;
	//		}
	//		// Steering away from wall
	//		else
	//		{
	//			m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//			m_WallsAvoidedValue += 1.f * deltaTime;
	//			Reinforcement(m_PositiveQ, 2); //TODO: maybe only memory size of 1
	//			m_Health += 1.f * deltaTime;
	//		}
	//	}
	//	else
	//	{
	//		m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//	}
	//
	//	// TODO: Add a strong negative reward for hitting the wall to reinforce the importance of avoidance
	//}
	//else
	//{
	//	m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//}

	//bool closeToWall = false;
	//if (m_WallSeen.has_value() /*&& !m_EnemySeen.has_value()*/)
	//{
	//	WallSeen wallData = m_WallSeen.value();
	//
	//	float minDistSqrd = Square(m_MaxDistance / 10.f); //Square((1.5f + m_Radius)); // we dont want too get too close
	//	if (wallData.distSqrd < minDistSqrd)
	//	{
	//		m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//		//hits a wall
	//		if (wallData.distSqrd < Square(.5f + m_Radius))
	//		{
	//			Reinforcement(m_NegativeQBig, 2);
	//			m_Health -= 0.1*deltaTime;
	//			m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//		}
	//
	//		//closeToWall = true;
	//		//float angle = GetRotation();
	//		//const Vector2 newDir(cos(angle), sin(angle));
	//		//Vector2 movementVec = m_CurrSpeed * newDir; //{ currPos - prevPos };
	//
	//		//Vector2 movementDir = movementVec.GetNormalized();
	//		//Vector2 wallDir = wallData.wallNormal.GetNormalized();
	//
	//		// Calculate alignment (cosine of the angle)
	//		//float alignment = Dot(movementDir, wallDir);
	//
	//		// Moving to wall
	//		//if (alignment < 0 /*|| m_CurrSpeed < 1.f*/)
	//		//{
	//			//m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f };
	//			m_WallsAvoidedValue -= 1.f * deltaTime;
	//			Reinforcement(m_NegativeQ, 2); //TODO: maybe only memory size of 1
	//			//m_Health -= 1.f* deltaTime;
	//		//}
	//		// Steering away from wall
	//		//else
	//		//{
	//		//	m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//		//	m_WallsAvoidedValue += 1.f * deltaTime;
	//		//	Reinforcement(m_PositiveQSmall, 2); //TODO: maybe only memory size of 1
	//		//	m_Health += 1.f * deltaTime;
	//		//}
	//	}
	//	else
	//	{
	//		//m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//		m_WallsAvoidedValue += 1.f * deltaTime;
	//		Reinforcement(m_PositiveQSmall, 2); //TODO: maybe only memory size of 1
	//		//m_Health += 1.f * deltaTime;
	//		m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//	}
	//
	//	// TODO: Add a strong negative reward for hitting the wall to reinforce the importance of avoidance
	//}
	//else
	//{
	//	m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//}

	Vector2 prevPos = m_FirstMemPos;// m_CurrentPos;
	Vector2 currPos = GetPosition();
	m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red

	if (m_WallSeen.has_value() /*&& !m_EnemySeen.has_value()*/)
	{
		WallSeen wallData = m_WallSeen.value();
		double currDistSqrd = wallData.wall->DistanceSquaredPointRect(currPos);
		float minDistSqrd = Square(m_MaxDistance / 10.f); //Square((1.5f + m_Radius)); // we dont want too get too close

		// If too close to the wall
		if (currDistSqrd < minDistSqrd)
		{
			// Check if moving towards the wall
			if (isgreater(wallData.distSqrd, currDistSqrd))
			{
				// Moving closer to the wall, reward negativly
				Reinforcement(m_NegativeQ, 2);
				m_WallsAvoidedValue += m_NegativeQ * 10;
				m_Health -= 1.f*deltaTime;
				m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
			}
			else
			{
				// Moving away from the wall, patt on head
				Reinforcement(m_PositiveQ, 2);
				m_WallsAvoidedValue += m_PositiveQ * 30;
			}
		}
	}
}

void QBot2::UpdateQMatrix(float deltaTime)
{
	// Wierd rendering issue if we update variables m_currpos etc too soon se we do this for now
	Vector2 prevPos = m_FirstMemPos;// m_CurrentPos;
	Vector2 currPos = GetPosition();

	if (m_EnemySeen.has_value()) 
	{
		EnemySeen enemyData = m_EnemySeen.value();

		constexpr float minDistSqrd = Square(1.5f); // Minimum safe distance (avoid getting too close)
		//constexpr float maxSpeed = 3.0f;           // Desired max speed when pursuing
		constexpr float safeApproachSpeed = 1.5f;  // Desired speed when close to the enemy

		// Calculate current squared distance to enemy
		double currDistSqrd = enemyData.enemy->GetPosition().DistanceSquared(currPos);

		// If not too close to the enemy
		if (currDistSqrd > minDistSqrd) 
		{
			// Check if moving towards the enemy
			if (isgreater(enemyData.distSqrd, currDistSqrd))// || (enemyData.isLookingAtEnemy && m_CurrSpeed > 1.0f))
			{
				// Moving closer to the enemy, reward positively
				Reinforcement(m_PositiveQ, 2);
				m_EnemyPursuitValue += m_PositiveQ * 30;
				m_Health += 1.f;

				// Is catching up to enemy?
				/*if (isgreater(enemyData.distSqrd, currDistSqrd))
				{
					Reinforcement(m_PositiveQ, 2);
					m_EnemyPursuitValue += m_PositiveQ * 20;
				}*/

				// Check if speed is optimal
				//if (m_CurrSpeed > maxSpeed) {
				//	// Penalize for moving too fast
				//	ReinforceNegative("Too fast while approaching enemy.");
				//}
			}
			else 
			{
				// Moving away from the enemy, penalize
				Reinforcement(m_NegativeQ, 2);
				m_EnemyPursuitValue += m_NegativeQ * 10;
			}
		}
		else {
			// Too close to the enemy
			if (m_CurrSpeed < safeApproachSpeed) 
			{
				// Reward for maintaining a safe speed
				Reinforcement(m_PositiveQ, 2);
				m_EnemyPursuitValue += m_PositiveQ * 10;
			}
			else 
			{
				// Penalize for excessive speed in close proximity
				Reinforcement(m_NegativeQ, 2);
				m_EnemyPursuitValue += m_NegativeQ * 10;
			}

			// Consider additional behavior (e.g., engage or hover near the enemy, maybe doge bullets?)
			//HandleCloseProximityBehavior();
		}

		if (enemyData.isLookingAtEnemy)
		{
			Reinforcement(m_PositiveQSmall, 2);
		}
	}

	HandleWallQValue(deltaTime);

	//m_distCooldown += deltaTime;
	//if (m_distCooldown > 5.)
	//{
	//	// Not doing anything else important like following an enemy or avoiding a wall
	//	if (/*!closeToWall &&*/ !m_EnemySeen.has_value())
	//	{
	//		float distance = currPos.DistanceSquared(m_SecMemDist);
	//		if (distance < Square((m_Radius*2)/*4.3f*/ /** 500 * deltaTime * TIMER->GetSpeed()*/)) // Tweak this value
	//		{
	//			// Find a better way to encourage exploration
	//			Reinforcement(m_NegativeQ, m_MemorySize);
	//			m_Health -= 5.f;
	//			m_ExplorationValue -= 0.1f;
	//			m_AliveColor = { 0.7f, 0.7f, 1.0f, 1.f }; // Blue
	//		}
	//		else
	//		{
	//			// Maybe do once? we dont want to overload the agent with renforcements
	//			Reinforcement(m_PositiveQ, m_MemorySize);
	//			m_Health += 3.f;
	//			m_ExplorationValue += 0.1f;
	//			m_AliveColor = { 0.7f, 1.0f, 0.7f, 1.f }; // Green
	//		}
	//	}
	//	else
	//	{
	//		//m_Health += 0.1f;
	//		m_AliveColor = { 1.f, 0.7f, 0.7f, 1.f }; // Red
	//	}
	//	m_SecMemDist = currPos;
	//	m_distCooldown = 0;
	//}
	
	if (!m_EnemySeen.has_value())
	{
		m_Health -= 1.f * deltaTime;
	}
}

void QBot2::UniformCrossover(QBot2* otherBrain)
{
	m_BotBrain.UniformCrossover(otherBrain->m_BotBrain);
}