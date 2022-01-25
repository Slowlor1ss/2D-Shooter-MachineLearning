#include "stdafx.h"
#include "QBot.h"
#include "Food.h"

// for setting the precision in cout for floating points.
#include <iomanip>

#include "App_MachineLearning.h"

QBot::QBot(float x,
           float y,
           float fov,
           float sFov,
           float angle,
           int memorySize,
           int nrInputs,
           int nrOutputs,
           bool useBias,
           float radius) : BaseAgent(radius),
   m_Radius(radius), m_Location(x, y), m_StartLocation(x, y), m_Angle(angle), m_FOV(fov),
   //m_AliveColor(randomFloat(), randomFloat(), randomFloat()),
   m_SFOV(sFov),
   m_AliveColor(0, 0.6f, 0.4f),
   m_DeadColor(.75f, 0.1f, .2f),
   m_NrOfInputs(nrInputs), //nr of inputs without the shooting inputs
   m_NrOfMovementInputs(nrInputs - 2),
   m_UseBias(useBias), //nr of outputs without the shooting outputs
   m_NrOfOutputs(nrOutputs),
   m_NrOfMovementOutputs(nrOutputs - 2),
   m_MemorySize(memorySize),
   m_StateMatrixMemoryArr(new FMatrix[m_MemorySize]),
   m_ActionMatrixMemoryArr(new FMatrix[m_MemorySize]),
   m_BotBrain(nrInputs + (useBias ? 1 : 0), nrOutputs),
   m_DeltaBotBrain(nrInputs + (useBias ? 1 : 0), nrOutputs),
   m_SAngle(1, m_NrOfMovementOutputs* (2 / 3.f)),
   m_SSpeed(1, m_NrOfMovementOutputs* (1 / 3.f)),
   m_SShoot{},
   m_prevPos(x, y)
{
	const int dividedNumbersOfOutput = (m_NrOfMovementOutputs / 3.f);
	{
		const float start = -m_SFOV / 2.f;
		const float step = m_SFOV / (dividedNumbersOfOutput - 1);
		for (int i = 0; i < dividedNumbersOfOutput; ++i)
		{
			const float value = start + i * step;
			m_SAngle.Set(0, i, value);
		}
		for (int i = 0; i < dividedNumbersOfOutput; ++i)
		{
			const float value = start + i * step;
			m_SAngle.Set(0, dividedNumbersOfOutput+i, value);
		}
	}
	{
		const float start = m_MaxSpeed;
		const float step = m_MaxSpeed / (dividedNumbersOfOutput - 1);
		for (int i = 0; i < dividedNumbersOfOutput; ++i)
		{
			const float value = start - i * step;
			m_SSpeed.Set(0, i, value);
		}
	}
	m_SShoot[0] = static_cast<int>(false);
	m_SShoot[1] = static_cast<int>(true);


	for (int i = 0; i < m_MemorySize; ++i)
	{
		m_StateMatrixMemoryArr[i].Resize(1, m_NrOfInputs + (m_UseBias ? 1 : 0));
		m_ActionMatrixMemoryArr[i].Resize(1, m_NrOfOutputs);
	}
	//m_BotBrain.Randomize(-1.0f, 1.0f);
	m_BotBrain.parseFile("../ReadMatrix.txt");
	if (m_UseBias) {
		m_BotBrain.SetRowAll(m_NrOfInputs, -10.0f);
	}

	//m_BotBrain.Print();

	SetPosition(m_Location);
}

QBot::~QBot()
{
	delete[] m_ActionMatrixMemoryArr;
	m_ActionMatrixMemoryArr = nullptr;
	delete[] m_StateMatrixMemoryArr;
	m_StateMatrixMemoryArr = nullptr;
}

void QBot::Update(vector<Food*>& foodList, const Vector2 enemyPos, const float deltaTime)
{
	currentIndex = (currentIndex + 1) % m_MemorySize;
	if (!m_Alive) {
		return;
	}
	m_Age += deltaTime;

	m_Visible.clear();
	m_StateMatrixMemoryArr[currentIndex].SetAll(0.0);

	const Vector2 dir(cos(m_Angle), sin(m_Angle));
	const float angleStep = m_FOV / (m_NrOfMovementInputs / 3.f);
	const float speedStep = m_MaxSpeed / (m_NrOfMovementInputs / 3.f);
	//const float angleStep = m_FOV / m_NrOfInputs;

	if (SettingsRL::m_TrainMoveToItems)
		UpdateFood(foodList, dir, angleStep);

	if (SettingsRL::m_TrainNavigation)
		UpdateNavigation(dir, angleStep, speedStep, deltaTime);

	if (SettingsRL::m_TrainShooting)
		UpdateEnemy(enemyPos, dir, angleStep, speedStep);

	//Updates the bot / Makes the bot move, rotate, and shoot according to what it decides to do
	UpdateBot(enemyPos, dir, deltaTime);

	//Make the bot lose 10 health per second
	m_Health -= 10 * deltaTime;
	if (m_Health < 0) {
		// update the bot brain, they did something bad
		if constexpr (!SettingsRL::m_TrainShooting)
			Reinforcement(m_NegativeQBig, m_MemorySize);
		m_Alive = false;
		m_DistanceEnemyAtDeath = GetPosition().DistanceSquared(enemyPos);
		CalculateFitness();
	}
}

void QBot::UpdateBot(Vector2 enemyPos, const Vector2 dir, const float deltaTime)
{
	//m_ActionMatrixMemoryArr[currentIndex].Print();
	//m_BotBrain.Print();
	//m_StateMatrixMemoryArr[currentIndex].Print();
	m_StateMatrixMemoryArr[currentIndex].Set(0, m_NrOfInputs, 1); //bias
	m_StateMatrixMemoryArr[currentIndex].MatrixMultiply(m_BotBrain, m_ActionMatrixMemoryArr[currentIndex]);
	m_ActionMatrixMemoryArr[currentIndex].Sigmoid();

	int r, cAngle, cAngle2, cSpeed, cShoot;
	m_ActionMatrixMemoryArr[currentIndex].Max(r, cAngle, 0, m_NrOfMovementOutputs * (1 / 3.f));
	m_ActionMatrixMemoryArr[currentIndex].Max(r, cAngle2, m_NrOfMovementOutputs * (1 / 3.f), m_NrOfMovementOutputs * (2 / 3.f));
	m_ActionMatrixMemoryArr[currentIndex].Max(r, cSpeed, m_NrOfMovementOutputs * (2 / 3.f), m_NrOfMovementOutputs);
	m_ActionMatrixMemoryArr[currentIndex].Max(r, cShoot, m_NrOfMovementOutputs, m_NrOfOutputs);

	//range goes from m_NrOfOutputs/2 to m_NrOfOutputs so the size is = to m_NrOfOutputs/2
	//but our value is m_NrOfOutputs/2 + the index we want
	cSpeed -= m_NrOfMovementOutputs * (2 / 3.f);
	cShoot -= m_NrOfMovementOutputs;

	const float dSpeed = m_SSpeed.Get(0, cSpeed);

	float dAngle = m_SAngle.Get(0, cAngle);
	dAngle += m_SAngle.Get(0, cAngle2);
	dAngle /= 2.f;
	m_Angle += dAngle * deltaTime;

	const Vector2 newDir(cos(m_Angle), sin(m_Angle));
	SetPosition(GetPosition() + newDir * dSpeed * deltaTime);
	m_Location = GetPosition();
	SetRotation(m_Angle);

	if (SettingsRL::m_TrainShooting)
	{
		
		if (m_ShootCounter > 0) {
			--m_ShootCounter;
		}
		
		if (m_SShoot[cShoot] && m_ShootCounter == 0 )
		{
			//only count a hit when the enemy was visible and it was a hit
			if(GetPosition().DistanceSquared(enemyPos) < Elite::Square(m_MaxDistance))
			{
				if (!m_IsEnemyBehindWall && AreEqual(AngleBetween(dir, enemyPos - m_Location), 0.f, 0.05f))
				{
					Reinforcement(m_PositiveQBig * 10, m_MemorySize);
					++m_EnemiesHit;
					m_Health += 1;
					cout << "Hit enemy\n";
				}
				else{
					Reinforcement(m_NegativeQSmall, m_MemorySize);
					//cout << "Missed enemy";
				}
				DEBUGRENDERER2D->DrawDirection(GetPosition(), dir, 1000, { 1,0,0 });
				m_ShootCounter = 10;
			}
		}

	}
}

void QBot::UpdateEnemy(const Vector2 enemyPos, const Vector2 dir, const float angleStep, const float speedStep)
{
	const Vector2& enemyLoc{ enemyPos };
	Vector2 enemyVector = enemyLoc - (m_Location - dir * 10);
	const float dist = (enemyLoc - m_Location).Magnitude();
	if (dist > m_MaxDistance) {
		return;
	}
	enemyVector *= 1 / dist;


	const float angle = AngleBetween(dir, enemyVector);
	if (angle > -m_FOV / 2 && angle < m_FOV / 2) {
		m_IsEnemyBehindWall = false;
		for (size_t i{ 4 }; i < m_vNavigationColliders.size(); ++i)
		{
			if (m_vNavigationColliders[i]->Intersection(m_Location, enemyLoc))
			{
				m_IsEnemyBehindWall = true;
				break;
			}
		}
		if (m_IsEnemyBehindWall)
			return;

		int index = static_cast<int>((angle + m_FOV / 2) / angleStep);
		int accelerationIndex = static_cast<int>(m_NrOfMovementInputs * (2 / 3.f) + ((angle + m_FOV / 2) / speedStep));
		float invDist = CalculateInverseDistance(dist);
		float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, index);
		if (invDist > currentDist) {
			m_StateMatrixMemoryArr[currentIndex].Set(0, index, invDist);
			m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
		}
	}
}

void QBot::UpdateNavigation(const Vector2& dir, const float& angleStep, const float& speedStep, float deltaTime)
{
	bool StayedAwayFromWalls{ true };
	bool NoWallsInFov{ true };
	for (const auto obstacle : m_vNavigationColliders)
	{
		Vector2 obstacleVector = obstacle->GetClosestPoint(m_Location) - (m_Location - dir * 10);
		const float dist = obstacle->DistancePointRect(m_Location);
		if (dist > m_MaxDistance) {
			continue;
		}
		obstacleVector *= 1 / dist;

		const float angle = AngleBetween(dir, obstacleVector);
		if (angle > -m_FOV / 2 && angle < m_FOV / 2) {
			NoWallsInFov = false;
			int angleIndex = static_cast<int>(m_NrOfMovementInputs * (1 / 3.f) + ((angle + m_FOV / 2) / angleStep));
			int accelerationIndex = static_cast<int>(m_NrOfMovementInputs * (2 / 3.f) + ((angle + m_FOV / 2) / speedStep));
			
			float invDist = CalculateInverseDistance(dist);
			float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, angleIndex);
			if (invDist > currentDist)
			{
				m_StateMatrixMemoryArr[currentIndex].Set(0, angleIndex, invDist);
				m_StateMatrixMemoryArr[currentIndex].Set(0, accelerationIndex, invDist);
			}

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

			++m_WallsHit;
			m_Health -= 10.0f * deltaTime;
			Reinforcement(m_NegativeQSmall, 50);
			break;
		}
	}




	if (m_WallCheckCounter > 0) {
		--m_WallCheckCounter;
	}

	if (m_WallCheckCounter == 0)
	{
		if (NoWallsInFov)
			Reinforcement(m_PositiveQ, 50);
		else if (StayedAwayFromWalls || NoWallsInFov)
			Reinforcement(m_PositiveQSmall, 50);

		m_WallCheckCounter = 100;
	}

	if (m_MoveAroundCounter > 0) {
		--m_MoveAroundCounter;
	}

	//TODO: make it framerate independent
	//encourage exploring the whole map and nor rotation around one point
	if (m_MoveAroundCounter == 0) {
		if (GetPosition().DistanceSquared(m_prevPos) < Square(30))
		{
			Reinforcement(m_NegativeQ, m_MemorySize);
			//m_Health -= 10.f;
		}

		m_prevPos = { GetPosition() };
		m_MoveAroundCounter = 2000;
	}
}

void QBot::UpdateFood(std::vector<Food*>& foodList, const Vector2& dir, const float& angleStep)
{
	bool cameClose = false;
	for (Food* food : foodList) {
		if (food->IsEaten()) {
			continue;
		}
		Vector2 foodLoc = food->GetLocation();
		Vector2 foodVector = foodLoc - (m_Location - dir * 10);
		const float dist = (foodLoc - m_Location).Magnitude();
		if (dist > m_MaxDistance) {
			continue;
		}
		foodVector *= 1 / dist;


		const float angle = AngleBetween(dir, foodVector);
		if (angle > -m_FOV / 2 && angle < m_FOV / 2) {
			bool isBehindWall{ false };
			for (size_t i{ 4 }; i < m_vNavigationColliders.size(); ++i)
			{
				if (m_vNavigationColliders[i]->Intersection(m_Location, foodLoc))
				{
					isBehindWall = true;
					break;
				}
			}
			if (isBehindWall)
				continue;

			m_Visible.push_back(food);

			const int index = static_cast<int>((angle + m_FOV / 2) / angleStep);
			const float invDist = CalculateInverseDistance(dist);
			const float currentDist = m_StateMatrixMemoryArr[currentIndex].Get(0, index);
			if (invDist > currentDist) {
				m_StateMatrixMemoryArr[currentIndex].Set(0, index, invDist);
			}
		}
		else if (dist < 10.0f + m_Radius) {
			cameClose = true;
		}


		if (dist < 2.0f + m_Radius) {
			food->Eat();
			m_CameCloseCounter = 50;
			++m_FoodEaten;
			m_Health += 20.0f;
			Reinforcement(m_PositiveQ, 50);
		}
	}

	if (m_CameCloseCounter > 0) {
		--m_CameCloseCounter;
	}

	//Add a negative reinforcement if the bot near misses food (only every 50 frames at max)
	if (cameClose && m_CameCloseCounter == 0) {
		Reinforcement(m_NegativeQ, 50);
		m_CameCloseCounter = 50;
	}
}

void QBot::Render(float deltaTime) {
	//Elite::Vector2 dir(cos(m_Angle), sin(m_Angle));
	Vector2 dir(cos(GetRotation()), sin(GetRotation()));
	//Elite::Vector2 leftVision(cos(m_Angle + m_FOV / 2), sin(m_Angle + m_FOV / 2));
	//Elite::Vector2 rightVision(cos(m_Angle - m_FOV / 2), sin(m_Angle - m_FOV / 2));

	//Elite::Vector2 perpDir(-dir.y, dir.x);

	////Color color = m_DeadColor;
	////if (m_Alive) {
	////	color = m_AliveColor;
	////}

	Color rayColor = { 1, 0, 0 };
	//Color dirRayColor = {0, 1, 0};
	////DEBUGRENDERER2D->DrawSolidCircle(m_Location, 2, dir, c);
	//if (m_Alive) {
	DEBUGRENDERER2D->DrawSegment(m_Location, m_Location + 10 * dir, rayColor);
	//	//DEBUGRENDERER2D->DrawSegment(m_Location - 10 * dir, m_Location + m_MaxDistance * leftVision, rayColor);
	//	//DEBUGRENDERER2D->DrawSegment(m_Location - 10 * dir, m_Location + m_MaxDistance * rightVision, rayColor);
	//}
	////#ifdef _DEBUG
	//	const float angleStep = m_FOV / (m_NrOfInputs / 2);
	//	for (int i = 0; i < m_NrOfInputs/2 + 1; ++i)
	//	{
	//		Elite::Vector2 vision(cos((m_Angle + m_FOV / 2)-angleStep*i), sin((m_Angle + m_FOV / 2)-angleStep*i));
	//		DEBUGRENDERER2D->DrawSegment(m_Location - 10 * dir, m_Location + m_MaxDistance * vision, rayColor);
	//	}
	//
	//	//The angle its moving towards
	//	int r, cAngle, cSpeed;
	//	m_ActionMatrixMemoryArr[currentIndex].Max(r, cAngle, 0, (m_NrOfOutputs / 2) - 1);
	//	m_ActionMatrixMemoryArr[currentIndex].Max(r, cSpeed, m_NrOfOutputs / 2, m_NrOfOutputs);
	//	//range goes from m_NrOfOutputs/2 to m_NrOfOutputs so the size is = to m_NrOfOutputs/2
	//	//but our value is m_NrOfOutputs/2 + the index we want
	//	cSpeed -= m_NrOfOutputs / 2;
	//
	//	const float dSpeed = m_SSpeed.Get(0, cSpeed);
	//	const float dAngle = m_SAngle.Get(0, cAngle);
	//
	//	const Vector2 dDir(cos(dAngle), sin(dAngle));
	//	//DEBUGRENDERER2D->DrawSegment(m_Location - 10 * dir, m_Location + m_MaxDistance * dDir, dirRayColor);
	//
	//	DEBUGRENDERER2D->DrawString(m_Location, to_string(static_cast<int>(m_Health)).c_str());
	//
	//
	//	//if (m_Alive) {
	//	//	for (Food* f : m_Visible) {
	//	//		Vector2 loc = f->GetLocation();
	//	//		DEBUGRENDERER2D->DrawCircle(loc, 2, color, 0.5f);
	//	//	}
	//	//}
	//
	//	// draw the vision
	//	for (int i = 0; i < m_NrOfInputs; ++i)
	//	{
	//
	//		if (m_StateMatrixMemoryArr[currentIndex].Get(0, i) > 0.0f) {
	//			DEBUGRENDERER2D->DrawSolidCircle(m_Location - 2.5 * dir - perpDir * 2.0f * (i - m_NrOfInputs / 2.0f), 1, perpDir, m_AliveColor);
	//		}
	//		else {
	//			DEBUGRENDERER2D->DrawSolidCircle(m_Location - 3.0 * dir - perpDir * 2.0f * (i - m_NrOfInputs / 2.0f), 1, perpDir, m_DeadColor);
	//		}
	//	}
	//
	//	char age[10];
	//	snprintf(age, 10, "%.1f seconds", m_Age);
	//	DEBUGRENDERER2D->DrawString(m_Location + m_MaxDistance * dir, age);
	////#endif
}

bool QBot::IsAlive() const
{
	return m_Alive;
}

void QBot::Reset()
{
	m_Health = 100;
	//m_TimeOfDeath = 0;
	m_Alive = true;
	m_FoodEaten = 0;
	m_EnemiesHit = 0;

	m_Age = 0;
	m_Location = m_StartLocation;
	SetPosition(m_StartLocation);


	//float startx = Elite::randomFloat(-50.0f, 50.0f);
	//float starty = Elite::randomFloat(-50.0f, 50.0f);
	//float startAngle = Elite::randomFloat(0, static_cast<float>(M_PI) * 2);
}

void QBot::CalculateFitness()
{
	m_Fitness = m_FoodEaten + m_Age + m_EnemiesHit /*+ ((10000 - m_WallsHit)/ 1000.f)*/;
}

void QBot::PrintInfo() const
{
	cout << "Died after " << std::setprecision(4) << m_Age << " seconds.\n";
	if (SettingsRL::m_TrainMoveToItems) cout << "Ate " << std::setprecision(4) << m_FoodEaten << " food.\n";
	if (SettingsRL::m_TrainNavigation) cout << "Hit " << std::setprecision(4) << m_WallsHit << " walls.\n";
	if (SettingsRL::m_TrainShooting) cout << "Hit " << std::setprecision(4) << m_EnemiesHit << " enemies.\n";
}

void QBot::MutateMatrix(const float mutationRate, const float mutationAmplitude) const
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

void QBot::Reinforcement(const float factor, const int memory) const
{
	// go back in time, and reinforce (or inhibit) the weights that led to the right/wrong decision.
	m_DeltaBotBrain.SetAll(0);

	//TODO: clean this up
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

		int rMax;
		int cMax;
		m_ActionMatrixMemoryArr[actualIndex].Max(rMax, cMax);

		const auto scVal = m_StateMatrixMemoryArr[actualIndex].GetNrOfColumns();
		for (int c{ 0 }; c < scVal; ++c)
		{
			if (m_StateMatrixMemoryArr[actualIndex].Get(0, c) > 0)
			{
				m_DeltaBotBrain.Add(c, cMax, timeFactor * factor * scVal);

				int rcMax{};
				do
				{
					rcMax = randomInt(m_DeltaBotBrain.GetNrOfColumns() - 1);
				} while (rcMax == cMax);

				m_DeltaBotBrain.Add(c, rcMax, -timeFactor * factor * scVal);
			}

			m_DeltaBotBrain.ScalarMultiply(oneDivMem);
			m_BotBrain.Add(m_DeltaBotBrain);
		}
	}
}

float QBot::CalculateInverseDistance(const float realDist) const
{
	// version 1 
	//return m_MaxDistance - realDist;
	// version 2
	const float nDist = realDist / m_MaxDistance;
	const float invDistSquared = m_MaxDistance / (1 + nDist * nDist);
	return invDistSquared;
}

void QBot::UniformCrossover(QBot* otherBrain)
{
	m_BotBrain.UniformCrossover(otherBrain->m_BotBrain);
}


