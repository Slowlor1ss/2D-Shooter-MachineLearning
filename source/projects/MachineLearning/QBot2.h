#pragma once

#include "framework/EliteMath/EMath.h"
#include "framework/EliteRendering/ERenderingTypes.h"
#include "projects/Shared/BaseAgent.h"
#include "projects/Shared/NavigationColliderElement.h"
#include "framework/EliteMath/EVector2.h"

using namespace Elite;
class SteeringAgent;

class QBot2 final : public BaseAgent
{
public:
	QBot2(
		Vector2 pos,
		float angle,
		float radius,
		float fov,
		int memorySize
	);

	virtual ~QBot2() override;

	QBot2(const QBot2&) = delete;
	QBot2(QBot2&&) noexcept = delete;
	QBot2& operator=(const QBot2&) = delete;
	QBot2& operator=(QBot2&&) noexcept = delete;

	virtual void Update(const std::vector<SteeringAgent*>& enemies, float deltaTime);
	virtual void Render(float deltaTime, const std::vector<SteeringAgent*>& enemies);

	bool IsAlive() const;
	void Reset();
	//Fitness from last death
	float GetRawFitness() const { return m_Fitness; }

	float GetFitness() const { return m_FitnessNormalized; }
	void SetNormalizedFitness(float fitness) { m_FitnessNormalized = fitness; }
	void PrintInfo() const;

	void MutateMatrix(float mutationRate, float mutationAmplitude) const;
	// The factor detrmines how we will adjust the weights +pos good -neg bad
	// The memory controls how far back in time we will go when adjusting weights based on past decisions
	void Reinforcement(float factor, int memory) const;
	float CalculateInverseDistance(float realDist) const;
	void UniformCrossover(QBot2* otherBrain);

	FMatrix* GetBotBrain() { return &m_BotBrain; }
	FMatrix GetRawBotBrain() { return m_BotBrain; }
	float GetAge() const { return m_Age; }
	int GetFoodEaten() const { return m_FoodEaten; }
	int GetWallsHit() const { return m_WallsHit; }
	int GetHits() const { return m_EnemiesHit; }
	int GetMisses() const { return m_EnemiesMisses; }
	float GetHealth() const { return m_Health; }

	//void SetBotBrain(const FMatrix& brain) { m_BotBrain.Set(brain); }
	void SetBotBrain(const FMatrix* brain) { m_BotBrain.Set(brain); }
	void SetObstacles(const std::vector<NavigationColliderElement*>& obstacles)
	{
		m_vNavigationColliders = obstacles;
	}

	bool operator<(const QBot2& other) const
	{
		return m_FitnessNormalized < other.m_FitnessNormalized;
	}
	float operator+(const QBot2& other) const
	{
		return m_FitnessNormalized + other.m_FitnessNormalized;
	}
	static float sum(const float left, const QBot2* other)
	{
		return left + other->m_FitnessNormalized;
	}

private:

	void CalculateFitness();

	void UpdateBot(const std::vector<SteeringAgent*>& enemies, Vector2 dir, float deltaTime);
	void UpdateNavigation(const Vector2& dir, const float&angleStep, float deltaTime);
	void UpdateEnemy(const std::vector<SteeringAgent*>& enemies, Vector2 dir, float angleStep, float deltaTime);

	std::tuple<float, float, float> SelectAction() const;
	std::tuple<float, float, float> Predict() const;

	std::vector<NavigationColliderElement*> m_vNavigationColliders;
	float m_Radius;

	Vector2 m_StartLocation;
	Vector2 m_CurrentPos;
	Vector2 m_PrevPos;
	float m_CurrSpeed{0.f};
	float m_Angle{};
	float m_FOV;
	float m_MaxDistance = 50.0f; // View distance

	Color m_AliveColor;
	Color m_DeadColor;

	float m_MaxSpeed{ 30.0f };
	float m_MaxAngleChange{};
	float m_Health{ 100.0f };
	bool m_Alive = true;
	float m_Age{ 0.0f };

	float m_distCooldown{};
	float m_WallHitCooldown{};

	float m_AngleStep;

	// fitness members
	//float m_TimeOfDeath = 0;
	int m_FoodEaten{};
	int m_WallsHit{};
	int m_EnemiesHit{};
	int m_EnemiesMisses{};
	int m_EnemiesSeen{};

	float m_EnemiesHitWeight{ 10.f };
	float m_EnemiesMissedWeight{ 1.f };
	float m_WallsHitWeight{ 0.5f };

	float m_Fitness{};
	float m_FitnessNormalized{};
	float m_DistanceClosestEnemyAtDeathSqrd{};

	/// <summary>
	/// INPUT - OUTPUT VARIABLES
	/// </summary>

	int m_NrOfInputs{10}; // States

	const int m_InputRotationIndex{0};
	const int m_InputSpeedIndex{1};
	const int m_InputEnemySeenIndex{2};
	const int m_InputEnemyDistIndex{3};
	const std::vector<int> m_InputObstacleProxIndices{4,5,6,7,8,9};
	std::vector<float> m_ObstacleSensorsInRad;

	int m_NrOfOutputs{ 8 }; // Actions

	const std::vector<int> m_OutputRotationIndices{ 0,1,2,3,4,5 };
	//std::vector<float> m_RotationsInRad;
	const int m_OutputSpeedIndex{ 6 };
	const int m_OutputShootIndex{ 7 };

	int m_MemorySize;

	// currentIndex stores the information at the current time.
	// instead of swapping or copying matrices, manipulate the currentIndex to
	// go back in time. currentIndex is updated at the start of the update method
	// so that the render method also has the correct currentIndex. But make sure
	// the matrices at 0 are also filled in, otherwise problems.
	int currentIndex{ -1 };
	FMatrix* m_StateMatrixMemoryArr;
	FMatrix* m_ActionMatrixMemoryArr;
	FMatrix m_BotBrain;
	FMatrix m_DeltaBotBrain;
	FMatrix m_MatAngle;
	FMatrix m_MatSpeed;
	int m_SShoot[2]{};

	// Q-factors, enable usage for different learning parameters for positive or for negative reinforcement.
	float m_NegativeQBig{ -0.001f };
	float m_NegativeQ{ -0.0001f };
	float m_NegativeQSmall{ -0.00001f };
	float m_PositiveQSmall{ 0.00001f };
	float m_PositiveQ{ 0.0001f };
	float m_PositiveQBig{ 0.001f };


	int m_CameCloseCounter{ 50 };
	int m_MoveAroundCounter{ 2000 };
	Vector2 m_prevPos;
	int m_WallCheckCounter{ 100 };
	bool m_Shoot{ false };
	int m_ShootCounter{ 10 };
	float m_SeenCounter{ 1 };
	bool m_IsEnemyBehindWall{ false };
};

