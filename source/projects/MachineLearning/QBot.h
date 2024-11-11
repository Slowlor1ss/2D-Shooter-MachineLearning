#ifndef QBOT_H
#define QBOT_H

#include "framework/EliteMath/EMath.h"
#include "framework/EliteRendering/ERenderingTypes.h"
#include "Food.h"
#include "projects/Shared/BaseAgent.h"
#include "projects/Shared/NavigationColliderElement.h"

class QBot final : public BaseAgent
{
public:
	QBot(float x,
		float y,
		float fov,
		float sFov,
		float angle,
		int memorySize,
		int nrInputs,
		int nrOutputs,
		bool useBias,
		float radius);

	virtual ~QBot() override;

	QBot(const QBot&) = delete;
	QBot(QBot&&) noexcept = delete;
	QBot& operator=(const QBot&) = delete;
	QBot& operator=(QBot&&) noexcept = delete;

	virtual void Update(vector<Food*>& foodList, Vector2 enemyPos, float deltaTime);
	void Render(float deltaTime) override;

	bool IsAlive() const;
	void Reset();
	//Fitness from last death
	float GetRawFitness() const { return m_Fitness; }

	float GetFitness() const { return m_FitnessNormalized; }
	void SetNormalizedFitness(float fitness) { m_FitnessNormalized = fitness; }
	void PrintInfo() const;

	void MutateMatrix(float mutationRate, float mutationAmplitude) const;
	void Reinforcement(float factor, int memory) const;
	float CalculateInverseDistance(float realDist) const;
	void UniformCrossover(QBot* otherBrain);

	FMatrix* GetBotBrain() { return &m_BotBrain; }
	FMatrix GetRawBotBrain() { return m_BotBrain; }
	float GetAge() const { return m_Age; }
	int GetFoodEaten() const { return m_FoodEaten; }

	//void SetBotBrain(const FMatrix& brain) { m_BotBrain.Set(brain); }
	void SetBotBrain(const FMatrix* brain) { m_BotBrain.Set(brain); }
	void SetObstacles(const std::vector<NavigationColliderElement*>& obstacles)
	{
		m_vNavigationColliders = obstacles;
	}

	bool operator<(const QBot& other) const
	{
		return m_FitnessNormalized < other.m_FitnessNormalized;
	}
	float operator+(const QBot& other) const
	{
		return m_FitnessNormalized + other.m_FitnessNormalized;
	}
	static float sum(const float left, const QBot* other)
	{
		return left + other->m_FitnessNormalized;
	}



private:

	void CalculateFitness();

	void UpdateBot(Vector2 enemyPos, Vector2 dir, float deltaTime);
	void UpdateNavigation(const Vector2& dir, const float& angleStep, const float& speedStep, float deltaTime);
	void UpdateFood(std::vector<Food*>& foodList, const Vector2& dir, const float& angleStep);
	void UpdateEnemy(Vector2 enemyPos, Vector2 dir, float angleStep, float speedStep);


	std::vector<NavigationColliderElement*> m_vNavigationColliders;
	float m_Radius;

	Vector2 m_StartLocation;
	float m_Angle;
	float m_FOV;
	float m_MaxDistance = 50.0f;
	float m_SFOV;
	Color m_AliveColor;
	Color m_DeadColor;
	float m_MaxSpeed = 30.0f;
	float m_Health{ 100.0f };
	bool m_Alive = true;
	float m_Age{ 0.0f };

	// fitness members
	//float m_TimeOfDeath = 0;
	int m_FoodEaten{};
	int m_WallsHit{};
	int m_EnemiesHit{};
	int m_EnemiesMisses{};
	float m_Fitness{};
	float m_FitnessNormalized{};
	float m_DistanceEnemyAtDeath{};

	vector<Food*> m_Visible;

	int m_NrOfInputs;
	int m_NrOfMovementInputs{};
	bool m_UseBias;
	int m_NrOfOutputs;
	int m_NrOfMovementOutputs{};
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
	FMatrix m_SAngle;
	FMatrix m_SSpeed;
	int m_SShoot[2];

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
	int m_ShootCounter{10};
	bool m_IsEnemyBehindWall{false};
};
#endif
