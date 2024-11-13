#ifndef DYNAMIC_Q_LEARNING
#define DYNAMIC_Q_LEARNING

class Wander;
class SteeringAgent;
class Population;
class Population2;

class DynamicQLearning final
{
public:
	DynamicQLearning(int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias );
	~DynamicQLearning();

	DynamicQLearning(const DynamicQLearning&) = delete;
	DynamicQLearning(DynamicQLearning&&) noexcept = delete;
	DynamicQLearning& operator=(const DynamicQLearning&) = delete;
	DynamicQLearning& operator=(DynamicQLearning&&) noexcept = delete;

	void Update(float deltaTime) const;
	void Render(float deltaTime) const;

	//QBot* GetQBot() const { return m_pQBot; }
	//QBot* GetQBot2() const { return m_pQBot2; }

private:

	//Population* m_pPopulation;
	Population2* m_pPopulation;
	static constexpr int m_EnemiesSize = 10;
	std::vector<SteeringAgent*> m_Enemies;
	Wander* m_Wander;

	//--Level--
	static constexpr float m_TrimWorldSize = 150.f;

	//int m_MemorySize;
	//int m_NrOfInputs;
	//int m_NrOfOutputs;
	//bool m_UseBias;

	// currentIndex stores the information at the current time.
	// instead of swapping or copying matrices, manipulate the currentIndex to
	// go back in time.
	//int currentIndex = 0;
	//QBot* m_pQBot{ nullptr };
	//QBot* m_pQBot2{ nullptr };

	// environment
	//vector<Food*> m_Foodstuff;
};

#endif
