#pragma once

class NavigationColliderElement;
class Food;
class QBot;

class Population
{
public:
	Population(int size, float worldSize, int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias);
	~Population();
	void UpdateFitnessProportionateSelection();
	void UpdateTournamentSelection();
	void UpdateSUS();

	Population(const Population&) = delete;
	Population(Population&&) noexcept = delete;
	Population& operator=(const Population&) = delete;
	Population& operator=(Population&&) noexcept = delete;

	void Update(float deltaTime, Elite::Vector2 enemyPos = {0, 0});
	void Render(float deltaTime, const Elite::Vector2 enemyPos) const;

private:
	void NormalizeFitness() const;
	float CalculateFitnessSum() const;
	float CalculateFitnessSum(unsigned int from, unsigned int to) const;
	Elite::FMatrix* SelectParentFPS(float sum) const;
	void SelectParentSUS(const float sum) const;
	void UpdateImGui();

	int m_BotInfoIndex{};
	bool m_ShowAllFood{false};
	int m_Size;
	int m_NrOfFood;
	int m_DeadCounter;
	int m_Generation;
	std::vector<QBot*> m_Bots;
	std::vector<vector<Food*>> m_Foodstuff;
	std::vector<NavigationColliderElement*> m_vNavigationColliders;
};

