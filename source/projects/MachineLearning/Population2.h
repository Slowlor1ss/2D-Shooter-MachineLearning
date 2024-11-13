#pragma once
#include <vector>

class NavigationColliderElement;
class QBot2;
class SteeringAgent;

class Population2
{
public:
	Population2(int size, float worldSize, int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias);
	~Population2();
	void UpdateFitnessProportionateSelection();
	void UpdateTournamentSelection();
	void UpdateSUS();

	Population2(const Population2&) = delete;
	Population2(Population2&&) noexcept = delete;
	Population2& operator=(const Population2&) = delete;
	Population2& operator=(Population2&&) noexcept = delete;

	void Update(float deltaTime, const std::vector<SteeringAgent*>& enemies);
	void Render(float deltaTime, const std::vector<SteeringAgent*>& enemies) const;

private:
	void NormalizeFitness() const;
	float CalculateFitnessSum() const;
	float CalculateFitnessSum(unsigned int from, unsigned int to) const;
	Elite::FMatrix* SelectParentFPS(float sum) const;
	void SelectParentSUS(const float sum) const;
	void UpdateImGui();

	int m_BotInfoIndex{};
	bool m_ShowAllFood{ false };
	int m_Size;
	int m_NrOfFood;
	int m_DeadCounter;
	int m_Generation;
	std::vector<QBot2*> m_Bots;
	std::vector<NavigationColliderElement*> m_vNavigationColliders;
};

