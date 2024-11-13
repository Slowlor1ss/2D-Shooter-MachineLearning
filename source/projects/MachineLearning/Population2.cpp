#include "stdafx.h"
#include "Population2.h"
#include "Food.h"
#include "projects/Shared/NavigationColliderElement.h"
#include <iomanip>
#include <numeric>
#include <execution>
#include "App_MachineLearning.h"
#include "QBot2.h"

Population2::Population2(int size, float worldSize, int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias)
	: m_Size(size)
	, m_NrOfFood(nrOfFood)
	//, m_AllBotsDead(false)
	, m_DeadCounter(0)
	, m_Generation(0)
{
	const float startX = randomFloat(-50.0f, 50.0f);
	const float startY = randomFloat(-50.0f, 50.0f);
	//TODO:create boundaries in app machine learning and pass it to here
	if (SettingsRL::m_TrainNavigation)
	{
		//Create Boundaries
		constexpr float blockSize{ 5.0f };
		constexpr float hBlockSize{ blockSize / 2.0f };
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-worldSize - hBlockSize, 0.f), blockSize, (worldSize + blockSize) * 2.0f));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(worldSize + hBlockSize, 0.f), blockSize, (worldSize + blockSize) * 2.0f));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(0.0f, worldSize + hBlockSize), worldSize * 2.0f, blockSize));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(0.0f, -worldSize - hBlockSize), worldSize * 2.0f, blockSize));

		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-worldSize + (36 / 2.f), -50.0f + (6 / 2.f)), 36, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-worldSize + (76 / 2.f), -28.0f + (6 / 2.f)), 76, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(18.0f + (36 / 2.f), -28.0f + (6 / 2.f)), 36, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-worldSize + (36 / 2.f), -6.0f + (6 / 2.f)), 36, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-35.0f + (36 / 2.f), 15.0f + (6 / 2.f)), 36, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(34.0f + (36 / 2.f), 15.0f + (6 / 2.f)), 36, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(-worldSize + (84 / 2.f), 43.0f + (6 / 2.f)), 84, 3));
		m_vNavigationColliders.push_back(new NavigationColliderElement(Vector2(34.0f + (36 / 2.f), 43.0f + (6 / 2.f)), 36, 3));
	}

	//std::vector<float> foodX{};
	//foodX.reserve(nrOfFood);
	//std::vector<float> foodY{};
	//foodY.reserve(nrOfFood);
	//for (int j = 0; j < nrOfFood; ++j)
	//{
	//	foodX.push_back(randomFloat(-(worldSize - 10), (worldSize - 10)));
	//	foodY.push_back(randomFloat(-(worldSize - 10), (worldSize - 10)));
	//}

	for (int i{ 0 }; i < size; ++i)
	{
		const float startAngle = randomFloat(0, static_cast<float>(M_PI) * 2);

		m_Bots.push_back(new QBot2({ 0.f, 0.f }, 0,
			1.5f, 2 * static_cast<float>(M_PI), memorySize));
		m_Bots.back()->SetObstacles(m_vNavigationColliders);
	}
}

Population2::~Population2()
{
	for (auto bot : m_Bots)
		SAFE_DELETE(bot);

	for (auto pNc : m_vNavigationColliders)
		SAFE_DELETE(pNc);
	m_vNavigationColliders.clear();
}

void Population2::Update(const float deltaTime, const std::vector<SteeringAgent*>& enemies)
{
	//Update UI
	UpdateImGui();
	//Draw circle around selected bot
	DEBUGRENDERER2D->DrawCircle(m_Bots[m_BotInfoIndex]->GetPosition(), 5, { 1,0,0 }, 0.1f);

	// Check if all bods are dead
	if (m_DeadCounter == m_Size) //TODO: i think this runs multiple times per round instead of once...
	{
		++m_Generation;

		switch (SettingsRL::m_SelectionMethod)
		{
			//Stochastic universal sampling
		case SettingsRL::SelectionMethod::UseSUS:
			UpdateSUS();
			break;
			//Fitness proportionate selection
		case SettingsRL::SelectionMethod::UseFPS:
			UpdateFitnessProportionateSelection();
			break;
			//Tournament selection
		case SettingsRL::SelectionMethod::UseTS:
			UpdateTournamentSelection();
			break;
		}

		m_DeadCounter = 0;
		return;
	}

	m_DeadCounter = 0;
	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		// Check if bot is not dead
		if (m_Bots[i]->IsAlive())
		{
			// Update bots
			m_Bots[i]->Update( enemies, deltaTime);
		}
		else
		{
			++m_DeadCounter;
		}
	}

	NormalizeFitness();
}

void Population2::UpdateSUS()
{
	QBot2* BestBot = *std::max_element(m_Bots.begin(), m_Bots.end(), [](const QBot2* x, const QBot2* y)
		{
			return x->GetFitness() < y->GetFitness();
		});
	BestBot->PrintInfo();
	cout << "Generation: " << m_Generation << endl;

	if (m_Generation % 10 == 0)
	{
		const auto BestBrain = BestBot->GetBotBrain();
		(*BestBrain).Print();
		(*BestBrain).MakeFile("../Matrix.txt", std::ios::app);
	}

	const float sum{ CalculateFitnessSum() };
	SelectParentSUS(sum);
	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		//m_Bots[i]->UniformCrossover(m_Bots[randomInt(static_cast<int>(m_Bots.size()))]);
		m_Bots[i]->MutateMatrix(0.1f, 0.001f);
		m_Bots[i]->Reset();
	}
}

void Population2::UpdateTournamentSelection()
{
	QBot2* BestBot = *std::max_element(m_Bots.begin(), m_Bots.end(), [](const QBot2* x, const QBot2* y)
		{
			return x->GetFitness() < y->GetFitness();
		});
	BestBot->PrintInfo();
	cout << "Generation: " << m_Generation << endl;
	const auto BestBrain = BestBot->GetBotBrain();

	if (m_Generation % 10 == 0)
	{
		(*BestBrain).Print();
		(*BestBrain).MakeFile("../Matrix.txt", std::ios::app);
	}

	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		if (m_Bots[i] != BestBot)
			m_Bots[i]->SetBotBrain(BestBrain);

		//m_Bots[i]->UniformCrossover(m_Bots[randomInt(static_cast<int>(m_Bots.size()))]);
		m_Bots[i]->MutateMatrix(0.1f, 0.001f);
		m_Bots[i]->Reset();
	}
}

void Population2::UpdateFitnessProportionateSelection()
{
	QBot2* BestBot = *std::max_element(m_Bots.begin(), m_Bots.end(), [](const QBot2* x, const QBot2* y)
		{
			return x->GetFitness() < y->GetFitness();
		});
	BestBot->PrintInfo();
	cout << "Generation: " << m_Generation << endl;

	if (m_Generation % 10 == 0)
	{
		const auto BestBrain = BestBot->GetBotBrain();
		(*BestBrain).Print();
		(*BestBrain).MakeFile("../Matrix.txt", std::ios::app);
	}

	const float sum{ CalculateFitnessSum() };

	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		m_Bots[i]->SetBotBrain(SelectParentFPS(sum));
		//m_Bots[i]->UniformCrossover(m_Bots[randomInt(static_cast<int>(m_Bots.size()))]);
		m_Bots[i]->MutateMatrix(0.1f, 0.001f);
		m_Bots[i]->Reset();
	}
}

void Population2::UpdateImGui()
{
	//------- UI --------
#ifdef PLATFORM_WINDOWS
#pragma region UI
	{
		//Setup
		constexpr int menuWidth = 200;
		int const width = DEBUGRENDERER2D->GetActiveCamera()->GetWidth();
		int const height = DEBUGRENDERER2D->GetActiveCamera()->GetHeight();
		bool windowActive = true;
		ImGui::SetNextWindowPos(ImVec2(static_cast<float>(width) - menuWidth - 10, 10));
		ImGui::SetNextWindowSize(ImVec2(static_cast<float>(menuWidth), static_cast<float>(height) - 90));
		ImGui::Begin("2D Shooter", &windowActive, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
		ImGui::PushAllowKeyboardFocus(false);
		ImGui::SetWindowFocus();
		ImGui::PushItemWidth(100);
		//Elements
		ImGui::Text("CONTROLS");
		ImGui::Indent();
		ImGui::Unindent();

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::Spacing();

		ImGui::Text("STATS");
		ImGui::Indent();
		ImGui::Text("%.3f ms/frame", 1000.0f / ImGui::GetIO().Framerate);
		ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
		ImGui::Text("%.1f Time Speed", TIMER->GetSpeed());

		if (ImGui::Button("+ Speed"))
		{
			TIMER->SetSpeed(TIMER->GetSpeed() + 1.f);
		}
		ImGui::SameLine();
		if (ImGui::Button("- Speed"))
		{
			float speed = TIMER->GetSpeed() - 1.f;
			if (speed < 1)
				speed = 1;
			TIMER->SetSpeed(speed);
		}

		if (ImGui::Button("++ Speed"))
		{
			TIMER->SetSpeed(TIMER->GetSpeed() + 10.f);
		}
		ImGui::SameLine();
		if (ImGui::Button("-- Speed"))
		{
			float speed = TIMER->GetSpeed() - 10.f;
			if (speed < 1)
				speed = 1;
			TIMER->SetSpeed(speed);
		}

		ImGui::Unindent();

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::Spacing();

		ImGui::Text("Agent Info");
		ImGui::SliderInt("Agent Index", &m_BotInfoIndex, 0, m_Bots.size() - 1);
		ImGui::Text("Walls Hit: %i", m_Bots[m_BotInfoIndex]->GetWallsHit());
		ImGui::Text("Hits: %i", m_Bots[m_BotInfoIndex]->GetHits());
		ImGui::Text("Misses: %i", m_Bots[m_BotInfoIndex]->GetMisses());
		ImGui::Text("Health: %.2f", m_Bots[m_BotInfoIndex]->GetHealth());
		ImGui::Text("Survive Time: %.1f", m_Bots[m_BotInfoIndex]->GetAge());

		//End
		ImGui::PopAllowKeyboardFocus();
		ImGui::End();
	}
#pragma endregion
#endif

}

void Population2::Render(const float deltaTime, const std::vector<SteeringAgent*>& enemies) const
{
	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		if (m_Bots[i]->IsAlive())
		{
			m_Bots[i]->Render(deltaTime, enemies);
		}
	}

	for (const auto collider : m_vNavigationColliders)
	{
		collider->RenderElement();
	}
}

//Struct for calculating the sum i couldn't overload pointers so i had to create a function
struct SumOperatorsQBot2
{
	float operator() (const float left, const QBot2* other) const
	{
		return left + other->GetFitness();
	}
	float operator() (const QBot2* other, const float left) const
	{
		return left + other->GetFitness();
	}
	float operator() (const QBot2* bot, const QBot2* other) const
	{
		return bot->GetFitness() + other->GetFitness();
	}
	float operator() (const float left, const float right) const
	{
		return left + right;
	}
};

void Population2::NormalizeFitness() const
{
	// TODO: possibly save these values in this class might be usefull and then we only calculate it once
	float minFit{ FLT_MAX }; // Coult also just be set to first bot's fit, but oh well
	float maxFit{ FLT_MIN };
	for (QBot2* bot : m_Bots)
	{
		float fit = bot->GetRawFitness();
		if (fit < minFit)
			minFit = fit;
		if (fit > maxFit)
			maxFit = fit;
	}

	if (AreEqual(maxFit,minFit))
	{
		minFit = 0.f;
	}

	for (QBot2* bot : m_Bots)
	{
		float firNorm = (bot->GetRawFitness() - minFit) / (maxFit - minFit);
		if (isnan(firNorm))
			__debugbreak();
		bot->SetNormalizedFitness(firNorm);
	}
}

//https://en.cppreference.com/w/cpp/algorithm/reduce
float Population2::CalculateFitnessSum() const
{
	return std::reduce(std::execution::par, m_Bots.cbegin(), m_Bots.cend(), 0.f, SumOperatorsQBot2());
	//return std::accumulate(m_Bots.cbegin(), m_Bots.cend(), 0.f, Sum);
}
float Population2::CalculateFitnessSum(const unsigned int from, unsigned int to) const
{
	if (to == 0)
		return m_Bots[0]->GetFitness();

	to += 1;//plus one because end is one behind last element

	if (to <= from) {
		std::cout << "to was smaller then or equal to from, putting it to from+1\n";
		to = from + 1;
	}
	if (to > m_Bots.size()) {
		std::cout << to << "to was too big, putting it to max possible value\n";
		to = m_Bots.size();
	}

	return std::reduce(std::execution::par, m_Bots.cbegin() + static_cast<int>(from),
		m_Bots.cbegin() + static_cast<int>(to), 0.f, SumOperatorsQBot2());
}

//Natural Selection
//Code that selects a parent to inherent the brain from,
//agents with higher fitness will have more chance to pass on their brain than agents with lower fitness
//Fitness proportionate selection
FMatrix* Population2::SelectParentFPS(const float sum) const
{
	const float rand{ randomFloat(sum) };

	float runningSum{};

	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
	{
		runningSum += m_Bots[i]->GetFitness();

		if (runningSum >= rand)
			return m_Bots[i]->GetBotBrain();
	}

	//Code should never reach this far if; it does the calculate sum function is wrong
	std::cout << __LINE__ << __FILE__ << "\nCode reached something it shouldnt've";
	return nullptr;
}

//Stochastic universal sampling
#pragma optimize("", off)
void Population2::SelectParentSUS(const float sum) const
{
	std::vector<FMatrix*> matingPool{};
	std::vector<float> pointers{};

	const auto maxFit{ sum };
	const int num = (float)m_Size * 0.2f; //Number of offspring to keep
	const auto dist = maxFit / static_cast<float>(num);
	const auto start = randomFloat(0, dist); // TODO: what is dist is lower then 0?
	for (size_t i{ 0 }; i < num; ++i)
	{
		pointers.push_back(start + static_cast<float>(i) * dist);
	}

	for (size_t i{ 0 }; i < pointers.size(); ++i)
	{
		auto j = 0;
		while (CalculateFitnessSum(0, j) < pointers[i])
		{
			j++;
			//TODO: maybe add if(j == pointers.size()) return;
		}
		matingPool.push_back(m_Bots[j]->GetBotBrain());
	}

	for (size_t i{ 0 }; i < m_Bots.size(); ++i)
		m_Bots[i]->SetBotBrain(matingPool[i % matingPool.size()]);
}
#pragma optimize("", on)
