//=== General Includes ===
#include "stdafx.h"
#include "DynamicQLearning.h"

#include "App_MachineLearning.h"
#include "Population.h"
#include "Population2.h"
#include "projects/Movement/SteeringBehaviors/SteeringAgent.h"
#include "projects/Movement/SteeringBehaviors/Steering/SteeringBehaviors.h"
#include <projects/Movement/SteeringBehaviors/CombinedSteering/CombinedSteeringBehaviors.h>


DynamicQLearning::DynamicQLearning(int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias)
	: m_pPopulation(new Population2(50, m_TrimWorldSize, nrOfFood, memorySize, nrOfInputs, nrOfOutputs, bias))
	, m_Enemies{}
	//, m_Wander(nullptr)
{
	if (SettingsRL::m_TrainShooting)
	{
		//m_Wander = new Pursuit();
		//m_Wander->SetMaxAngleChange(2.*M_PI);
		//m_Wander->SetWanderOffset();
		//m_Wander->SetWanderRadius();

		m_pSeek = new Seek();
		m_pDrunkWander = new Wander();
		m_pDrunkWander->SetWanderOffset(10.f);
		m_pDrunkWander->SetMaxAngleChange(M_PI);;
		vector<BlendedSteering::WeightedBehavior> weightedSteeringBehaviors;
		weightedSteeringBehaviors.push_back({ m_pSeek, 0.2f });
		weightedSteeringBehaviors.push_back({ m_pDrunkWander, 0.8f });
		m_pBlendedSteering = new BlendedSteering(weightedSteeringBehaviors);

		//m_pDrunkAgent = new SteeringAgent();
		//m_pDrunkAgent->SetSteeringBehavior(m_pBlendedSteering);
		//m_pDrunkAgent->SetBodyColor(Elite::Color(1, 0, 0));
		//m_pDrunkAgent->SetAutoOrient(true);


		for (int i = 0; i < m_EnemiesSize; ++i)
		{
			m_Enemies.emplace_back(new SteeringAgent(1.5f, {1,0,0,0.5f}, Elite::ENEMY_CATEGORY));
			m_Enemies[i]->SetSteeringBehavior(m_pBlendedSteering);
			m_Enemies[i]->SetMaxLinearSpeed(2);
			m_Enemies[i]->SetPosition({ Elite::randomFloat(-50.0f, 50.0f), Elite::randomFloat(-50.0f, 50.0f) });
			m_Enemies[i]->SetAutoOrient(true);
		}
	}
}

DynamicQLearning::~DynamicQLearning() 
{
	SAFE_DELETE(m_pPopulation);
	if (SettingsRL::m_TrainShooting)
	{
		for (int i = 0; i < m_EnemiesSize; ++i)
		{
			delete m_Enemies[i];
			m_Enemies[i] = nullptr;
		}

		//SAFE_DELETE(m_Enemy);
		//SAFE_DELETE(m_Wander);

		SAFE_DELETE(m_pDrunkAgent);
		SAFE_DELETE(m_pBlendedSteering);
		SAFE_DELETE(m_pDrunkWander);
		SAFE_DELETE(m_pSeek);
	}
}

void DynamicQLearning::UpdateUI(const float deltaTime) const
{
	m_pPopulation->UpdateUI(deltaTime);
}

void DynamicQLearning::Update(const float deltaTime) const
{
	if (SettingsRL::m_TrainShooting)
	{
		//m_Enemies[1]->TrimToWorld(m_TrimWorldSize - 10);
		for (int i = 0; i < m_EnemiesSize; ++i)
		{
			//m_Enemies[i]->TrimToWorld(m_TrimWorldSize - 5);
			m_Enemies[i]->Update(deltaTime);
		}

		//m_Enemy->TrimToWorld(m_TrimWorldSize-10);
		//m_Enemy->Update(deltaTime);
		m_pPopulation->Update(deltaTime, m_Enemies);
	}
	else
		m_pPopulation->Update(deltaTime, m_Enemies); // Enemies should be emty here
}

void DynamicQLearning::Render(const float deltaTime) const
{
	m_pPopulation->Render(deltaTime, m_Enemies);

	if (SettingsRL::m_TrainShooting)
	{
		for (SteeringAgent* enemy : m_Enemies)
		{
			enemy->Render(deltaTime);
		}
	}
}
