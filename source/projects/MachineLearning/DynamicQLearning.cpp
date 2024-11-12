//=== General Includes ===
#include "stdafx.h"
#include "DynamicQLearning.h"

#include "App_MachineLearning.h"
#include "Population.h"
#include "projects/Movement/SteeringBehaviors/SteeringAgent.h"
#include "projects/Movement/SteeringBehaviors/Steering/SteeringBehaviors.h"


DynamicQLearning::DynamicQLearning(int nrOfFood, int memorySize, int nrOfInputs, int nrOfOutputs, bool bias)
	: m_pPopulation(new Population(50, m_TrimWorldSize, nrOfFood, memorySize, nrOfInputs, nrOfOutputs, bias))
	, m_Enemy(nullptr)
	, m_Wander(nullptr)
{
	if (SettingsRL::m_TrainShooting)
	{
		m_Enemy = new SteeringAgent(1.5f, { 1,0,0,0.5f });
		m_Wander = new Wander();
		m_Enemy->SetSteeringBehavior(m_Wander);
		m_Enemy->SetPosition({ 0, 50 });
	}
}

DynamicQLearning::~DynamicQLearning() 
{
	SAFE_DELETE(m_pPopulation);
	if (SettingsRL::m_TrainShooting)
	{
		SAFE_DELETE(m_Enemy);
		SAFE_DELETE(m_Wander);
	}
}
void DynamicQLearning::Update(const float deltaTime) const
{
	if (SettingsRL::m_TrainShooting)
	{
		m_Enemy->TrimToWorld(m_TrimWorldSize-10);
		m_Enemy->Update(deltaTime);
		m_pPopulation->Update(deltaTime, m_Enemy->GetPosition());
	}
	else
		m_pPopulation->Update(deltaTime);
}

void DynamicQLearning::Render(const float deltaTime) const
{
	m_pPopulation->Render(deltaTime, m_Enemy->GetPosition());
	if (SettingsRL::m_TrainShooting)
		m_Enemy->Render(deltaTime);
}
