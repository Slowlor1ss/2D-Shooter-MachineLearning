#include "stdafx.h"
using namespace Elite;

#include "App_MachineLearning.h"

App_MachineLearning::~App_MachineLearning()
{
	SAFE_DELETE(m_pDynamicQEnv);
}

void App_MachineLearning::Start()
{
	//Initialization
	//----------- CAMERA ------------
	DEBUGRENDERER2D->GetActiveCamera()->SetZoom(100.0f);
	DEBUGRENDERER2D->GetActiveCamera()->SetCenter(Elite::Vector2(0, 0));

	//std::cout << "\n\n\n";
	//FMatrix<> test{7,5};
	//test.Randomize(0,1);
	//test.Print();
	//test.MakeFile("../Matrix.txt", std::ios::trunc);
	//std::cout << "\n\n\n";
	//test.parseFile("../Matrix.txt");
	//test.Print();
	//std::cout << "\n\n\n";

	constexpr int nrOfFood{100 * static_cast<int>(SettingsRL::m_TrainMoveToItems) };
	int orentationStates = 8;
	int moveStates = 8;
	m_pDynamicQEnv = new DynamicQLearning(nrOfFood, 100, 8*3+2, 5*3+2, true);
}

void App_MachineLearning::UpdateUI(const float deltaTime)
{
	//Update that is being called after the physics simulation
	m_pDynamicQEnv->UpdateUI(deltaTime);
}

void App_MachineLearning::Update(const float deltaTime)
{
	//Update that is being called after the physics simulation
	m_pDynamicQEnv->Update(deltaTime);
}

void App_MachineLearning::Render(const float deltaTime) const
{
	m_pDynamicQEnv->Render(deltaTime);
}


