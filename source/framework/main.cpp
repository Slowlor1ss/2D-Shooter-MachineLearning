//Precompiled Header [ALWAYS ON TOP IN CPP]
#include "stdafx.h"

//-----------------------------------------------------------------
// Includes
//-----------------------------------------------------------------
//Application
#include "EliteInterfaces/EIApp.h"
#include "projects/App_Selector.h"

//---------- Registered Applications -----------
#ifdef ActiveApp_Sandbox
	#include "projects/Movement/Sandbox/App_Sandbox.h"
#elif defined(ActiveApp_SteeringBehaviors)
	#include "projects/Movement/SteeringBehaviors/Steering/App_SteeringBehaviors.h"
 #elif defined(ActiveApp_CombinedSteering)
 	#include "projects/Movement/SteeringBehaviors/CombinedSteering/App_CombinedSteering.h"
#elif defined(ActiveApp_Flocking)
	#include "projects/Movement/SteeringBehaviors/Flocking/App_Flocking.h"
#elif defined(ActiveApp_GraphTheory)
	#include "projects/Movement/Pathfinding/GraphTheory/App_GraphTheory.h"
#elif defined(ActiveApp_Astar)
	#include "projects/Movement/Pathfinding/PathfindingAStar/App_PathfindingAStar.h"
#elif defined(ActiveApp_NavGraph)
	#include "projects/Movement/Pathfinding/NavMeshGraph/App_NavMeshGraph.h"
#elif defined(ActiveApp_StateMachine)
	#include "projects/DecisionMaking/FiniteStateMachines/App_AgarioGame.h"
#elif defined(ActiveApp_BehaviourTree)
	#include "projects/DecisionMaking/BehaviorTrees/App_AgarioGame_BT.h"
#elif defined(ActiveApp_InfluenceMap)
	#include "projects/DecisionMaking/InfluenceMaps/App_InfluenceMap.h"
#elif defined(ActiveApp_MachineLearning)
	#include "projects/MachineLearning/App_MachineLearning.h"

#endif


//Hotfix for genetic algorithms project
bool gRequestShutdown = false;

//Main
#undef main //Undefine SDL_main as main
int main(int argc, char* argv[])
{
	auto seed{ (unsigned)time(0) };
	//auto seed{ (unsigned)1731509834 };
	srand(seed);
	std::cout << "Seed: " << seed <<'\n';

	int x{}, y{};
	bool runExeWithCoordinates{ argc == 3 };

	if (runExeWithCoordinates)
	{
		x = stoi(string(argv[1]));
		y = stoi(string(argv[2]));
	}

	try
	{
		//Window Creation
		Elite::WindowParams params;
		EliteWindow* pWindow = new EliteWindow();
		ELITE_ASSERT(pWindow, "Window has not been created.");
		pWindow->CreateEWindow(params);

		if (runExeWithCoordinates)
			pWindow->SetWindowPosition(x, y);

		//Create Frame (can later be extended by creating FrameManager for MultiThreaded Rendering)
		EliteFrame* pFrame = new EliteFrame();
		ELITE_ASSERT(pFrame, "Frame has not been created.");
		pFrame->CreateFrame(pWindow);

		//Create a 2D Camera for debug rendering in this case
		Camera2D* pCamera = new Camera2D(params.width, params.height);
		ELITE_ASSERT(pCamera, "Camera has not been created.");
		DEBUGRENDERER2D->Initialize(pCamera);

		//Create Immediate UI 
		Elite::EImmediateUI* pImmediateUI = new Elite::EImmediateUI();
		ELITE_ASSERT(pImmediateUI, "ImmediateUI has not been created.");
		pImmediateUI->Initialize(pWindow->GetRawWindowHandle());

		//Create Physics
		PHYSICSWORLD; //Boot

		//Start Timer
		TIMER->Start();

#if !defined(ActiveApp_MachineLearning)
		//Application Creation
		IApp* myApp = nullptr;
#endif

#ifdef ActiveApp_Sandbox
		myApp = new App_Sandbox();
#elif defined(ActiveApp_CombinedSteering)
		myApp = new App_CombinedSteering();
#elif defined(ActiveApp_Flocking)
		myApp = new App_Flocking();
#elif defined(ActiveApp_SteeringBehaviors)
		myApp = new App_SteeringBehaviors();
#elif defined(ActiveApp_GraphTheory)
		myApp = new App_GraphTheory();
#elif defined(ActiveApp_Astar)
		myApp = new App_PathfindingAStar();
#elif defined(ActiveApp_NavGraph)
		myApp = new App_NavMeshGraph();
#elif defined(ActiveApp_StateMachine)
		myApp = new App_AgarioGame();
#elif defined(ActiveApp_BehaviourTree)
		myApp = new App_AgarioGame_BT();
#elif defined(ActiveApp_InfluenceMap)
		myApp = new App_InfluenceMap();
#elif defined(ActiveApp_MachineLearning)
		App_MachineLearning* myApp = nullptr;
		myApp = new App_MachineLearning();
#endif
		ELITE_ASSERT(myApp, "Application has not been created.");
		//Boot application
		myApp->Start();

		float timeSinceLastUpdate{};
		constexpr float TimePerFrame = 1.f / 30.f;

		//Application Loop
		while (!pWindow->ShutdownRequested())
		{
			//Timer
			TIMER->Update();
			auto const elapsed = TIMER->GetElapsed();

			timeSinceLastUpdate += elapsed;

			//Window procedure first, to capture all events and input received by the window
			if (!pImmediateUI->FocussedOnUI())
				pWindow->ProcedureEWindow();
			else
				pImmediateUI->EventProcessing();


			//New frame Immediate UI (Flush)
			pImmediateUI->NewFrame(pWindow->GetRawWindowHandle(), elapsed);

			myApp->UpdateUI(elapsed);

			for (size_t i = 0; i < TIMER->GetSpeed(); i++)
			{
				//Update (Physics, App)
				float fixedStep = 1/60.f;
				PHYSICSWORLD->Simulate(fixedStep);
				myApp->Update(fixedStep);
			}

			pCamera->Update();

			if (timeSinceLastUpdate > TimePerFrame)
			{
				timeSinceLastUpdate -= TimePerFrame;
				//Render and Present Frame
				PHYSICSWORLD->RenderDebug();
				myApp->Render(elapsed);
				pFrame->SubmitAndFlipFrame(pImmediateUI);
			}
		}

		//Reversed Deletion
		SAFE_DELETE(myApp);
		SAFE_DELETE(pImmediateUI);
		SAFE_DELETE(pCamera);
		SAFE_DELETE(pFrame);
		SAFE_DELETE(pWindow);

		//Shutdown All Singletons
		PHYSICSWORLD->Destroy();
		DEBUGRENDERER2D->Destroy();
		INPUTMANAGER->Destroy();
		TIMER->Destroy();
	}
	catch (const Elite_Exception& e)
	{
		std::cout << e._msg << " Error: " << std::endl;
#ifdef PLATFORM_WINDOWS
		system("pause");
#endif
		return 1;
	}

	return 0;
}
