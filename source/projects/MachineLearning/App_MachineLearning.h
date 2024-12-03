#ifndef MACHINE_LEARNING_APPLICATION_H
#define MACHINE_LEARNING_APPLICATION_H

#include "framework/EliteInterfaces/EIApp.h"
#include "DynamicQLearning.h"

class App_MachineLearning final : public IApp
{
public:
	//Constructor & Destructor
	App_MachineLearning() = default;
	~App_MachineLearning() override;

	//App Functions
	void Start() override;
	void UpdateUI(float deltaTime);
	void Update(float deltaTime) override;
	void Render(float deltaTime) const override;

private:

	//Datamembers
	DynamicQLearning* m_pDynamicQEnv{nullptr};


	//C++ make the class non-copyable
	App_MachineLearning(const App_MachineLearning&) = delete;
	App_MachineLearning(App_MachineLearning&&) noexcept = delete;
	App_MachineLearning& operator=(const App_MachineLearning&) = delete;
	App_MachineLearning& operator=(App_MachineLearning&&) noexcept = delete;
};
#endif

//Settings for our reinforced learning project
struct SettingsRL
{
	//Settings
	static constexpr bool m_TrainShooting{ true };
	static constexpr bool m_TrainNavigation{ true };
	static constexpr bool m_TrainMoveToItems{ false };

	static constexpr bool m_RespawnFoodEachGen{ true };

	////Stochastic universal sampling
	//static constexpr bool m_UseSUS{ true };
	////Fitness proportionate selection
	//static constexpr bool m_UseFPS{ false };
	////Tournament selection
	//static constexpr bool m_UseTS{ false };
	//
	////Allow only one of these 3 bools to be true at a time
	//static_assert("Exactly one of these booleans must be true" && !(m_UseSUS ? (m_UseFPS || m_UseTS) : (m_UseFPS && m_UseTS)));

	enum class SelectionMethod
	{
		UseSUS, //Stochastic universal sampling
		UseFPS, //Fitness proportionate selection
		UseTS	//Tournament selection
	};
	static constexpr SelectionMethod m_SelectionMethod{ SelectionMethod::UseSUS };
};
