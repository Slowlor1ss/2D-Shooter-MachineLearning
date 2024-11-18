/*=============================================================================*/
// Copyright 2021-2022 Elite Engine
// Authors: Matthieu Delaere
/*=============================================================================*/
// LevelElement.h: container for adding static colliders that are flagged as
// navigation blockers
/*=============================================================================*/
#ifndef ELITE_NAVIGATION_COLLIDER
#define ELITE_NAVIGATION_COLLIDER

class NavigationColliderElement final
{
public:
	//--- Constructor & Destructor
	NavigationColliderElement(const Elite::Vector2& position, float width, float height);
	~NavigationColliderElement();

	//--- Functions ---
	void RenderElement();
	Elite::Vector2 CalculateNormal(const Elite::Vector2& pos) const;
	Elite::Vector2 GetClosestPoint(const Elite::Vector2& pos) const;
	float DistancePointRect(const Elite::Vector2& pos) const;
	float DistanceSquaredPointRect(const Elite::Vector2& pos) const;
	bool Intersection(const Elite::Vector2& p1, const Elite::Vector2& p2) const;

private:
	//--- Datamembers ---
	RigidBody* m_pRigidBody = nullptr;
	float m_Width = 0.0f;
	float m_Height = 0.0f;
	Elite::Vector2 m_Position = Elite::ZeroVector2;
	float m_Left;
	float m_Right;
	float m_Top;
	float m_Bottom;
};
#endif