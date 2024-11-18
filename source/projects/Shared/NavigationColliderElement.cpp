//=== General Includes ===
#include "stdafx.h"
#include "NavigationColliderElement.h"

//=== Constructor & Destructor ===
NavigationColliderElement::NavigationColliderElement(const Elite::Vector2& position, float width, float height) :
	m_Position(position),
	m_Width(width),
	m_Height(height),
	m_Left{ m_Position.x - (m_Width * 0.5f) },
	m_Right{ m_Position.x + (m_Width * 0.5f) },
	m_Top{ m_Position.y + (m_Height * 0.5f) },
	m_Bottom{ m_Position.y - (m_Height * 0.5f) }
{
	//Create Rigidbody
	const Elite::RigidBodyDefine define = Elite::RigidBodyDefine(0.01f, 0.1f, Elite::eStatic, false);
	const Transform transform = Transform(position, Elite::ZeroVector2);
	m_pRigidBody = new RigidBody(define, transform, PhysicsFlags::NavigationCollider, false);
	m_pRigidBody->SetMass(FLT_MAX);

	//Add shape
	Elite::EPhysicsBoxShape shape;
	shape.width = width;
	shape.height = height;
	m_pRigidBody->AddShape(&shape);

	m_pRigidBody->AddCollisionFiltering(Elite::WALL_CATEGORY);
}

NavigationColliderElement::~NavigationColliderElement()
{
	SAFE_DELETE(m_pRigidBody);
}

//=== Functions ===
void NavigationColliderElement::RenderElement()
{
	//Do Nothing
}

// Assumes we cant rotate our colliders!
Elite::Vector2 NavigationColliderElement::CalculateNormal(const Elite::Vector2& pos) const
{
	// Identify which edge the point lies on
	if (pos.x < m_Left && pos.y > m_Top) // top-left corner
	{
		return { -1, 1 }; // Normal pointing diagonally from the corner
	}
	else if (pos.x > m_Right && pos.y > m_Top) // top-right corner
	{
		return { 1, 1 }; // Normal pointing diagonally from the corner
	}
	else if (pos.x < m_Left && pos.y < m_Bottom) // bottom-left corner
	{
		return { -1, -1 }; // Normal pointing diagonally from the corner
	}
	else if (pos.x > m_Right && pos.y < m_Bottom) // bottom-right corner
	{
		return { 1, -1 }; // Normal pointing diagonally from the corner
	}
	else if (pos.y > m_Top) // top side
	{
		return { 0, 1 }; // Normal pointing up
	}
	else if (pos.x > m_Right) // right side
	{
		return { 1, 0 }; // Normal pointing right
	}
	else if (pos.y < m_Bottom) // bottom side
	{
		return { 0, -1 }; // Normal pointing down
	}
	else if (pos.x < m_Left) // left side
	{
		return { -1, 0 }; // Normal pointing left
	}

	throw std::invalid_argument("failed all checks");
	return {};
}

//https://math.stackexchange.com/questions/356792/how-to-find-nearest-point-on-line-of-rectangle-from-anywhere
Elite::Vector2 NavigationColliderElement::GetClosestPoint(const Elite::Vector2& pos) const
{
	if (pos.x < m_Left && pos.y > m_Top) //top left corner
	{
		return { m_Left, m_Top };
	}
	else if (pos.x > m_Right && pos.y > m_Top) //top right corner
	{
		return { m_Right,m_Top };
	}
	else if (pos.x < m_Left && pos.y < m_Bottom) //bottom left corner
	{
		return { m_Left,m_Bottom };
	}
	else if (pos.x > m_Right && pos.y < m_Bottom) //bottom right corner
	{
		return { m_Right,m_Bottom };
	}
	else if (pos.y > m_Top) //top side
	{
		return { pos.x,m_Top };
	}
	else if (pos.x > m_Right) //right side
	{
		return { m_Right,pos.y };
	}
	else if (pos.y < m_Bottom) //bottom side
	{
		return { pos.x,m_Bottom };
	}
	else if (pos.x < m_Left) //left side
	{
		return { m_Left,pos.y };
	}

	std::cerr << "failed all checks in GetClosestPoint function in" << __FILE__ << '\n';
	return {};
}

float NavigationColliderElement::DistancePointRect(const Elite::Vector2& pos) const
{
	const auto line = pos-GetClosestPoint(pos);
	return sqrt(line.x*line.x+line.y*line.y);

	//auto dist = pos - m_Position;
	//dist.x = abs(dist.x);
	//dist.y = abs(dist.y);
	//const float dist_x = Dot(dist, { 1,0 }) - (m_Width * 0.5f);
	//const float dist_y = Dot(dist, { 0,1 }) - (m_Height * 0.5f);
	//return sqrtf(dist_x * dist_x + dist_y * dist_y);
}

float NavigationColliderElement::DistanceSquaredPointRect(const Elite::Vector2& pos) const
{
	const auto line = pos - GetClosestPoint(pos);
	return (line.x * line.x + line.y * line.y);

	//auto dist = pos - m_Position;
	//dist.x = abs(dist.x);
	//dist.y = abs(dist.y);
	//const float dist_x = dist.x - (m_Width * 0.5f);
	//const float dist_y = dist.y - (m_Height * 0.5f);
	//return dist_x * dist_x + dist_y * dist_y;
}

bool NavigationColliderElement::Intersection(const Elite::Vector2& p1, const Elite::Vector2& p2) const
{
	//const Elite::Line line{ p1, p2 };
	//if (m_Right < line.p1.x) {
	//	// player is not intersecting line
	//	return false;
	//}

	//if (m_Left > line.p2.x) {
	//	// player is not intersecting line
	//	return false;
	//}

	//if (m_Bottom < line.p2.y && m_Top > line.p2.y) {
	//	// bottom of player is below line 0, and player is intersecting line
	//	return true;
	//}

	//// player is not intersecting line
	//return false;

	const Elite::Line line{ p1, p2 };

	if ((line.p1.x <= m_Left && line.p2.x <= m_Left) || (line.p1.y <= m_Bottom && line.p2.y <= m_Bottom) || (line.p1.x >= m_Right && line.p2.x >= m_Right) || (line.p1.y >= m_Top && line.p2.y >= m_Top))
		return false;

	return true;

//	float x1{ (m_Left - line.p1.x) / (line.p2.x - line.p1.x) };
//	float x2{ (m_Left + m_Width - line.p1.x) / (line.p2.x - line.p1.x) };
//	float y1{ (m_Bottom - line.p1.y) / (line.p2.y - line.p1.y) };
//	float y2{ (m_Bottom + m_Height - line.p1.y) / (line.p2.y - line.p1.y) };
//
//#pragma push_macro("disable_min")
//#undef min
//	using std::max; using std::min;
//	float tMin{ max(min(x1,x2),min(y1,y2)) };
//	float tMax{ min(max(x1,x2), max(y1,y2)) };
//#pragma pop_macro("disable_min")
//	if (tMin > tMax)
//	{
//		return false;
//	}
//	if (tMin > 0 || tMax < 1)
//	{
//		return true;
//	}
//
//	return false;
}
