#include "stdafx.h"
#include "StringPlusPlus.h"
#include <algorithm>
#include <string>
#include <iterator>

StringPP StringPP::Reverse() const
{
	StringPP str{ *this };
	const size_t n = str.length();

	// Swap character starting from two corners 
	for (size_t i = 0; i < n / 2; i++)
	{
		std::swap(str[i], str[(n - i - 1)]);
	}

	return str;
}

StringPP StringPP::Indent(const int level, const char delimiter) const
{
	return std::string(static_cast<size_t>(level), delimiter) + *this;
}

StringPP StringPP::PrefixFill(const int desiredLength, const char fillChar) const
{
	return std::string(desiredLength - this->length(), fillChar) + *this;
}

StringPP StringPP::SuffixFill(const int desiredLength, const char fillChar) const
{
	return *this + std::string(desiredLength - this->length(), fillChar);
}

StringPP StringPP::ToLower() const
{
	StringPP str{ *this };

	std::transform(str.begin(), str.end(), str.begin(),
		[](const unsigned char c) { return static_cast<const unsigned char>(std::tolower(c)); });

	return str;
}

StringPP StringPP::RemoveSpaces() const
{
	StringPP str{ *this };

	std::size_t spaces = str.find(" ");
	while (spaces != std::string::npos) //delete all the spaces and store without spaces in text
	{
		str.erase(str.find(" "), 1);
		spaces = str.find(" ", spaces + 1);
	}

	return str;
}

StringPP StringPP::RemoveFromText(const std::string& wordOrSomething) const
{
	StringPP str{ *this };

	std::size_t toRemove = str.find(wordOrSomething);
	while (toRemove != std::string::npos)
	{
		str.erase(str.find(wordOrSomething), 1);
		toRemove = str.find(wordOrSomething, toRemove + 1);
	}

	return str;
}

StringPP StringPP::AddBetweenLetters(const std::string& inBetween) const
{
	StringPP str{ *this };
	std::string temp{};
	for (size_t i = 0; i < str.length(); i++)
	{
		temp += str[i] + inBetween;
	}
	str = temp;
	return str;
}

StringPP StringPP::SurroundLetters(const std::string& surround) const
{
	StringPP str{ *this };
	std::string temp{};
	for (size_t i = 0; i < str.length(); i++)
	{
		//Using += To avoid string concatenation resulting in allocation of unnecessary temporary stings
		temp += surround + str[i] += surround;
	}
	str = temp;
	return str;
}



bool StringPP::StartsWith(const StringPP& prefix) const
{
	return this->compare(0, prefix.length(), prefix) == 0;
}

std::vector<StringPP> StringPP::Split(const char delim) const
{
	std::vector<StringPP> returnValue{};
	std::stringstream ss(*this);
	StringPP token;

	while (std::getline(ss, token, delim))
		returnValue.push_back(token);

	return returnValue;
}
