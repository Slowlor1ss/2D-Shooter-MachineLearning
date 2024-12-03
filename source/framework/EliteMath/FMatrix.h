/*=============================================================================*/
// Copyright 2021-2022 Elite Engine
// Authors: Koen Samyn
/*=============================================================================*/
// FMatrix.cpp: FMatrix class
/*=============================================================================*/
#ifndef ELITE_MATH_FMatrix
#define	ELITE_MATH_FMatrix

#undef max

#include <random>

#include "projects/Shared/Utils_General.h"
#include <immintrin.h>  // AVX/AVX2 support

namespace Elite 
{

	template <typename T>
	concept Arithmetic = std::is_arithmetic_v<T>;

	template<Arithmetic T = float>
	class FMatrix
	{
	private:
		T* m_Data{};
		int m_Rows{}, m_Columns{};
		int m_Size{};
		int rcToIndex(int r, int c) const
		{
			return c * m_Rows + r;
		}
	public:
		FMatrix(): m_Data(nullptr) {}
		FMatrix(int rows, int columns): 
			m_Rows(rows),
			m_Columns(columns),
			m_Data(new T[rows * columns]),
			m_Size(rows * columns)
		{}

		virtual ~FMatrix()
		{
			delete[] m_Data;
			m_Data = nullptr;
		}

		void Resize(const int nrOfRows, const int nrOfColumns)
		{
			m_Rows = nrOfRows;
			m_Columns = nrOfColumns;
			m_Size = m_Rows * m_Columns;
			m_Data = new T[m_Size];
		}
#pragma optimize("", off)
		void Set(const int row, const int column, const T value) const
		{
			const int index = rcToIndex(row, column);
			if (index > -1 && index < m_Size) 
			{
				m_Data[index] = value;
			}
			else 
			{
				printf("Wrong index! [%d, %d]\n", row, column);
				__debugbreak();
			}
		}
		void SetAllZero() const
		{
			std::memset(m_Data, T{}, m_Size * sizeof(T));
		}
		//	void SetAll(const T value) const
		//{
		//	for (int i = 0; i < m_Size; ++i)
		//	{
		//		m_Data[i] = value;
		//	}
		//}
		void SetAll(const T value) const // Test this code better
		{
			int size = m_Size;
			__m256 val = _mm256_set1_ps(value);  // Create a 256-bit vector filled with `value`

			int i = 0;
			for (; i <= size - 8; i += 8)
			{
				_mm256_storeu_ps(&m_Data[i], val);  // Store the value in 8 consecutive positions
			}

			// Fill any remaining elements with the scalar loop in case m_Size is not a multiple of 8
			for (; i < size; ++i)
			{
				m_Data[i] = value;
			}
		}
#pragma optimize("", on)
		void SetRowAll(const int row, const T value) const
		{
			for (int c = 0; c < m_Columns; ++c) 
			{
				Set(row, c, value);
			}
		}
#pragma optimize("", off)
		void Add(const int row, const int column, const T toAdd) const
		{
			const int index = rcToIndex(row, column);
			if (index > -1 && index < m_Size) {
				m_Data[index] += toAdd;
			}
			else 
			{
				printf("Wrong index! [%d, %d]\n", row, column);
				__debugbreak();
			}
		}
#pragma optimize("", on)

		T Get(const int row, const int column) const
		{
			const int index = rcToIndex(row, column);
			if (index > -1 && index < m_Size) {
				return m_Data[index];
			}
			else {
				__debugbreak();
				return -1;
			}
		}

		void Randomize(const T min, const T max) const
		{
#if _DEBUG
			if (min == max) throw std::invalid_argument("min and max cannot be equal to prevent division by zero.");
#endif

			for (int i = 0; i < m_Size; ++i)
			{
				T r = min + static_cast <T> (rand()) / (static_cast <float> (RAND_MAX / (max - min)));
				m_Data[i] = r;
			}
		}

		int GetNrOfRows() const
		{
			return m_Rows;
		}
		int GetNrOfColumns() const
		{
			return m_Columns;
		}
		void MatrixMultiply(const FMatrix<T>& op2, const FMatrix<T>& result) const
		{
			const int maxRows = min(GetNrOfRows(), result.GetNrOfRows());
			const int maxColumns = min(op2.GetNrOfColumns(), result.GetNrOfColumns());

			for (int c_row = 0; c_row < maxRows; ++c_row)
			{
				for (int c_column = 0; c_column < maxColumns; ++c_column)
				{
					T sum = 0;
					for (int index = 0; index < GetNrOfColumns(); ++index)
					{
						sum += Get(c_row, index) * op2.Get(index, c_column);
					}
					result.Set(c_row, c_column, sum);
				}
			}
		}
		void ScalarMultiply(const T scalar) const
		{
			for (int i = 0; i < m_Size; ++i) 
			{
				m_Data[i] *= scalar;
			}
		}

		void UniformCrossover(FMatrix<T>& other)
		{
			for (int i = 0; i < m_Size; ++i)
			{
				if (randomInt(2)) // 50/50 percent chance 
				{
					std::swap(m_Data[i], other.m_Data[i]);
				}
			}
		}

		void Copy(const FMatrix<T>& other) const
		{
			const int maxRows = min(GetNrOfRows(), other.GetNrOfRows());
			const int maxColumns = min(GetNrOfColumns(), other.GetNrOfColumns());

			for (int c_row = 0; c_row < maxRows; ++c_row) {
				for (int c_column = 0; c_column < maxColumns; ++c_column) {
					const T oVal = other.Get(c_row, c_column);
					Set(c_row, c_column, oVal);
				}
			}
		}

		//void Set(const FMatrix<T>& other) const
		//{
		//	if (other.m_Size > m_Size)
		//	{
		//		printf("Different sizes!\n");
		//		return;
		//	}

		//	for (int i = 0; i < m_Size; ++i)
		//	{
		//		//const T value = 0;//*other.m_Data;
		//		//Set(0, i, *other.m_Data);
		//		m_Data[i] = *other.m_Data;
		//	}
		//}

		void Set(const FMatrix<T>* other) const
		{
			if (other->m_Size > m_Size)
			{
				printf("Different sizes!\n");
				return;
			}

			for (int i = 0; i < m_Size; ++i)
			{
				//const T value = 0;//*other.m_Data;
				//Set(0, i, *other.m_Data);
				m_Data[i] = other->m_Data[i];
			}
		}

		//void Add(const FMatrix<T>& other) const
		//{
		//	int maxRows = min(GetNrOfRows(), other.GetNrOfRows());
		//	int maxColumns = min(GetNrOfColumns(), other.GetNrOfColumns());

		//	for (int c_row = 0; c_row < maxRows; ++c_row) {
		//		for (int c_column = 0; c_column < maxColumns; ++c_column) {
		//			T oVal = other.Get(c_row, c_column);
		//			T thisVal = Get(c_row, c_column);
		//			Set(c_row, c_column, thisVal + oVal);
		//		}
		//	}
		//}
		void Add(const FMatrix<T>& other) const
		{
			const int maxRows = std::min(GetNrOfRows(), other.GetNrOfRows());
			const int maxColumns = std::min(GetNrOfColumns(), other.GetNrOfColumns());

			// Directly access m_Data and avoid using Get/Set if possible
			for (int c_row = 0; c_row < maxRows; ++c_row) 
			{
				for (int c_column = 0; c_column < maxColumns; ++c_column) 
				{
					const int indexThis = rcToIndex(c_row, c_column);    // Index for the current matrix
					const int indexOther = other.rcToIndex(c_row, c_column); // Index for the other matrix

					// Assuming rcToIndex is correct and returns valid indices
					if (indexThis >= 0 && indexThis < m_Size && indexOther >= 0 && indexOther < other.m_Size) {
						m_Data[indexThis] += other.m_Data[indexOther]; // Add directly to m_Data
					}
					else {
						printf("Index out of bounds! [%d, %d]\n", c_row, c_column);
					}
				}
			}
		}
		void FastAdd(const FMatrix<T>& other) const
		{
			const int maxRows = std::min(GetNrOfRows(), other.GetNrOfRows());
			const int maxColumns = std::min(GetNrOfColumns(), other.GetNrOfColumns());
			const int stride = GetNrOfColumns(); // Assuming a row-major layout

			// Using pointers for direct access
			T* __restrict dataPtr = m_Data; // Assuming m_Data is a T array or pointer
			const T* __restrict otherDataPtr = other.m_Data;

			for (int c_row = 0; c_row < maxRows; ++c_row)
			{
				int rowOffset = c_row * stride;
				for (int c_column = 0; c_column < maxColumns; ++c_column)
				{
					// Calculate index without calling rcToIndex
					int indexThis = rowOffset + c_column;
					dataPtr[indexThis] += otherDataPtr[indexThis];
				}
			}
		}

		void Subtract(const FMatrix<T>& other) const
		{
			int maxRows = min(GetNrOfRows(), other.GetNrOfRows());
			int maxColumns = min(GetNrOfColumns(), other.GetNrOfColumns());

			for (int c_row = 0; c_row < maxRows; ++c_row) 
			{
				for (int c_column = 0; c_column < maxColumns; ++c_column) 
				{
					T oVal = other.Get(c_row, c_column);
					T thisVal = Get(c_row, c_column);
					Set(c_row, c_column, thisVal - oVal);
				}
			}
		}
		void Sigmoid() const
		{
			for (int i = 0; i < m_Size; ++i)
			{
#ifdef  _DEBUG
				const T val = m_Data[i];
				T res = 1 / (1 + exp(-val));
				if (isnan(static_cast<float>(res)))
				{
					Print();
					__debugbreak();
				}
				m_Data[i] = res;
#else
				const T val = m_Data[i];
				m_Data[i] = static_cast<T>(1 / (1 + exp(-val)));
#endif //  _DEBUG
			}
		}

		T Sum() const
		{
			T sum = 0;
			for (int i = 0; i < m_Size; ++i)
			{
				sum += m_Data[i];
			}
			return sum;
		}
		T Dot(const FMatrix<T>& op2) const
		{
			int mR = min(GetNrOfRows(), op2.GetNrOfRows());
			int mC = min(GetNrOfColumns(), op2.GetNrOfColumns());

			T dot = 0;
			for (int c_row = 0; c_row < mR; ++c_row) {
				for (int c_column = 0; c_column < mC; ++c_column) {
					T v1 = Get(c_row, c_column);
					T v2 = Get(c_row, c_column);
					dot += v1 * v2;
				}
			}
			return dot;
		}
		T Max() const
		{
			T max = -std::numeric_limits<T>::max();
			for (int c_row = 0; c_row < m_Rows; ++c_row) {
				for (int c_column = 0; c_column < m_Columns; ++c_column) {
					T value = Get(c_row, c_column);
					if (value > max) {
						max = value;
					}

				}
			}
			return max;
		}
		T Max(int& r, int& c) const
		{
			T max = -std::numeric_limits<T>::max();
			for (int c_row = 0; c_row < m_Rows; ++c_row) {
				for (int c_column = 0; c_column < m_Columns; ++c_column) {
					const T value = Get(c_row, c_column);
					if (value > max) {
						max = value;
						r = c_row;
						c = c_column;
					}

				}
			}
			return max;
		}
		T Max(int& r, int& c, int fromCol, int toCol) const
		{
			T max = -std::numeric_limits<T>::max();
			for (int c_row = 0; c_row < m_Rows; ++c_row) {
				for (int c_column = fromCol; c_column < toCol; ++c_column) {
					const T value = Get(c_row, c_column);
					if (value > max) {
						max = value;
						r = c_row;
						c = c_column;
					}

				}
			}
			return max;
		}
		T MaxOfRow(int r) const
		{
			T max = -FLT_MAX;
			for (int c_column = 0; c_column < m_Columns; ++c_column) {
				T value = Get(r, c_column);
				if (value > max) {
					max = value;

				}
			}
			return max;
		}


		void Print() const
		{
			for (int c_row = 0; c_row < m_Rows; ++c_row) {
				for (int c_column = 0; c_column < m_Columns; ++c_column) {
					const T value = Get(c_row, c_column);
					printf("%.3f\t", value);
				}
				printf("\n");
			}
		}

		void MakeFile(const std::string& filePath, std::ios_base::openmode openMode = std::ios::app) const
		{
			//Checking if file already exists
#ifdef _DEBUG
//https://stackoverflow.com/questions/2324658/how-to-determine-the-version-of-the-c-standard-used-by-the-compiler
#if __cplusplus > 201103L //If using c++17 or higher
			{
				std::filesystem::path f{ "file.txt" };
				if (!std::filesystem::exists(f)) {
					__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
						" already exists", LogVerbosity::error);
				}
			}
#else
			{
				std::ifstream file(filePath);
				if (file.is_open()) {
					__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
						" already exists", LogVerbosity::error);
				}
			}
#endif
#endif

			std::ofstream file{};
			file.open(filePath, openMode);

			if (file)
			{
				if (file.is_open())
				{
					std::string toWrite{};
					toWrite += "\n# BotBrain Matrix\n";
					toWrite += "Rows count " + std::to_string(m_Rows) + '\n';
					toWrite += "Column count " + std::to_string(m_Columns) + "\n\n";

					for (int i{0}; i < m_Rows; ++i)
					{
						toWrite += "M ";
						for (int j{0}; j < m_Columns; ++j)
						{
							toWrite += std::to_string(Get(i, j)) + " ";
						}
						toWrite += '\n';
					}
					toWrite += "end";
					file.write(toWrite.c_str(), toWrite.size());
					file.close();
				}
#ifdef _DEBUG
				else
				{
					__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
						" failed to open maybe its open in another program?", LogVerbosity::warning);
				}
#endif
			}
#ifdef _DEBUG
			else
			{
				__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
					" not found", LogVerbosity::warning);
			}
#endif
		}

		void parseFile(const std::string& filePath)
		{
#ifdef _DEBUG
			__LOG__("Matrix file parsing started...");
#endif

			//Clear all data
			delete[] m_Data;
			m_Data = nullptr;
			m_Size = 0;
			m_Rows = 0;
			m_Columns = 0;

			std::ifstream file{};
			file.open(filePath);
			if (file)
			{
				if (file.is_open())
				{
					int currentRow{};
					std::string sCommand;
					// start a while iteration ending when the end of file is reached (ios::eof)
					while (!file.eof())
					{
						//read the first word of the string, use the >> operator (istream::operator>>) 
						file >> sCommand;
						//use conditional statements to process the different commands	
						if (sCommand == "#")
						{
							//Ignore Comment
						}
						else if (sCommand == "Rows")
						{
							file.ignore(6);
							int row;
							file >> row;
							m_Rows = row;
						}
						else if (sCommand == "Column")
						{
							file.ignore(6);
							int column;
							file >> column;
							m_Columns = column;
							m_Data = new T[m_Rows * m_Columns];
							m_Size = m_Rows * m_Columns;
						}
						else if (sCommand == "M")
						{
							T data;
							for (int col{0}; col < m_Columns; ++col)
							{
								file >> data;
								Set(currentRow, col, data);
							}
							++currentRow;
						}
						//read till end of line and ignore all remaining chars
						file.ignore(1000, '\n');
					}
#ifdef _DEBUG
					__LOG__("Matrix file parsing ended");
#endif
					file.close();
				}
#ifdef _DEBUG
				else
				{
					__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
						" failed to open maybe its open in another program?", LogVerbosity::warning);
				}
#endif
			}
#ifdef _DEBUG
			else
			{
				__LOGV__(std::string(strrchr(filePath.c_str(), '/') ? strrchr(filePath.c_str(), '/') + 1 : filePath.c_str()) +
					" not found", LogVerbosity::warning);
			}
#endif
		}
	};
}
#endif

