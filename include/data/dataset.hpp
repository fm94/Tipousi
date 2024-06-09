#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <random>
#include <vector>

namespace Tipousi
{
    namespace Data
    {
        class Dataset
        {
          public:
            Dataset(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y,
                    size_t batch_size);
            ~Dataset() = default;

            void shuffle();

            using DataPair = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;

            class Iterator
            {
              public:
                Iterator(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y,
                         size_t index, size_t batch_size);
                Iterator &operator++();
                bool      operator!=(const Iterator &other) const;
                DataPair  operator*() const;

              private:
                const Eigen::MatrixXf &m_X;
                const Eigen::MatrixXf &m_y;
                size_t                 m_index;
                size_t                 m_batch_size;
            };

            Iterator begin() const;
            Iterator end() const;

          private:
            Eigen::MatrixXf m_X;
            Eigen::MatrixXf m_y;
            size_t          m_batch_size;
        };
    };  // namespace Data
};      // namespace Tipousi