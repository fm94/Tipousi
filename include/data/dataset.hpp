#pragma once
#include <Eigen/Dense>
#include <vector>

namespace Tipousi
{
    namespace Data
    {
        class Dataset
        {
          public:
            Dataset(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
            ~Dataset() = default;

            using DataPair = std::pair<Eigen::MatrixXf, Eigen::MatrixXf>;

            class Iterator
            {
              public:
                Iterator(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
                         size_t index);
                Iterator &operator++();
                bool      operator!=(const Iterator &other) const;
                DataPair  operator*() const;

              private:
                const Eigen::MatrixXd &m_X;
                const Eigen::MatrixXd &m_y;
                size_t                 m_index;
            };

            Iterator begin() const;
            Iterator end() const;

          private:
            Eigen::MatrixXd m_X;
            Eigen::MatrixXd m_y;
        };
    };  // namespace Data
};      // namespace Tipousi