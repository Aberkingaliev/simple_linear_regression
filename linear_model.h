#pragma once
#include <vector>

class LinearModel {
    protected:
        std::vector<double> weights;
        double bias;

    public:
        LinearModel() = default;
        explicit LinearModel(int l);
        virtual void train(
            std::vector< std::vector<double> >& X,
            std::vector< double >& y,
            double learning_rate,
            double epochs
        ) = 0;
        virtual double predict(
            std::vector< double >& X
        ) = 0;

};

class LinearRegression : public LinearModel {
    public:
        LinearRegression(int l);

         void train(
            std::vector< std::vector<double> >& X,
            std::vector< double >& y,
            double learning_rate,
            double epochs
        ) override;
         double predict(
            std::vector< double >& X
        ) override;

};
