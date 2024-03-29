#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;

VectorXd SolvePALU(const MatrixXd& A,
                   const VectorXd& b)
{
    VectorXd solutionPALU = A.fullPivLu().solve(b);
    return solutionPALU;
}

VectorXd SolveQR(const MatrixXd& A,
                 const VectorXd& b)
{
    VectorXd solutionQR = A.householderQr().solve(b);
    return solutionQR;
}

void Errors(const MatrixXd& A,
                   const VectorXd& b,
                   const VectorXd& solution,
                   double& errRelPALU,
                   double& errRelQR)
{
    errRelPALU = (SolvePALU(A,b)-solution).norm()/solution.norm();
    errRelQR = (SolveQR(A,b)-solution).norm()/solution.norm();
}

bool CheckSV(const MatrixXd& A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16)
    {
        return false;
    }

    return true;
}

int main()
{
    Vector2d x(-1.0e+0, -1.0e+00);

    Matrix2d A1{{5.547001962252291e-01, -3.770900990025203e-02}, { 8.320502943378437e-01,
                                                                  -9.992887623566787e-01}};
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);


    Matrix2d A2{{5.547001962252291e-01, -5.540607316466765e-01}, { 8.320502943378437e-01,
                                                                  -8.324762492991313e-01}};
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);


    Matrix2d A3{{5.547001962252291e-01, -5.547001955851905e-01}, { 8.320502943378437e-01,
                                                                  -8.320502947645361e-01}};
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);

    if (!CheckSV(A1))
    {
        cout << "System 1 is unsolvable." << endl;
        return 1;
    }

    if (!CheckSV(A2))
    {
        cout << "System 2 is unsolvable." << endl;
        return 1;
    }

    if (!CheckSV(A3))
    {
        cout << "System 3 is unsolvable." << endl;
        return 1;
    }

    double erPALU1 = 0;
    double erQR1 = 0;
    Errors(A1, b1, x, erPALU1, erQR1);
    VectorXd solutionPALU1 = SolvePALU(A1,b1);
    VectorXd solutionQR1 = SolveQR(A1, b1);
    cout << scientific << setprecision(16) << "1: PALU: [ " << solutionPALU1.transpose() << " ], QR: [ " << solutionQR1.transpose() << " ]'" << endl;
    cout << "Relative Error PALU: " << erPALU1 << ", Relative Error QR: " << erQR1 << endl;

    double erPALU2 = 0;
    double erQR2 = 0;
    Errors(A2, b2, x, erPALU2, erQR2);
    VectorXd solutionPALU2 = SolvePALU(A2,b2);
    VectorXd solutionQR2 = SolveQR(A2, b2);
    cout << "2: PALU: [ " << solutionPALU2.transpose() << " ], QR: [ " << solutionQR2.transpose() << " ]'" << endl;
    cout << "Relative Error PALU: " << erPALU1 << ", Relative Error QR: " << erQR1 << endl;

    double erPALU3 = 0;
    double erQR3 = 0;
    Errors(A3, b3, x, erPALU3, erQR3);
    VectorXd solutionPALU3 = SolvePALU(A3,b3);
    VectorXd solutionQR3 = SolveQR(A3, b3);
    cout << "3: PALU: [ " << solutionPALU3.transpose() << " ], QR: [ " << solutionQR3.transpose() << " ]'" << endl;
    cout << "Relative Error PALU: " << erPALU3 << ", Relative Error QR: " << erQR3 << endl;

    return 0;
}
