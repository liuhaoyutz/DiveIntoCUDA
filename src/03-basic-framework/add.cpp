#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;
void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

// x, y两个数组对应位置相加，得到新的数组z
int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    // 将数组x的元素值初始化为1.23，数组y的元素值初始化为2.34
    for (int n = 0; n < N; ++n)
    {
        x[n] = a;
        y[n] = b;
    }

    // 数组x, y对应位置元素相加，赋值给数组z
    add(x, y, z, N);
    // 验证结果是否正确，z的元素值应该是3.57
    check(z, N);

    free(x);
    free(y);
    free(z);
    return 0;
}

void add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

// 检查结果是否正确，z[n]-c 结果应该为0
void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}

