#ifndef TA_MY_TEST_H_MATH
#define TA_MY_TEST_H_MATH

/* math */
#define PI 3.14159265358979323846
#define e  2.7182818284590452354
#define ln_2 0.69314718055994530942
#define ln_10 2.30258509299404568402
#define first_aim_money 1000000000.0f

#define fabs(a) ((a)>0?(a):(-(a)))

float ta_max(float a, float b);
double ta_pow(double a,int n);
double ta_eee(double x);
double ta_exp(double x);
float ta_rand();
int ta_floor(double x);
double ta_sqrt(double x);
double ta_ln(double x);
double ta_log(double a,double N);
double ta_sin(double x);
double ta_cos(double x);
double ta_tan(double x);

void reverse(char *str, int len);
int intToStr(int x, char str[], int d);
void ftoa(float n, char *res, int afterpoint);
void bubble_sort_top(float *arr, int len);

#endif /*TA_MY_TEST_H_MATH*/
