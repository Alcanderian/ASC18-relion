/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
#ifndef COMPLEX_H_
#define COMPLEX_H_
#include <iostream>
#include <cmath>
#include "src/macros.h"

class Complex
{

    public:

    RFLOAT real;
    RFLOAT imag;

    // Constructor
    Complex(RFLOAT _r = 0.0, RFLOAT _i = 0.0);

    Complex operator+(Complex &op);
    void operator+=(Complex &op);

    Complex operator-(Complex &op);
    void operator-=(Complex &op);

    // Complex operator*(Complex &op);

    void operator*=(RFLOAT op);

    // Complex operator*(RFLOAT op);

    Complex operator/(Complex &op);

    Complex operator/(RFLOAT op);

    void operator/=(RFLOAT op);

    // Complex conjugated
    Complex conj();

    // Abs value: sqrt(real*real+imag*imag)
    RFLOAT abs();

    // Norm value: real*real+imag*imag
    RFLOAT norm();

    // Phase angle: atan2(imag,real)
    RFLOAT arg();


};


static inline Complex operator+(const Complex& lhs, const Complex& rhs)
{
    return Complex(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

static inline Complex operator-(const Complex& lhs, const Complex& rhs)
{
    return Complex(lhs.real - rhs.real, lhs.imag - rhs.imag);

}

static inline Complex operator*(const Complex& lhs, const Complex& rhs)
{
    return Complex((lhs.real * rhs.real) - (lhs.imag * rhs.imag), (lhs.real * rhs.imag) + (lhs.imag * rhs.real));
}

static inline Complex operator*(const Complex& lhs, const RFLOAT& val)
{
    return Complex(lhs.real * val , lhs.imag * val);
}

static inline Complex operator*(const RFLOAT& val, const Complex& rhs)
{
    return Complex(rhs.real * val , rhs.imag * val);
}

static inline void operator+=(Complex& lhs, const Complex& rhs)
{
    lhs.real += rhs.real;
    lhs.imag += rhs.imag;
}
static inline void operator-=(Complex& lhs, const Complex& rhs)
{
    lhs.real -= rhs.real;
    lhs.imag -= rhs.imag;
}

static inline Complex conj(const Complex& op)
{
    return Complex(op.real, -op.imag);
}

static inline RFLOAT abs(const Complex& op)
{
    return sqrt(op.real*op.real + op.imag*op.imag);
}

static inline RFLOAT arg(const Complex& op)
{
    return atan2(op.imag, op.real);
}

static inline RFLOAT norm(const Complex& op)
{
    return op.real*op.real + op.imag*op.imag;
}
//static inline Complex conj(const Complex& op);
//static inline RFLOAT abs(const Complex& op);
//static inline RFLOAT norm(const Complex& op);
//static inline RFLOAT arg(const Complex& op);

//static inline Complex operator+(const Complex& lhs, const Complex& rhs);
//static inline Complex operator-(const Complex& lhs, const Complex& rhs);
//static inline Complex operator*(const Complex& lhs, const Complex& rhs);
//static inline Complex operator*(const Complex& lhs, const RFLOAT& val);
//static inline Complex operator*(const RFLOAT& val, const Complex& rhs);
//static inline Complex operator/(const Complex& lhs, const RFLOAT& val);

//static inline void operator+=(Complex& lhs, const Complex& rhs);
//static inline void operator-=(Complex& lhs, const Complex& rhs);

#endif
