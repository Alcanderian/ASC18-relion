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
#include "src/complex.h"

// Constructor with two arguments
Complex::Complex(RFLOAT _r, RFLOAT _i)
{
    real = _r;
    imag = _i;
}

Complex Complex::operator+ (Complex &op)
{
    return Complex(real + op.real, imag + op.imag);
}

void Complex::operator+= (Complex &op)
{
    real += op.real;
    imag += op.imag;
}

Complex Complex::operator- (Complex &op)
{
    return Complex(real - op.real, imag - op.imag);
}
void Complex::operator-= (Complex &op)
{
    real -= op.real;
    imag -= op.imag;
}

// Complex Complex::operator* (Complex &op)
// {
//     return Complex((real * op.real) - (imag * op.imag), (real * op.imag) + (imag * op.real));
// }

// Complex Complex::operator* (RFLOAT op)
// {
//     return Complex(real*op, imag*op);
// }

void Complex::operator*= (RFLOAT op)
{
    real *= op;
    imag *= op;
}

Complex Complex::operator/(RFLOAT op)
{
    return Complex(real/op, imag/op);
}

Complex Complex::operator/(Complex &op)
{
    RFLOAT cd = op.norm();
    RFLOAT realval = real*op.real + imag*op.imag;
    RFLOAT imagval = imag*op.real - real*op.imag;
    return Complex(realval/cd, imagval/cd);
}

void Complex::operator/=(RFLOAT op)
{
    real /= op;
    imag /= op;
}

Complex Complex::conj()
{
    return Complex(real, -imag);
}


RFLOAT Complex::abs()
{
    return sqrt(real*real + imag*imag);
}

RFLOAT Complex::norm()
{
    return real*real + imag*imag;
}


RFLOAT Complex::arg()
{
    return atan2(imag, real);
}