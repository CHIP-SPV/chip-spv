/*
 * This is counterpart to hipcl_mathlib.hh
 * ATM it can't be used right after compilation because of a problem with mangling.
 *
 * HIP with default AS set to 4 mangles functions with pointer args to:
 *   float @_Z13opencl_sincosfPf(float, float addrspace(4)*)
 * while OpenCL code compiled for SPIR mangles to either
 *   float @_Z6sincosfPU3AS4f(float, float addrspace(4)*)
 * or
 *   float @_Z6sincosfPf(float, float *)
*/

#define NON_OVLD
#define OVLD __attribute__((overloadable))
//#define AI __attribute__((always_inline))
#define EXPORT NON_OVLD

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define DEFAULT_AS
#define PRIVATE_AS __private

#define CL_NAME(N) opencl_##N
#define CL_NAME2(N, S) opencl__##N##_##S

#define CL_NAME_MANGLED_ATOM(NAME, S) CL_NAME2(atomic_##NAME, S)

#define DEFOCML_OPENCL1F(NAME) \
float OVLD NAME(float f); \
double OVLD NAME(double f); \
half OVLD NAME(half f); \
half2 OVLD NAME(half2 f); \
EXPORT float CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT double CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return NAME(x); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return NAME(x); }


#define DEFOCML_OPENCL2F(NAME) \
float OVLD NAME(float x, float y); \
double OVLD NAME(double x, double y); \
half OVLD NAME(half x, half y); \
half2 OVLD NAME(half2 x, half2 y); \
EXPORT float CL_NAME2(NAME, f)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y) { return NAME(x, y); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y) { return NAME(x, y); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y) { return NAME(x, y); }

/*****************************************************************/

#define DEF_OPENCL1F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT double CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return NAME(x); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return NAME(x); }

#define DEF_OPENCL2F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y) { return NAME(x, y); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y) { return NAME(x, y); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y) { return NAME(x, y); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y) { return NAME(x, y); }

#define DEF_OPENCL3F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y, float z) { return NAME(x, y, z); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y, double z) { return NAME(x, y, z); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y, half z) { return NAME(x, y, z); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y, half2 z) { return NAME(x, y, z); }

#define DEF_OPENCL4F(NAME) \
EXPORT float CL_NAME2(NAME, f)(float x, float y, float z, float w) { return NAME(x, y, z, w); } \
EXPORT double CL_NAME2(NAME, d)(double x, double y, double z, double w) { return NAME(x, y, z, w); } \
EXPORT half CL_NAME2(NAME, h)(half x, half y, half z, half w) { return NAME(x, y, z, w); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x, half2 y, half2 z, half2 w) { return NAME(x, y, z, w); }

#define DEF_OPENCL1B(NAME) \
EXPORT int CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT long CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT half CL_NAME2(NAME, h)(half x) { return as_half(convert_short(NAME(x))); } \
EXPORT half2 CL_NAME2(NAME, h2)(half2 x) { return as_half2(NAME(x)); }


#define DEF_OPENCL1INT(NAME) \
EXPORT int CL_NAME2(NAME, f)(float x) { return NAME(x); } \
EXPORT int CL_NAME2(NAME, d)(double x) { return NAME(x); } \
EXPORT int CL_NAME2(NAME, h)(half x) { return NAME(x); }
//EXPORT int CL_NAME(NAME, h2)(half2 x) { return NAME(x); }

#define DEF_OPENCL1F_NATIVE(NAME) \
EXPORT float CL_NAME2(NAME##_native, f)(float x) { return native_##NAME(x); }

// +7
DEF_OPENCL1F(acos)
DEF_OPENCL1F(asin)
DEF_OPENCL1F(acosh)
DEF_OPENCL1F(asinh)
DEF_OPENCL1F(atan)
DEF_OPENCL2F(atan2)
DEF_OPENCL1F(atanh)
DEF_OPENCL1F(cbrt)
DEF_OPENCL1F(ceil)

DEF_OPENCL2F(copysign)

DEF_OPENCL1F(cos)
DEF_OPENCL1F(cosh)
DEF_OPENCL1F(cospi)

// OCML
float OVLD i0(float f);
double OVLD i0(double f);
EXPORT float CL_NAME2(cyl_bessel_i0, f)(float x) { return i0(x); }
EXPORT double CL_NAME2(cyl_bessel_i0, d)(double x) { return i0(x); }
float OVLD i1(float f);
double OVLD i1(double f);
EXPORT float CL_NAME2(cyl_bessel_i1, f)(float x) { return i1(x); }
EXPORT double CL_NAME2(cyl_bessel_i1, d)(double x) { return i1(x); }


DEF_OPENCL1F(erfc)
DEF_OPENCL1F(erf)

// OCML
DEFOCML_OPENCL1F(erfcinv)
DEFOCML_OPENCL1F(erfcx)
DEFOCML_OPENCL1F(erfinv)

DEF_OPENCL1F(exp10)
DEF_OPENCL1F(exp2)
DEF_OPENCL1F(exp)
DEF_OPENCL1F(expm1)
DEF_OPENCL1F(fabs)
DEF_OPENCL2F(fdim)
DEF_OPENCL1F(floor)

DEF_OPENCL3F(fma)

DEF_OPENCL2F(fmax)
DEF_OPENCL2F(fmin)
DEF_OPENCL2F(fmod)

float OVLD frexp(float f, PRIVATE_AS int *i);
double OVLD frexp(double f, PRIVATE_AS int *i);
EXPORT float CL_NAME2(frexp, f)(float x, DEFAULT_AS int *i) {
  int tmp;
  float ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT float CL_NAME2(frexp, d)(double x, DEFAULT_AS int *i) {
  int tmp;
  double ret = frexp(x, &tmp);
  *i = tmp;
  return ret;
}

DEF_OPENCL2F(hypot)
DEF_OPENCL1INT(ilogb)

DEF_OPENCL1B(isfinite)
DEF_OPENCL1B(isinf)
DEF_OPENCL1B(isnan)

DEFOCML_OPENCL1F(j0)
DEFOCML_OPENCL1F(j1)

float OVLD ldexp(float f, int k);
double OVLD ldexp(double f, int k);
EXPORT float CL_NAME2(ldexp, f)(float x, int k) { return ldexp(x, k); }
EXPORT double CL_NAME2(ldexp, d)(double x, int k) { return ldexp(x, k); }

float OVLD lgamma(float f, PRIVATE_AS int *signp);
double OVLD lgamma(double f, PRIVATE_AS int *signp);
EXPORT float CL_NAME2(lgamma, f)(float x) { int sign; return lgamma(x, &sign); }
EXPORT double CL_NAME2(lgamma, d)(double x) { int sign; return lgamma(x, &sign); }

DEF_OPENCL1F(log10)
DEF_OPENCL1F(log1p)
DEF_OPENCL1F(log2)
DEF_OPENCL1F(logb)
DEF_OPENCL1F(log)

// modf
float OVLD modf(float f, PRIVATE_AS float *i);
double OVLD modf(double f, PRIVATE_AS double *i);
EXPORT float CL_NAME2(modf, f)(float x, DEFAULT_AS float *i) {
  float tmp;
  float ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}
EXPORT float CL_NAME2(modf, d)(double x, DEFAULT_AS double *i) {
  double tmp;
  double ret = modf(x, &tmp);
  *i = tmp;
  return ret;
}

// OCML
DEFOCML_OPENCL1F(nearbyint)
DEFOCML_OPENCL2F(nextafter)

float OVLD length(float4 f);
double OVLD length(double4 f);
EXPORT float CL_NAME2(norm4d, f)(float x, float y, float z, float w) { float4 temp = (float4)(x, y, z, w); return length(temp); }
EXPORT double CL_NAME2(norm4d, d)(double x, double y, double z, double w) { double4 temp = (double4)(x, y, z, w); return length(temp); }
EXPORT float CL_NAME2(norm3d, f)(float x, float y, float z) { float4 temp = (float4)(x, y, z, 0.0f); return length(temp); }
EXPORT double CL_NAME2(norm3d, d)(double x, double y, double z) { double4 temp = (double4)(x, y, z, 0.0); return length(temp); }


// OCML ncdf / ncdfinv
DEFOCML_OPENCL1F(normcdf)
DEFOCML_OPENCL1F(normcdfinv)

DEF_OPENCL2F(pow)
DEF_OPENCL2F(remainder)
// OCML
DEFOCML_OPENCL1F(rcbrt)

// remquo
float OVLD remquo(float x,   float y,  PRIVATE_AS int *quo);
double OVLD remquo(double x, double y, PRIVATE_AS int *quo);
EXPORT float CL_NAME2(remquo, f)(float x, float y, DEFAULT_AS int *quo) {
  int tmp;
  float rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}
EXPORT float CL_NAME2(remquo, d)(double x, double y, DEFAULT_AS int *quo) {
  int tmp;
  double rem = remquo(x, y, &tmp);
  *quo = tmp;
  return rem;
}

// OCML
DEFOCML_OPENCL2F(rhypot)

// OCML rlen3 / rlen4
float OVLD rlen4(float4 f);
double OVLD rlen4(double4 f);
float OVLD rlen3(float3 f);
double OVLD rlen3(double3 f);

EXPORT float CL_NAME2(rnorm4d, f)(float x, float y, float z, float w) { float4 temp = (float4)(x, y, z, w); return rlen4(temp); }
EXPORT double CL_NAME2(rnorm4d, d)(double x, double y, double z, double w) { double4 temp = (double4)(x, y, z, w); return rlen4(temp); }
EXPORT float CL_NAME2(rnorm3d, f)(float x, float y, float z) { float3 temp = (float3)(x, y, z); return rlen3(temp); }
EXPORT double CL_NAME2(rnorm3d, d)(double x, double y, double z) { double3 temp = (double3)(x, y, z); return rlen3(temp); }


DEF_OPENCL1F(round)
DEF_OPENCL1F(rsqrt)

// OCML
float OVLD scalbn(float f, int k);
double OVLD scalbn(double f, int k);
EXPORT float CL_NAME2(scalbn, f)(float x, int k) { return scalbn(x, k); }
EXPORT double CL_NAME2(scalbn, d)(double x, int k) { return scalbn(x, k); }
// OCML
DEFOCML_OPENCL2F(scalb)

DEF_OPENCL1B(signbit)

DEF_OPENCL1F(sin)
DEF_OPENCL1F(sinh)
DEF_OPENCL1F(sinpi)
DEF_OPENCL1F(sqrt)
DEF_OPENCL1F(tan)
DEF_OPENCL1F(tanh)
DEF_OPENCL1F(tgamma)
DEF_OPENCL1F(trunc)



// sincos
float OVLD sincos(float x, PRIVATE_AS float *cosval);
double OVLD sincos(double x, PRIVATE_AS double *cosval);

EXPORT float CL_NAME2(sincos, f)(float x, DEFAULT_AS float *cos) {
  PRIVATE_AS float tmp;
  PRIVATE_AS float sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

EXPORT float CL_NAME2(sincos, d)(double x, DEFAULT_AS double *cos) {
  PRIVATE_AS double tmp;
  PRIVATE_AS double sin = sincos(x, &tmp);
  *cos = tmp;
  return sin;
}

// OCML
DEFOCML_OPENCL1F(y0)
DEFOCML_OPENCL1F(y1)

/* native */

DEF_OPENCL1F_NATIVE(cos)
DEF_OPENCL1F_NATIVE(sin)
DEF_OPENCL1F_NATIVE(tan)

DEF_OPENCL1F_NATIVE(exp10)
DEF_OPENCL1F_NATIVE(exp)

DEF_OPENCL1F_NATIVE(log10)
DEF_OPENCL1F_NATIVE(log2)
DEF_OPENCL1F_NATIVE(log)

/* other */

EXPORT void CL_NAME(local_barrier)() { barrier(CLK_LOCAL_MEM_FENCE); }

EXPORT void CL_NAME(local_fence)() { mem_fence(CLK_LOCAL_MEM_FENCE); }

EXPORT void CL_NAME(global_fence)() { mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); }

/* memory routines */

// sets size bytes of the memory pointed to by ptr to value
// interpret ptr as a unsigned char so that it writes as bytes
EXPORT void* CL_NAME(memset)(DEFAULT_AS void* ptr, int value, size_t size) {
  volatile unsigned char* temporary = ptr;

  for(int i=0;i<size;i++)
    temporary[i] = value;
  
    return ptr;
}

EXPORT void* CL_NAME(memcpy)(DEFAULT_AS void *dest, DEFAULT_AS const void * src, size_t n) {
  volatile unsigned char* temporary_dest = dest;
  volatile const unsigned char* temporary_src = src;

  for(int i=0;i<n;i++)
    temporary_dest[i] = temporary_src[i];

  return dest;
}

/**********************************************************************/

EXPORT uint CL_NAME2(popcount, ui)(uint var) {
  return popcount(var);
}

EXPORT ulong CL_NAME2(popcount, ul)(ulong var) {
  return popcount(var);
}


EXPORT int CL_NAME2(clz, i)(int var) {
  return clz(var);
}

EXPORT long CL_NAME2(clz, li)(long var) {
  return clz(var);
}

EXPORT int CL_NAME2(ctz, i)(int var) {
  return ctz(var);
}

EXPORT long CL_NAME2(ctz, li)(long var) {
  return ctz(var);
}


EXPORT int CL_NAME2(hadd, i)(int x, int y) {
  return hadd(x, y);
}

EXPORT int CL_NAME2(rhadd, i)(int x, int y) {
  return hadd(x, y);
}

EXPORT uint CL_NAME2(uhadd, ui)(uint x, uint y) {
  return hadd(x, y);
}

EXPORT uint CL_NAME2(urhadd, ui)(uint x, uint y) {
  return hadd(x, y);
}


EXPORT int CL_NAME2(mul24, i)(int x, int y) {
  return mul24(x, y);
}

EXPORT int CL_NAME2(mulhi, i)(int x, int y) {
  return mul_hi(x, y);
}

EXPORT long CL_NAME2(mul64hi, li)(long x, long y) {
  return mul_hi(x, y);
}


EXPORT uint CL_NAME2(umul24, ui)(uint x, uint y) {
  return mul24(x, y);
}

EXPORT uint CL_NAME2(umulhi, ui)(uint x, uint y) {
  return mul_hi(x, y);
}

EXPORT ulong CL_NAME2(umul64hi, uli)(ulong x, ulong y) {
  return mul_hi(x, y);
}




/**********************************************************************/

#define DEF_OPENCL_ATOMIC2(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(DEFAULT_AS int *address, int i) {          \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, i);                                             \
    else {                                                                     \
      volatile local int *li = to_local(address);                              \
      if (li)                                                                  \
        return atomic_##NAME(li, i);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address, uint ui) {                                     \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, ui);                                            \
    else {                                                                     \
      volatile local uint *li = to_local(address);                             \
      if (li)                                                                  \
        return atomic_##NAME(li, ui);                                          \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address,                                               \
      ulong ull) {                                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi, ull);                                             \
    else {                                                                     \
      volatile local ulong *li =                                               \
          to_local((DEFAULT_AS ulong *)address);                               \
      if (li)                                                                  \
        return atom_##NAME(li, ull);                                           \
      else                                                                     \
        return 0;                                                              \
    }                                                                          \
  };

DEF_OPENCL_ATOMIC2(add)
DEF_OPENCL_ATOMIC2(sub)
DEF_OPENCL_ATOMIC2(xchg)
DEF_OPENCL_ATOMIC2(min)
DEF_OPENCL_ATOMIC2(max)
DEF_OPENCL_ATOMIC2(and)
DEF_OPENCL_ATOMIC2(or)
DEF_OPENCL_ATOMIC2(xor)

#define DEF_OPENCL_ATOMIC1(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(DEFAULT_AS int *address) {                 \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local int *li = to_local(address);                                \
    if (li)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address) {                                              \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi);                                                \
    volatile local uint *li = to_local(address);                               \
    if (li)                                                                    \
      return atomic_##NAME(li);                                                \
    return 0;                                                                  \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address) {                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi);                                                  \
    volatile local ulong *li = to_local((DEFAULT_AS ulong *)address);          \
    if (li)                                                                    \
      return atom_##NAME(li);                                                  \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC1(inc)
DEF_OPENCL_ATOMIC1(dec)

#define DEF_OPENCL_ATOMIC3(NAME)                                               \
  int CL_NAME_MANGLED_ATOM(NAME, i)(                                           \
      DEFAULT_AS int *address, int cmp, int val) {                             \
    volatile global int *gi = to_global(address);                              \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local int *li = to_local(address);                                \
    if (li)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  uint CL_NAME_MANGLED_ATOM(NAME, u)(                                          \
      DEFAULT_AS uint *address, uint cmp,                                      \
      uint val) {                                                              \
    volatile global uint *gi = to_global(address);                             \
    if (gi)                                                                    \
      return atomic_##NAME(gi, cmp, val);                                      \
    volatile local uint *li = to_local(address);                               \
    if (li)                                                                    \
      return atomic_##NAME(li, cmp, val);                                      \
    return 0;                                                                  \
  };                                                                           \
  ulong CL_NAME_MANGLED_ATOM(NAME, l)(                                         \
      DEFAULT_AS ulong *address, ulong cmp,                                    \
      ulong val) {                                                             \
    volatile global ulong *gi =                                                \
        to_global((DEFAULT_AS ulong *)address);                                \
    if (gi)                                                                    \
      return atom_##NAME(gi, cmp, val);                                        \
    volatile local ulong *li = to_local((DEFAULT_AS ulong *)address);          \
    if (li)                                                                    \
      return atom_##NAME(li, cmp, val);                                        \
    return 0;                                                                  \
  };

DEF_OPENCL_ATOMIC3(cmpxchg)

/* This code adapted from AMD's HIP sources */

static OVLD float atomic_add_f(volatile local float *address, float val) {
  volatile local uint *uaddr = (volatile local uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

static OVLD double atom_add_d(volatile local double *address, double val) {
  volatile local ulong *uaddr = (volatile local ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVLD float atomic_exch_f(volatile local float *address, float val) {
  return as_float(atomic_xchg((volatile local uint *)(address), as_uint(val)));
}

static OVLD float atomic_add_f(volatile global float *address, float val) {
  volatile global uint *uaddr = (volatile global uint *)address;
  uint old = *uaddr;
  uint r;

  do {
    r = old;
    old = atomic_cmpxchg(uaddr, r, as_uint(val + as_float(r)));
  } while (r != old);

  return as_float(r);
}

static OVLD double atom_add_d(volatile global double *address, double val) {
  volatile global ulong *uaddr = (volatile global ulong *)address;
  ulong old = *uaddr;
  ulong r;

  do {
    r = old;
    old = atom_cmpxchg(uaddr, r, as_ulong(val + as_double(r)));
  } while (r != old);

  return as_double(r);
}

static OVLD float atomic_exch_f(volatile global float *address, float val) {
  return as_float(atomic_xchg((volatile global uint *)(address), as_uint(val)));
}

static OVLD uint atomic_inc2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r+1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_dec2_u(volatile local uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r-1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_inc2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, ((r >= val) ? 0 : (r+1)));
  } while (r != old);

  return r;
}

static OVLD uint atomic_dec2_u(volatile global uint *address, uint val) {
  uint old = *address;
  uint r;
  do {
    r = old;
    old = atom_cmpxchg(address, r, (((r == 0) || (r > val)) ? val : (r-1)));
  } while (r != old);

  return r;
}

EXPORT float CL_NAME_MANGLED_ATOM(add, f)(DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_add_f(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return atomic_add_f(li, val);
  return 0;
}

EXPORT double CL_NAME_MANGLED_ATOM(add, d)(DEFAULT_AS double *address,
                                  double val) {
  volatile global double *gi = to_global((DEFAULT_AS double *)address);
  if (gi)
    return atom_add_d(gi, val);
  volatile local double *li = to_local((DEFAULT_AS double *)address);
  if (li)
    return atom_add_d(li, val);
  return 0;
}

EXPORT float CL_NAME_MANGLED_ATOM(exch, f)(DEFAULT_AS float *address,
                                 float val) {
  volatile global float *gi = to_global(address);
  if (gi)
    return atomic_exch_f(gi, val);
  volatile local float *li = to_local(address);
  if (li)
    return atomic_exch_f(li, val);
  return 0;
}

EXPORT uint CL_NAME_MANGLED_ATOM(inc2, u)(DEFAULT_AS uint *address,
                                 uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return atomic_inc2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return atomic_inc2_u(li, val);
  return 0;
}

EXPORT uint CL_NAME_MANGLED_ATOM(dec2, u)(DEFAULT_AS uint *address,
                                 uint val) {
  volatile global uint *gi = to_global((DEFAULT_AS uint *)address);
  if (gi)
    return atomic_dec2_u(gi, val);
  volatile local uint *li = to_local((DEFAULT_AS uint *)address);
  if (li)
    return atomic_dec2_u(li, val);
  return 0;
}
/**********************************************************************/

int OVLD intel_sub_group_shuffle(int var, uint srcLane);
float OVLD intel_sub_group_shuffle(float var, uint srcLane);
int OVLD intel_sub_group_shuffle_xor(int var, uint value);
float OVLD intel_sub_group_shuffle_xor(float var, uint value);
int OVLD intel_sub_group_shuffle_up(int prev, int curr, uint delta);
float OVLD intel_sub_group_shuffle_up(float prev, float curr, uint delta);
int OVLD intel_sub_group_shuffle_down(int prev, int curr, uint delta);
float OVLD intel_sub_group_shuffle_down(float prev, float curr, uint delta);

EXPORT int CL_NAME2(shfl, i)(int var, int srcLane) {
  return intel_sub_group_shuffle(var, srcLane);
};
EXPORT float CL_NAME2(shfl, f)(float var, int srcLane) {
  return intel_sub_group_shuffle(var, srcLane);
};

EXPORT int CL_NAME2(shfl_xor, i)(int var, int value) {
  return intel_sub_group_shuffle_xor(var, value);
};
EXPORT float CL_NAME2(shfl_xor, f)(float var, int value) {
  return intel_sub_group_shuffle_xor(var, value);
};

EXPORT int CL_NAME2(shfl_up, i)(int var, uint delta) {
  int tmp = 0;
  int tmp2 = intel_sub_group_shuffle_down(tmp, var, delta);
  return intel_sub_group_shuffle_up(tmp2, var, delta);
};
EXPORT float CL_NAME2(shfl_up, f)(float var, uint delta) {
  float tmp = 0;
  float tmp2 = intel_sub_group_shuffle_down(tmp, var, delta);
  return intel_sub_group_shuffle_up(tmp2, var, delta);
};

EXPORT int CL_NAME2(shfl_down, i)(int var, uint delta) {
  int tmp = 0;
  int tmp2 = intel_sub_group_shuffle_up(var, tmp, delta);
  return intel_sub_group_shuffle_down(var, tmp2, delta);
};
EXPORT float CL_NAME2(shfl_down, f)(float var, uint delta) {
  float tmp = 0;
  float tmp2 = intel_sub_group_shuffle_up(var, tmp, delta);
  return intel_sub_group_shuffle_down(var, tmp2, delta);
};


int CL_NAME(group_all)(int pred) { return sub_group_all(pred); }
int CL_NAME(group_any)(int pred) { return sub_group_any(pred); }
ulong CL_NAME(group_ballot)(int pred) { return sub_group_reduce_add(pred ? (ulong)1 << get_sub_group_local_id() : 0); }

EXPORT int CL_NAME2(max, i)(int a, int b) { return max(a, b); }

EXPORT uint CL_NAME2(max, u)(uint a, uint b) { return max(a, b); }

EXPORT long CL_NAME2(max, l)(long a, long b) { return max(a, b); }

EXPORT ulong CL_NAME2(max, ul)(ulong a, ulong b) { return max(a, b); }

EXPORT int CL_NAME2(min, i)(int a, int b) { return min(a, b); }

EXPORT uint CL_NAME2(min, u)(uint a, uint b) { return min(a, b); }

EXPORT long CL_NAME2(min, l)(long a, long b) { return min(a, b); }

EXPORT ulong CL_NAME2(min, ul)(ulong a, ulong b) { return min(a, b); }

EXPORT int CL_NAME2(abs, i)(int a) { return abs(a); }
EXPORT long CL_NAME2(abs, l)(long a) { return abs(a); }

typedef struct {
  intptr_t  image;
  intptr_t  sampler;
} *hipTextureObject_t;

EXPORT float CL_NAME2(tex2D, f)(hipTextureObject_t textureObject,
				float x, float y) {
  return read_imagef(
    __builtin_astype(textureObject->image, read_only image2d_t),
    __builtin_astype(textureObject->sampler, sampler_t),
    (float2)(x, y)).x;
}
